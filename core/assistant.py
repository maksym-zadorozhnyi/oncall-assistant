"""Ядро асистента: ask(питання, сесія) -> відповідь. Консоль, іспит і бот викликають саме її.

Цикл для одного питання:
1. Код збирає запит: системний промпт (з моментом «зараз»), опис функцій,
   кілька останніх реплік розмови і питання.
2. LLM відповідає текстом (кінець) або просить викликати функції (tool use).
3. Код виконує функції з core/tools.py. Параметр now підставляє код із сесії:
   LLM його не бачить, а якщо передасть — код ігнорує.
4. Результати (JSON) — назад у LLM. Кроки 2–4 — щонайбільше max_tool_rounds разів.
5. Код додає службові рядки («⚠ дані застарілі», «Дивився: …», «Момент: …»)
   і пише запис у журнал data/logs/assistant.jsonl.

LLM нічого не рахує: числа — лише з результатів функцій.

Консоль:
    python -m core.assistant --now "2026-11-10 08:00"   # момент в історії (місцевий час)
    python -m core.assistant                            # живий режим
    python -m core.assistant --verbose                  # ще й JSON кожної функції
    python -m core.assistant --show-prompt --now "2026-11-10 08:00"   # лише промпт, без LLM
    python -m core.assistant --show-tools               # лише опис функцій для LLM, без LLM
"""

import argparse
import copy
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from core import llm, tool_specs, tools
from core.llm import LLMError

CORE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CORE_DIR / "assistant.yaml"
PROMPT_PATH = CORE_DIR / "prompts" / "system.md"
LOG_PATH = CORE_DIR.parent / "data" / "logs" / "assistant.jsonl"   # у .gitignore

WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя"]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]
MOMENT_NOTE = "Це момент в історії: говори так, наче зараз — цей момент; даних після нього немає."
SHIFT_RE = re.compile(r"^([+-])\s*(\d+)\s*([mhd])$")        # +1h, -1d, +30m
UNITS = {"m": "minutes", "h": "hours", "d": "days"}
# Число у тексті: не частина назви (sim-1, w3wp, UTC+2) і не продовження іншого числа
NUMBER_RE = re.compile(r"(?<![\w.,:+\-])\d+(?:[.,]\d+)?(?![\w])")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Сесія і відповідь
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Стан розмови: момент «зараз», історія реплік, хто і звідки питає."""
    now: str | None = None             # момент в історії (місцевий час «2026-11-10 08:00»); None — живий режим
    history: list = field(default_factory=list)   # [{"role": ..., "content": текст}] — без JSON функцій
    source: str = "console"            # console / telegram / exam
    model: str | None = None           # None — з assistant.yaml
    effort: str | None = None          # None — з assistant.yaml
    mode: str = "tools"                # етап 3 додасть режими «без даних» і «сирі дані»
    user: str | None = None            # хто питає (Telegram ID) — лише для журналу
    moment_note: str | None = None     # «Момент змінено: …» — додається до наступного питання в історії


@dataclass
class Answer:
    """Результат ask(): текст для людини і все, що треба для журналу та іспиту."""
    text: str                          # текст LLM + службові рядки
    llm_text: str                      # лише текст LLM
    service_lines: list
    tool_calls: list                   # [{name, input, ms, kb, error, trimmed, ignored}]
    tool_results: list                 # JSON-результати функцій (у журнал не пишемо)
    llm_calls: int
    tokens_in: int
    tokens_out: int
    ms_total: int
    stale: bool
    error: str | None
    moment: str | None
    model: str
    effort: str


# ---------------------------------------------------------------------------
# Момент «зараз»
# ---------------------------------------------------------------------------

class MomentError(Exception):
    """Момент не можна поставити (не розібрали, раніше за дані, у майбутньому)."""


def local_now() -> pd.Timestamp:
    """Справжній поточний час, місцевий (UTC+2), до хвилини."""
    return pd.Timestamp.now(tz="UTC").floor("min").tz_convert(tools.LOCAL_TZ)


def as_local(text: str) -> pd.Timestamp:
    """Рядок «2026-11-10 08:00» (місцевий) -> Timestamp з поясом UTC+2."""
    return tools.to_utc(text).tz_convert(tools.LOCAL_TZ)


def show(t: pd.Timestamp) -> str:
    return t.strftime(tools.SHOW_FORMAT)


def data_bounds(servers: list) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Перші і останні дані по всіх серверах (місцевий час)."""
    firsts = [as_local(s["data_from"]["ts_local"]) for s in servers if s.get("data_from")]
    lasts = [as_local(s["data_until"]["ts_local"]) for s in servers if s.get("data_until")]
    return (min(firsts) if firsts else None), (max(lasts) if lasts else None)


def parse_moment(text: str, current: str | None, servers: list | None = None) -> tuple[str | None, str | None]:
    """Новий момент із команди /now (консоль) або /moment (бот).

    text: «2026-11-12 03:00», «+1h», «-1d», «+30m», «live». current — поточний момент сесії
    (None — живий режим; тоді зсув — від справжнього поточного часу).
    Повертає (момент або None для живого режиму, попередження або None).
    Раніше за перші дані або в майбутньому — MomentError; пізніше за останні дані — можна,
    з попередженням (тоді відповіді будуть з позначкою «дані застарілі»)."""
    text = (text or "").strip()
    if not text:
        raise MomentError("момент не задано; приклади: 2026-11-10 08:00, +1h, -1d, live")
    if text.lower() == "live":
        return None, None
    m = SHIFT_RE.match(text.lower())
    if m:
        sign, amount, unit = m.groups()
        base = as_local(current) if current else local_now()
        delta = pd.Timedelta(**{UNITS[unit]: int(amount)})
        t = base + delta if sign == "+" else base - delta
    else:
        try:
            t = as_local(text)
        except tools.ToolError:
            raise MomentError(f"не розумію момент «{text}»; приклади: 2026-11-10 08:00, +1h, -1d, live") from None
    if servers is None:
        servers = tools.list_servers().get("servers", [])
    first, last = data_bounds(servers)
    if t > local_now():
        raise MomentError(f"{show(t)} — це ще майбутнє; для поточного часу — live")
    if first is not None and t < first:
        raise MomentError(f"на {show(t)} даних немає: перші дані — {show(first)}")
    warning = None
    if last is not None and t > last:
        warning = (f"після {show(last)} даних немає — відповіді будуть з позначкою «дані застарілі»")
    return show(t), warning


def moment_text(moment: str | None) -> str:
    """Момент людською мовою: «середа, 11.11.2026 08:00 (момент в історії)»."""
    if moment is None:
        return f"живий режим (зараз {local_now().strftime('%d.%m.%Y %H:%M')})"
    t = as_local(moment)
    return f"{WEEKDAYS[t.weekday()]}, {t.strftime('%d.%m.%Y %H:%M')} (момент в історії)"


def set_moment(session: Session, moment: str | None) -> None:
    """Змінити момент сесії (/now у консолі, /moment у боті).
    Якщо розмова вже була, LLM має знати, що попередні відповіді — на інший момент:
    код додає примітку до наступного питання в історії (так чергування реплік
    user / assistant не порушується)."""
    if moment == session.now:
        return
    session.now = moment
    if session.history:
        session.moment_note = (f"[Службова примітка. Момент змінено: тепер {moment_text(moment)}. "
                               f"Попередні відповіді були на попередній момент і не є помилкою.]")


# ---------------------------------------------------------------------------
# Системний промпт
# ---------------------------------------------------------------------------

def server_role(description: str | None) -> str:
    """Роль сервера з опису — лише частина до двокрапки («Сервер БД: SQL Server, база …,
    нічний бекап о 02:00» -> «Сервер БД»). Решта опису — те, чого немає в даних (розклад,
    назва бази): LLM подавала б це як факт, тож у промпт не беремо."""
    role = (description or "").split(":")[0].strip()
    return role or "без опису"


def servers_text(servers: list) -> str:
    """Сервери для промпту — лише статичне і те, що є в даних: назва, роль, диски, служби,
    пули, сайти. (data_age_h і stale з list_servers рахуються за справжнім годинником —
    у промпт не беремо.)"""
    lines = []
    for s in servers:
        svc = [x["name"] for x in s.get("services", []) if x.get("kind") == "service"]
        pools = [x["name"] for x in s.get("services", []) if x.get("kind") == "apppool"]
        parts = [f"- {s['name']} — {server_role(s.get('description'))}",
                 f"диски: {', '.join(s.get('disks', [])) or 'немає даних'}"]
        if svc:
            parts.append(f"служби: {', '.join(svc)}")
        if pools:
            parts.append(f"пули IIS: {', '.join(pools)}")
        if s.get("sites"):
            parts.append("сайти: " + ", ".join(f"{x['site']} → {x['app_pool']}" for x in s["sites"]))
        lines.append("; ".join(parts))
    return "\n".join(lines) if lines else "- серверів немає"


def build_system_prompt(moment: str | None, servers: list, clock: pd.Timestamp | None = None) -> str:
    """Системний промпт для моменту сесії. clock — «справжній» час (для тестів живого режиму)."""
    t = as_local(moment) if moment else (clock if clock is not None else local_now())
    first, last = data_bounds(servers)
    # Для моменту в історії — «до моменту»: інакше LLM знала б, що далі дані є
    data_to = min(t, last) if (moment and last is not None) else last
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(
        weekday=WEEKDAYS[t.weekday()], date=t.strftime("%d.%m.%Y"), time=t.strftime("%H:%M"),
        moment_note=MOMENT_NOTE if moment else "",
        today=t.strftime("%d.%m.%Y"), yesterday=(t - pd.Timedelta(days=1)).strftime("%d.%m.%Y"),
        servers=servers_text(servers),
        data_from=first.strftime("%d.%m.%Y %H:%M") if first is not None else "немає даних",
        data_to=data_to.strftime("%d.%m.%Y %H:%M") if data_to is not None else "немає даних",
    ).replace(" \n", "\n")


# ---------------------------------------------------------------------------
# Виконання функцій
# ---------------------------------------------------------------------------

def to_json(obj) -> str:
    """Компактний JSON для LLM (менше токенів)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def json_size(obj) -> int:
    return len(to_json(obj).encode("utf-8"))


def is_chronological(items: list) -> bool:
    """Список за часом (найновіші — в кінці): записи з ts_local або пари [час, значення]."""
    first = items[0]
    if isinstance(first, dict):
        return "ts_local" in first
    return isinstance(first, list) and bool(first) and isinstance(first[0], str) \
        and bool(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", first[0]))


def longest_list(obj, parent=None, key=None):
    """Найбільший (у байтах JSON) список усередині результату: (батько, ключ, список)."""
    best = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = longest_list(v, obj, k)
            if found and (best is None or found[3] > best[3]):
                best = found
    elif isinstance(obj, list):
        if parent is not None and len(obj) >= 2:
            best = (parent, key, obj, json_size(obj))
        for x in obj:
            if isinstance(x, dict):
                found = longest_list(x)
                if found and (best is None or found[3] > best[3]):
                    best = found
    return best


def fit_result(result: dict, max_kb: float) -> tuple[dict, list]:
    """Запобіжник розміру: обрізаємо найдовший список, а не рядок JSON (JSON лишається правильним).
    Список за часом — лишаємо найновіші; інший — перші. Повертає (результат, примітки)."""
    limit = max_kb * 1024
    if json_size(result) <= limit:
        return result, []
    result, notes = copy.deepcopy(result), []
    for _ in range(5):                       # кілька списків — по черзі, від найбільшого
        if json_size(result) <= limit:
            break
        found = longest_list(result)
        if found is None:
            break
        parent, key, items, _size = found
        newest_last, keep = is_chronological(items), len(items)
        while keep > 1 and json_size(result) > limit:
            keep = max(1, int(keep * 0.8))
            parent[key] = items[-keep:] if newest_last else items[:keep]
        notes.append(f"{key}: показано {keep} з {len(items)} "
                     f"({'останні' if newest_last else 'перші'}) — обрізано за розміром")
    result["trimmed"] = notes
    return result, notes


def run_tool(call, session: Session, specs: list, cfg: dict) -> tuple[dict, str, dict]:
    """Виконати одну функцію на запит LLM. Повертає (результат, JSON для LLM, запис для журналу).
    now — лише з сесії; параметри, яких немає в описі для LLM (зокрема now), — ігноруються."""
    t0 = time.perf_counter()
    ignored = []
    if call.name not in tool_specs.TOOL_NAMES:
        result = {"error": f"невідомий інструмент «{call.name}»; є: {', '.join(tool_specs.TOOL_NAMES)}"}
    else:
        allowed = tool_specs.allowed_params(specs, call.name)
        args = {k: v for k, v in call.input.items() if k in allowed}
        ignored = sorted(set(call.input) - allowed)
        args.update(cfg.get("hidden_args", {}).get(call.name, {}))
        args["now"] = session.now
        result = tools.TOOLS[call.name](**args)
    result, trimmed = fit_result(result, cfg["tool_result_max_kb"])
    text = to_json(result)
    entry = {"name": call.name, "input": call.input, "ms": round((time.perf_counter() - t0) * 1000),
             "kb": round(len(text.encode("utf-8")) / 1024, 1), "error": result.get("error"),
             "trimmed": trimmed, "ignored": ignored}
    return result, text, entry


# ---------------------------------------------------------------------------
# Службові рядки (їх додає код, а не LLM)
# ---------------------------------------------------------------------------

def short(local_text: str | None) -> str:
    """«2026-11-18 22:07» -> «18.11 22:07»."""
    if not local_text:
        return "невідомо"
    return pd.Timestamp(local_text).strftime("%d.%m %H:%M")


def find_stale(obj, out: list) -> list:
    """Усі місця в результатах, де stale = true: [(сервер, час останніх даних)]."""
    if isinstance(obj, dict):
        if obj.get("stale") is True:
            out.append((obj.get("server"), (obj.get("data_until") or {}).get("ts_local")))
        for v in obj.values():
            find_stale(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_stale(v, out)
    return out


def stale_line(results: list) -> str | None:
    found = find_stale(results, [])
    if not found:
        return None
    times = sorted({short(u) for _, u in found})
    if len(times) == 1:
        return f"⚠ дані застарілі: останні на {times[0]}"
    by_server = sorted({f"{s} — {short(u)}" for s, u in found if s})
    return "⚠ дані застарілі: останні " + "; ".join(by_server)


def norm_number(s: str) -> str:
    """«8,9» і «8.90» -> «8.9»; «08» -> «8»: щоб однакові числа збігалися."""
    x = float(s.replace(",", "."))
    return str(int(x)) if x == int(x) else repr(round(x, 6))


def numbers_in(text: str) -> set[str]:
    """Усі числа в тексті (нормалізовані). Назви на кшталт sim-1, w3wp, UTC+2 — не числа."""
    return {norm_number(m) for m in NUMBER_RE.findall(text or "")}


def known_numbers(question: str, system: str) -> set[str]:
    """Числа, які LLM знала без функцій: з питання і з промпту (дата, поріг 90 % тощо).
    Номери правил промпту («1.», «12.») не враховуємо."""
    prompt = re.sub(r"(?m)^\s*\d+\.\s", "", system)
    return numbers_in(question) | numbers_in(prompt)


def seen_line(calls: list) -> str | None:
    """«Дивився: прогноз диска (sim-2, E:); ріст тек (sim-2, E:)» — людські назви функцій."""
    labels = []
    for c in calls:
        a = c["input"]
        parts = [a.get("server") or ("усі сервери" if c["name"] == "get_status" else None),
                 a.get("drive"), a.get("metric"),
                 "за пам'яттю" if a.get("by") == "ram" else None,
                 f"за {a['period_h']} год" if a.get("period_h") else None,
                 ", ".join(a["types"]) if isinstance(a.get("types"), list) and a["types"] else None,
                 a.get("source")]
        args = ", ".join(str(p) for p in parts if p)
        label = tool_specs.HUMAN_NAMES.get(c["name"], c["name"]) + (f" ({args})" if args else "")
        if label not in labels:
            labels.append(label)
    return ("Дивився: " + "; ".join(labels)) if labels else None


# ---------------------------------------------------------------------------
# Журнал
# ---------------------------------------------------------------------------

def write_log(record: dict) -> None:
    """Один рядок JSON на питання. Ключів і токенів тут немає і бути не може:
    у запис потрапляють лише питання, відповідь, виклики функцій і числа."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Головна функція
# ---------------------------------------------------------------------------

def history_window(history: list, n: int) -> list:
    """Останні n реплік; перша має бути від користувача (так вимагає API)."""
    tail = history[-n:] if n > 0 else []
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]
    return tail


def ask(question: str, session: Session, chat_fn=None, cfg: dict | None = None) -> Answer:
    """Відповісти на питання в межах сесії. chat_fn — функція виклику LLM
    (за замовчуванням llm.chat; у тестах — підставна)."""
    cfg = cfg or load_config()
    chat_fn = chat_fn or llm.chat
    model = session.model or cfg["model"]
    effort = session.effort or cfg["effort"]
    t0 = time.perf_counter()

    servers = tools.list_servers().get("servers", [])
    system = build_system_prompt(session.now, servers)
    specs = tool_specs.build_specs(servers)
    # Після зміни моменту — службова примітка перед питанням (лише якщо розмова вже була)
    user_text = f"{session.moment_note}\n{question}" if session.moment_note else question
    messages = history_window(session.history, cfg["history_messages"]) + \
        [{"role": "user", "content": user_text}]

    calls, results = [], []
    llm_calls = tokens_in = tokens_out = 0
    llm_text, error = "", None
    try:
        for round_no in range(cfg["max_tool_rounds"] + 1):
            reply = chat_fn(system=system, messages=messages, tools=specs, model=model, effort=effort,
                            max_tokens=cfg["max_tokens"], timeout_s=cfg["timeout_s"],
                            max_retries=cfg["max_retries"])
            llm_calls += 1
            tokens_in += reply.tokens_in
            tokens_out += reply.tokens_out
            if not reply.tool_calls:                       # остаточна відповідь
                llm_text = reply.text
                if not llm_text:
                    error = "порожня відповідь LLM"
                    llm_text = "LLM повернула порожню відповідь; спробуйте ще раз."
                break
            if round_no == cfg["max_tool_rounds"]:          # ліміт кроків вичерпано
                error = f"ліміт {cfg['max_tool_rounds']} кроків"
                llm_text = (f"Не вдалося відповісти за {cfg['max_tool_rounds']} кроків; "
                            f"спробуйте уточнити питання.")
                break
            # Відповідь моделі — у розмову без змін (зокрема блоки thinking);
            # усі результати функцій — одним повідомленням (так вимагає API)
            messages.append({"role": "assistant", "content": reply.content})
            blocks = []
            for call in reply.tool_calls:
                result, text, entry = run_tool(call, session, specs, cfg)
                calls.append(entry)
                results.append(result)
                blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": text,
                               "is_error": "error" in result})
            messages.append({"role": "user", "content": blocks})
    except LLMError as e:
        error, llm_text = str(e), str(e)

    # Службові рядки — код, а не LLM
    lines = []
    stale = stale_line(results)
    if stale:
        lines.append(stale)
    if not calls and not error and numbers_in(llm_text) - known_numbers(question, system):
        lines.append("⚠ відповідь без даних з серверів")
    seen = seen_line(calls)
    if seen:
        lines.append(seen)
    if session.now:
        lines.append(f"Момент: {short(session.now)} (історія)")
    text = llm_text + ("\n\n" + "\n".join(lines) if lines else "")

    # Історія — лише текст (без JSON функцій) + «Дивився: …», щоб «а на D:?» знало сервер
    # Примітка про зміну моменту лишається в історії разом із питанням
    if not error:
        session.history += [{"role": "user", "content": user_text},
                            {"role": "assistant", "content": llm_text + (f"\n{seen}" if seen else "")}]
        session.history = session.history[-cfg["history_messages"]:]
        session.moment_note = None

    answer = Answer(text=text, llm_text=llm_text, service_lines=lines, tool_calls=calls,
                    tool_results=results, llm_calls=llm_calls, tokens_in=tokens_in, tokens_out=tokens_out,
                    ms_total=round((time.perf_counter() - t0) * 1000), stale=bool(stale), error=error,
                    moment=session.now, model=model, effort=effort)
    write_log({"ts": pd.Timestamp.now(tz="UTC").strftime(tools.TS_FORMAT), "source": session.source,
               "user": session.user, "moment": session.now, "model": model, "effort": effort,
               "mode": session.mode, "question": question,
               "moment_note": user_text != question, "tool_calls": calls, "llm_calls": llm_calls,
               "tokens_in": tokens_in, "tokens_out": tokens_out, "ms_total": answer.ms_total,
               "stale": answer.stale, "answer": text, "error": error})
    return answer


# ---------------------------------------------------------------------------
# Консоль
# ---------------------------------------------------------------------------

def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def moment_header(moment: str | None, model: str, effort: str) -> str:
    if moment:
        t = as_local(moment)
        where = f"{moment} ({WEEKDAYS_SHORT[t.weekday()]}, історія)"
    else:
        where = f"живий режим (зараз {show(local_now())})"
    return f"Момент: {where}, модель {model}, effort {effort}"


def print_answer(ans: Answer, verbose: bool) -> None:
    for c, r in zip(ans.tool_calls, ans.tool_results):
        args = " ".join(str(v) for k, v in c["input"].items())
        extra = f", помилка: {c['error']}" if c["error"] else ""
        extra += f", ігноровано: {', '.join(c['ignored'])}" if c["ignored"] else ""
        extra += ", обрізано" if c["trimmed"] else ""
        print(f"  [{c['name']} {args}] {str(c['kb']).replace('.', ',')} КБ, {c['ms']} мс{extra}")
        if verbose:
            print("  " + json.dumps(r, ensure_ascii=False, indent=1).replace("\n", "\n  "))
    print(ans.text)
    secs = f"{ans.ms_total / 1000:.1f}".replace(".", ",")
    print(f"  ({ans.llm_calls} викл. LLM, {fmt_int(ans.tokens_in)} / {fmt_int(ans.tokens_out)} токенів, "
          f"{secs} с)")


HELP = """Команди: /now <час> | /now +1h | /now -1d | /now live | /reset | /model <назва> | /exit
Час — місцевий, наприклад: /now 2026-11-12 03:00"""


def main() -> None:
    # UTF-8 і для консолі, і коли вивід іде у файл / конвеєр (інакше Windows ламає кирилицю)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Асистент чергового інженера — консоль")
    p.add_argument("--now", help="момент в історії, місцевий час: \"2026-11-10 08:00\"")
    p.add_argument("--verbose", action="store_true", help="показувати JSON кожної функції")
    p.add_argument("--model", help="інша модель, ніж в assistant.yaml")
    p.add_argument("--effort", choices=["low", "medium", "high"], help="інший effort, ніж в assistant.yaml")
    p.add_argument("--show-prompt", action="store_true", help="показати системний промпт і вийти (без LLM)")
    p.add_argument("--show-tools", action="store_true", help="показати опис функцій для LLM і вийти (без LLM)")
    args = p.parse_args()

    cfg = load_config()
    servers = tools.list_servers().get("servers", [])
    session = Session(source="console", model=args.model, effort=args.effort)
    if args.now:
        try:
            session.now, warning = parse_moment(args.now, None, servers)
        except MomentError as e:
            sys.exit(f"Помилка: {e}")
        if warning:
            print(f"⚠ {warning}")

    if args.show_prompt:
        print(build_system_prompt(session.now, servers))
        return
    if args.show_tools:
        specs = tool_specs.build_specs(servers)
        print(json.dumps(specs, ensure_ascii=False, indent=1))
        sizes = ", ".join(f"{s['name']} {json_size(s) / 1024:.1f}".replace(".", ",") for s in specs)
        print(f"\nРозмір (КБ): {sizes}; разом {json_size(specs) / 1024:.1f} КБ".replace(".", ","))
        return

    print(moment_header(session.now, session.model or cfg["model"], session.effort or cfg["effort"]))
    print(HELP)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            cmd, _, rest = q.partition(" ")
            if cmd == "/exit":
                break
            if cmd == "/reset":
                session.history = []
                session.moment_note = None
                print("Історію розмови очищено.")
            elif cmd == "/now":
                if not rest.strip():
                    print(moment_header(session.now, session.model or cfg["model"], session.effort or cfg["effort"]))
                    continue
                try:
                    moment, warning = parse_moment(rest, session.now)
                    set_moment(session, moment)       # + примітка в історії, якщо розмова вже була
                except MomentError as e:
                    print(f"Помилка: {e}")
                    continue
                print(moment_header(session.now, session.model or cfg["model"], session.effort or cfg["effort"]))
                if warning:
                    print(f"⚠ {warning}")
            elif cmd == "/model" and rest.strip():
                session.model = rest.strip()
                print(f"Модель: {session.model}")
            else:
                print(HELP)
            continue
        if not sys.stdin.isatty():         # питання з файлу: показати їх у виводі (транскрипт)
            print(q)
        print_answer(ask(q, session, cfg=cfg), args.verbose)


if __name__ == "__main__":
    main()
