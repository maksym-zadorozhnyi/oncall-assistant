"""Тести ядра асистента (core/assistant.py, core/tool_specs.py). Запуск: python -m pytest

Жодних справжніх викликів API: LLM — підставна функція з заготовленими відповідями,
а llm.get_client у кожному тесті підмінено на «вибух».
База — та сама тимчасова, що й у тестах інструментів (tests/test_tools.py):
сервер t-1, дані 2 доби з 2026-01-05 02:00 місцевого; SitePool стоїть D+50…D+70.
"""
import inspect
import json
import re

import pandas as pd
import pytest

from core import assistant as A
from core import llm, tool_specs, tools
from core.llm import LLMError, LLMReply, ToolCall
from test_tools import D, MINUTES, db, local  # noqa: F401  (db — фікстура тимчасової бази)

MOMENT = local(D + 60)                      # 2026-01-06 03:00 місцевого


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch, tmp_path):
    """Справжній клієнт API в тестах заборонено; журнал — у тимчасову теку."""
    def boom(*args, **kwargs):
        raise AssertionError("справжній виклик API в тесті")
    monkeypatch.setattr(llm, "get_client", boom)
    monkeypatch.setattr(A, "LOG_PATH", tmp_path / "logs" / "assistant.jsonl")


def reply(text: str = "", calls: list | None = None) -> LLMReply:
    """Заготовлена відповідь LLM: текст і/або запити на функції [(назва, параметри)]."""
    calls = [ToolCall(f"tu_{i}", n, inp) for i, (n, inp) in enumerate(calls or [])]
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]   # як у Haiku 5.5
    if text:
        content.append({"type": "text", "text": text})
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
    return LLMReply(text=text, tool_calls=calls, content=content, stop_reason="tool_use" if calls else "end_turn",
                    model="fake", tokens_in=100, tokens_out=10, ms=5)


class FakeLLM:
    """Підставна LLM: віддає відповіді по черзі і запам'ятовує кожен запит."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))   # копія на момент виклику
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


def tool_results(request: dict) -> list:
    """Блоки tool_result з останнього повідомлення запиту."""
    return [b for b in request["messages"][-1]["content"] if b.get("type") == "tool_result"]


# ---------------------------------------------------------------------------
# Цикл LLM ↔ функції
# ---------------------------------------------------------------------------

def test_one_tool_now_from_session(db):
    """Одна функція: now підставляє код із сесії; відповідь моделі повертається без змін."""
    first = reply(calls=[("get_disk_forecast", {"server": "t-1", "drive": "C:"})])
    fake = FakeLLM(first, reply("Диск C: росте ~2 %/добу."))
    s = A.Session(now=MOMENT, source="test")
    ans = A.ask("Коли заповниться C:?", s, chat_fn=fake)
    assert ans.llm_calls == 2 and len(ans.tool_calls) == 1 and ans.error is None
    res = json.loads(tool_results(fake.requests[1])[0]["content"])
    assert res["now"]["ts_local"] == MOMENT                     # саме момент сесії
    assert fake.requests[1]["messages"][-2] == {"role": "assistant", "content": first.content}
    assert "Дивився: прогноз диска (t-1, C:)" in ans.text
    assert f"Момент: {A.short(MOMENT)} (історія)" in ans.text


def test_now_from_llm_is_ignored(db):
    """Якщо LLM передасть now — код його ігнорує і бере момент сесії."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "t-1", "now": "2026-01-05 03:00"})]), reply("ок"))
    ans = A.ask("Що з t-1?", A.Session(now=MOMENT), chat_fn=fake)
    res = json.loads(tool_results(fake.requests[1])[0]["content"])
    assert res["now"]["ts_local"] == MOMENT
    assert ans.tool_calls[0]["ignored"] == ["now"]


def test_parallel_tools_one_message(db):
    """Кілька функцій за раз — усі результати одним повідомленням, з відповідними id."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "t-1"}),
                                ("get_folder_growth", {"server": "t-1", "drive": "C:"})]), reply("готово"))
    ans = A.ask("Що з t-1 і що росте?", A.Session(now=MOMENT), chat_fn=fake)
    blocks = tool_results(fake.requests[1])
    assert [b["tool_use_id"] for b in blocks] == ["tu_0", "tu_1"]
    assert fake.requests[1]["messages"][-1]["role"] == "user" and ans.llm_calls == 2
    assert "Дивився: стан (t-1); ріст тек (t-1, C:)" in ans.text


def test_several_rounds(db):
    """Кілька кроків: функція -> ще функція -> текст."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "t-1"})]),
                   reply(calls=[("get_timeline", {"server": "t-1", "types": ["service"]})]),
                   reply("SitePool зупинено о 03:50."))
    ans = A.ask("Чому лежить сайт?", A.Session(now=MOMENT), chat_fn=fake)
    assert ans.llm_calls == 3 and [c["name"] for c in ans.tool_calls] == ["get_status", "get_timeline"]
    assert "хронологія (t-1, service)" in ans.text


def test_round_limit(db):
    """LLM без кінця просить функції — після 6 кроків зрозуміле повідомлення, в історію не пишемо."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "t-1"})]))
    s = A.Session(now=MOMENT)
    ans = A.ask("?", s, chat_fn=fake)
    assert ans.llm_calls == 7 and len(ans.tool_calls) == 6
    assert ans.llm_text.startswith("Не вдалося відповісти за 6 кроків") and ans.error
    assert s.history == []


def test_tool_error_goes_back_to_llm(db):
    """Помилка функції (невідомий сервер) — результат з is_error, LLM відповідає далі."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "sim-9"})]), reply("Сервера sim-9 немає."))
    ans = A.ask("Що з sim-9?", A.Session(now=MOMENT), chat_fn=fake)
    block = tool_results(fake.requests[1])[0]
    assert block["is_error"] is True and "невідомий сервер" in json.loads(block["content"])["error"]
    assert ans.error is None and ans.tool_calls[0]["error"]


def test_api_error_message(db):
    """Помилка API — повідомлення користувачу, без падіння; в історію не пишемо."""
    s = A.Session(now=MOMENT)
    ans = A.ask("Що з t-1?", s, chat_fn=FakeLLM(LLMError("LLM зараз недоступна: тест")))
    assert ans.text.startswith("LLM зараз недоступна") and ans.error and s.history == []


# ---------------------------------------------------------------------------
# Службові рядки
# ---------------------------------------------------------------------------

def test_stale_line_added_by_code(db):
    """Момент після кінця даних: «⚠ дані застарілі», навіть якщо LLM про це мовчить."""
    late = local(MINUTES + 600)
    ans = A.ask("Що з t-1?", A.Session(now=late),
                chat_fn=FakeLLM(reply(calls=[("get_status", {"server": "t-1"})]), reply("Усе гаразд.")))
    assert ans.stale and f"⚠ дані застарілі: останні на {A.short(local(MINUTES - 1))}" in ans.text


def test_no_data_line(db):
    """Без функцій, але з числами — «⚠ відповідь без даних»; привітання і поріг 90 % — ні."""
    s = A.Session(now=MOMENT)
    assert "⚠ відповідь без даних" in A.ask("Що з CPU?", s, chat_fn=FakeLLM(reply("CPU 95 %."))).text
    assert "⚠" not in A.ask("привіт", s, chat_fn=FakeLLM(reply("Привіт! Чим допомогти?"))).text
    assert "⚠" not in A.ask("поріг?", s, chat_fn=FakeLLM(reply("Поріг — 90 %."))).text
    # числа з питання і назви на кшталт t-1 — не вигадані
    assert "⚠" not in A.ask("що з t-1 о 14:00?", s, chat_fn=FakeLLM(reply("Про 14:00 на t-1 спитаю дані."))).text


def test_numbers_in_text():
    """sim-1, w3wp, UTC+2 — не числа; 8,9 і 8.9 — однакові."""
    assert A.numbers_in("sim-1 w3wp.exe UTC+2 E:") == set()
    assert A.numbers_in("8,9 %/добу, 42 год") == A.numbers_in("8.9 і 42.0") == {"8.9", "42"}


# ---------------------------------------------------------------------------
# Історія, журнал, промпт, момент
# ---------------------------------------------------------------------------

def test_history_text_only_with_seen_line(db):
    """В історії — лише текст і «Дивився: …»; наступне питання бачить її, без JSON функцій."""
    s = A.Session(now=MOMENT)
    A.ask("Коли заповниться C:?", s, chat_fn=FakeLLM(
        reply(calls=[("get_disk_forecast", {"server": "t-1", "drive": "C:"})]), reply("Через 2 доби.")))
    assert s.history[1] == {"role": "assistant", "content": "Через 2 доби.\nДивився: прогноз диска (t-1, C:)"}
    fake = FakeLLM(reply("ок"))
    A.ask("а що там росте?", s, chat_fn=fake)
    msgs = fake.requests[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert all(isinstance(m["content"], str) for m in msgs)


def test_history_limit(db):
    s = A.Session(now=MOMENT)
    for i in range(6):
        A.ask(f"питання {i}", s, chat_fn=FakeLLM(reply("ок")))
    assert len(s.history) == A.load_config()["history_messages"] == 8
    assert s.history[0] == {"role": "user", "content": "питання 2"}


def test_log_written_without_secrets(db, monkeypatch):
    """Журнал: один рядок на питання з потрібними полями; ключа API там немає."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-SECRET-123")
    A.ask("Що з t-1?", A.Session(now=MOMENT, source="test", user="42"),
          chat_fn=FakeLLM(reply(calls=[("get_status", {"server": "t-1"})]), reply("ок")))
    text = A.LOG_PATH.read_text(encoding="utf-8")
    rec = json.loads(text.strip().splitlines()[-1])
    for key in ("ts", "source", "user", "moment", "model", "effort", "mode", "question", "tool_calls",
                "llm_calls", "tokens_in", "tokens_out", "ms_total", "stale", "answer", "error"):
        assert key in rec
    assert rec["moment"] == MOMENT and rec["effort"] == "low" and rec["tool_calls"][0]["name"] == "get_status"
    assert "SECRET" not in text


def test_system_prompt_moment(db):
    """Момент в історії: день тижня, «вчора», примітка; дані — «до моменту»; лише статичні поля серверів."""
    servers = tools.list_servers()["servers"]
    p = A.build_system_prompt(MOMENT, servers)
    assert "Зараз: вівторок, 06.01.2026 03:00" in p and "вчора = 05.01.2026" in p
    assert A.MOMENT_NOTE in p
    assert "Дані в базі: з 05.01.2026 02:00 до 06.01.2026 03:00." in p
    assert "t-1 — тестовий; диски: C:" in p and "пули IIS: SitePool" in p
    block = A.servers_text(servers)                  # блок серверів — без свіжості за справжнім годинником
    assert block in p and "stale" not in block and "data_age" not in block
    # момент після кінця даних — «до» = останні дані, а не момент
    assert "до 07.01.2026 01:59." in A.build_system_prompt(local(MINUTES + 600), servers)
    # живий режим — без примітки про історію
    live = A.build_system_prompt(None, servers, clock=pd.Timestamp("2026-11-10 08:00", tz=tools.LOCAL_TZ))
    assert "Зараз: вівторок, 10.11.2026 08:00" in live and A.MOMENT_NOTE not in live


def test_moment_change_note(db):
    """Після /now код додає примітку «Момент змінено: тепер …» до наступного питання;
    вона лишається в історії; без попередньої розмови примітки немає."""
    s = A.Session(now=MOMENT)
    A.set_moment(s, local(D + 120))                       # історії ще немає — примітки теж
    assert s.moment_note is None
    A.ask("Коли заповниться C:?", s, chat_fn=FakeLLM(reply("Через 2 доби.")))
    A.set_moment(s, local(D + 180))
    fake = FakeLLM(reply("Тепер інакше."))
    A.ask("А зараз?", s, chat_fn=fake)
    last = fake.requests[0]["messages"][-1]["content"]
    assert "Момент змінено: тепер вівторок, 06.01.2026 05:00" in last and last.endswith("\nА зараз?")
    assert s.history[2]["content"] == last and s.moment_note is None   # у історії; лише один раз
    fake = FakeLLM(reply("ок"))
    A.ask("ще?", s, chat_fn=fake)
    assert fake.requests[0]["messages"][-1]["content"] == "ще?"
    A.set_moment(s, None)                                  # живий режим
    assert "тепер живий режим" in s.moment_note


def test_server_role_only():
    """У промпт — лише роль з опису (до двокрапки): розкладу й назви бази в даних немає."""
    servers = [{"name": "sim-2", "description": "Сервер БД: SQL Server, база AppDb, нічний бекап о 02:00",
                "disks": ["E:"], "services": [{"name": "MSSQLSERVER", "kind": "service"}]}]
    text = A.servers_text(servers)
    assert text.startswith("- sim-2 — Сервер БД; диски: E:; служби: MSSQLSERVER")
    assert "02:00" not in text and "AppDb" not in text
    assert A.server_role(None) == "без опису" and A.server_role("тестовий") == "тестовий"


def test_prompt_rules():
    """Правила зупинки 1 є в промпті."""
    p = A.PROMPT_PATH.read_text(encoding="utf-8")
    for rule in ("десятковий роздільник — кома", "Не порівнюй числа словами", "виклич її сам",
                 "Не називай користувачу назв функцій"):
        assert rule in p


def test_parse_moment(db):
    """Зсуви, live, дата; раніше за дані і майбутнє — помилка; пізніше за дані — з попередженням."""
    servers = tools.list_servers()["servers"]
    assert A.parse_moment("+1h", MOMENT, servers) == (local(D + 120), None)
    assert A.parse_moment("-1d", MOMENT, servers) == (local(60), None)
    assert A.parse_moment("+30m", MOMENT, servers)[0] == local(D + 90)
    assert A.parse_moment("live", MOMENT, servers) == (None, None)
    assert A.parse_moment(local(D), None, servers) == (local(D), None)
    late, warning = A.parse_moment("+2d", MOMENT, servers)
    assert late == local(D + 60 + 2880) and "даних немає" in warning
    for bad in ("-2d", "2027-01-01 00:00", "вчора", ""):
        with pytest.raises(A.MomentError):
            A.parse_moment(bad, MOMENT, servers)


def test_fit_result_trims_longest_list():
    """Завеликий результат: обрізається найдовший список (найновіші лишаються), JSON правильний."""
    items = [{"ts_local": f"2026-01-05 {h:02d}:00", "text": "х" * 300} for h in range(24)]
    res, notes = A.fit_result({"server": "t-1", "items": items, "notes": []}, max_kb=2)
    assert A.json_size(res) <= 2.5 * 1024 and res["items"][-1] == items[-1]
    assert re.match(r"items: показано \d+ з 24 \(останні\)", notes[0]) and res["trimmed"] == notes
    small = {"a": [1, 2, 3]}
    assert A.fit_result(small, max_kb=2) == (small, [])


# ---------------------------------------------------------------------------
# Специфікації інструментів
# ---------------------------------------------------------------------------

def test_specs_match_functions(db):
    """Кожен інструмент має опис; кожен параметр є в сигнатурі функції; now, max_points,
    with_line немає ні в описах, ні в схемах (LLM їх не бачить)."""
    specs = tool_specs.build_specs()
    assert {s["name"] for s in specs} == set(tools.TOOLS) - {"list_servers"}
    for s in specs:
        sig = inspect.signature(tools.TOOLS[s["name"]])
        assert set(s["input_schema"]["properties"]) <= set(sig.parameters)
        assert set(s["input_schema"]["required"]) <= set(s["input_schema"]["properties"])
        assert s["description"] and tool_specs.CODE_ONLY_MARK not in s["description"]
        assert "Коли викликати" in s["description"]
    seen_by_llm = json.dumps(specs, ensure_ascii=False)
    for hidden in tool_specs.HIDDEN_PARAMS:          # ціле слово: size_now_gb — інше поле
        assert not re.search(rf"\b{hidden}\b", seen_by_llm), hidden
    assert specs[0]["input_schema"]["properties"]["server"]["enum"] == ["t-1"]
