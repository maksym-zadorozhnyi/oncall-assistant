"""Функції-інструменти: єдине джерело чисел для дашборду, бота (етап 1) і далі.

Кожна функція:
- приймає назву сервера і момент «зараз» (now): None — останні дані в базі,
  інакше — момент в історії; бачить лише дані до цієї хвилини включно;
- час без поясу на вході — місцевий (UTC+2), переводиться в UTC тут, у коді;
- повертає словник (JSON) з фактами, без висновків «це проблема»;
- не читає sim_incidents (ключ відповідей симулятора);
- помилку повертає словником {"error": "..."}, а не падає.

Docstring кожної функції — майбутній опис інструменту для LLM.

Перевірка з консолі:
    python -m core.tools get_status --server sim-2 --now "2026-11-10 08:00"
"""

import argparse
import json
import logging
import math
import re
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

from core import forecast
from core.db import DB_PATH, db_session

# ---------------------------------------------------------------------------
# Налаштування
# ---------------------------------------------------------------------------

LOCAL_TZ = "Etc/GMT-2"          # місцевий час = UTC+2 (як у симуляторі; знак у назві навпаки)
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # формат часу в базі
SHOW_FORMAT = "%Y-%m-%d %H:%M"  # формат часу у відповіді
STALE_MIN = 10                  # сервер «замовк», якщо даних немає довше
SNAPSHOT_MAX_MIN = 10           # знімок процесів — не давніший за стільки хвилин
BACKUP_CHECK_MAX_H = 2          # перевірка бекапу щогодинна; давніша — вік «на зараз» невідомий
DEFAULT_WINDOW_H = 24           # вікно за замовчуванням для історії і хронології
NOISE_EVENTS = {1014, 37}       # шумові події: у хронології лише кількість
INFO_EVENTS = {7036, 18264}     # події рівня Information, які все одно показуємо
TIMELINE_LEVELS = {"Warning", "Error", "Critical"}
TEXT_MAX = 200                  # довжина тексту події у хронології

UNITS = {"cpu_pct": "%", "ram_pct": "%", "app_error_pct": "%", "disk_read_mbs": "МБ/с",
         "disk_write_mbs": "МБ/с", "net_in_mbs": "МБ/с", "net_out_mbs": "МБ/с",
         "app_response_ms": "мс", "disk_used_pct": "%", "disk_free_gb": "ГБ"}

# Що означає checks.value для кожного типу перевірки
CHECK_VALUE = {"http": "response_ms", "cert": "days_left", "backup": "age_h", "sql_port": "response_ms"}

TOOLS = {}                      # назва -> функція (на етапі 1 — інструменти для LLM)
log = logging.getLogger(__name__)


class ToolError(Exception):
    """Зрозуміла помилка для користувача або LLM (невідомий сервер, немає даних)."""

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.extra = extra


def tool(func):
    """Реєструє функцію як інструмент. Будь-яка помилка -> {"error": ...}, а не падіння:
    LLM може передати дивні параметри, і бот не має падати.
    ToolError — зрозуміле пояснення; інша помилка — «внутрішня», а сам виняток — у лог."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolError as e:
            return {"error": str(e), **e.extra}
        except Exception as e:
            log.exception("інструмент %s(%s, %s) упав", func.__name__, args, kwargs)
            return {"error": f"внутрішня помилка: {type(e).__name__}: {e}"}
    TOOLS[func.__name__] = wrapper
    return wrapper


def as_number(value, name: str, minimum: float, integer: bool = True, strict: bool = False):
    """Параметр-число з перевіркою: «50», 50.0 -> 50. Менше за minimum (або рівне при
    strict) чи не число — зрозуміла помилка."""
    try:
        x = float(value)
        if integer:
            if x != int(x):
                raise ValueError
            x = int(x)
    except (TypeError, ValueError, OverflowError):
        x = None
    if x is None or (x <= minimum if strict else x < minimum):
        sign = ">" if strict else "≥"
        raise ToolError(f"{name} має бути {'цілим ' if integer else ''}числом {sign} {minimum:g}, "
                        f"а не «{value}»")
    return x


# ---------------------------------------------------------------------------
# Час
# ---------------------------------------------------------------------------

def to_utc(value) -> pd.Timestamp | None:
    """Момент з входу -> UTC. None -> None. Без поясу — місцевий час (UTC+2)."""
    if value is None or value == "":
        return None
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ToolError(f"не розумію час «{value}»; формат: 2026-11-10 08:00 (місцевий)")
    if t.tzinfo is None:
        t = t.tz_localize(LOCAL_TZ)
    return t.tz_convert("UTC").floor("min")


def db_ts(t: pd.Timestamp) -> str:
    """Час у форматі бази (для порівняння в SQL)."""
    return t.strftime(TS_FORMAT)


def cutoff(now: pd.Timestamp) -> str:
    """Верхня межа «до now включно»: усі записи цієї хвилини (сирі мають секунди)."""
    return db_ts(now + pd.Timedelta(minutes=1))


def parse_db(ts: str) -> pd.Timestamp:
    """Час з бази ('2026-11-10T06:00:00Z') -> Timestamp UTC."""
    return pd.Timestamp(ts).tz_convert("UTC")


def loc(t) -> str | None:
    """Місцевий час рядком: '2026-11-10 08:00'."""
    if t is None:
        return None
    if isinstance(t, str):
        t = parse_db(t)
    return t.tz_convert(LOCAL_TZ).strftime(SHOW_FORMAT)


def pair(t) -> dict | None:
    """Обидва часи: {"ts_utc": ..., "ts_local": ...}."""
    if t is None:
        return None
    if isinstance(t, str):
        t = parse_db(t)
    return {"ts_utc": t.tz_convert("UTC").strftime(SHOW_FORMAT), "ts_local": loc(t)}


def localize_text(text: str | None) -> str | None:
    """Час UTC усередині тексту (напр. «останній успішний: 2026-10-26T00:40:00Z») -> місцевий."""
    if not text:
        return text
    return re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z",
                  lambda m: f"{loc(m.group(0))} (місц.)", text)


# ---------------------------------------------------------------------------
# Числа і база
# ---------------------------------------------------------------------------

def num(v, nd: int = 2):
    """Округлення; None і NaN -> None (у JSON — null)."""
    if v is None:
        return None
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def unit_of(metric: str) -> str:
    return UNITS.get(metric.split(":")[0], "")


def connect():
    """З'єднання з базою. DB_PATH читається під час виклику — тести підміняють його."""
    return db_session(DB_PATH)


def find_server(conn, name: str):
    """Рядок servers за назвою або ToolError зі списком відомих серверів."""
    row = conn.execute("SELECT * FROM servers WHERE name = ?", (name,)).fetchone()
    if row is None:
        known = [r["name"] for r in conn.execute("SELECT name FROM servers ORDER BY id")]
        raise ToolError(f"невідомий сервер «{name}»", known_servers=known)
    return row


def inventory(server_row) -> dict:
    try:
        return json.loads(server_row["inventory_json"] or "{}")
    except json.JSONDecodeError:
        return {}


def data_range(conn, server_id: int) -> tuple[str | None, str | None]:
    """Перші й останні дані сервера (за рядом cpu_pct у metrics_clean — він є завжди)."""
    r = conn.execute("SELECT MIN(ts), MAX(ts) FROM metrics_clean WHERE server_id = ? "
                     "AND metric = 'cpu_pct'", (server_id,)).fetchone()
    return r[0], r[1]


def resolve_now(conn, server_row, now) -> tuple[pd.Timestamp, bool]:
    """Момент «зараз» у UTC і чи це живий режим.
    Живий режим (now=None) — справжній поточний час: якщо сервер давно не надсилав даних,
    це видно (stale), а не підміняється останніми даними. Момент раніше за перші дані — помилка."""
    first, last = data_range(conn, server_row["id"])
    if last is None:
        raise ToolError(f"для сервера {server_row['name']} ще немає даних")
    t = to_utc(now)
    if t is None:
        return pd.Timestamp.now(tz="UTC").floor("min"), True
    if t < parse_db(first):
        raise ToolError(f"немає даних до {loc(t)} (місц.); перші дані {server_row['name']} — "
                        f"{loc(first)}", data_from=pair(first))
    return t, False


def last_ts_before(conn, server_id: int, now: pd.Timestamp) -> str | None:
    r = conn.execute("SELECT MAX(ts) FROM metrics_clean WHERE server_id = ? AND metric = 'cpu_pct' "
                     "AND ts < ?", (server_id, cutoff(now))).fetchone()
    return r[0]


def age_text(hours: float) -> str:
    """Скільки часу тому словами: «25 хв», «7 год», «2,9 дн»."""
    if hours < 1:
        return f"{hours * 60:.0f} хв"
    if hours < 48:
        return f"{hours:.0f} год"
    return f"{hours / 24:.1f} дн".replace(".", ",")


def freshness(conn, server_id: int, t: pd.Timestamp) -> dict:
    """Одне правило «дані застаріли» для всіх функцій: data_until (останні дані сервера
    не пізніше now), data_age_h (скільки годин тому), stale (старші за STALE_MIN хвилин),
    і для stale — stale_note словами."""
    until = last_ts_before(conn, server_id, t)
    until_t = parse_db(until)
    age_h = max((t - until_t).total_seconds() / 3600, 0.0)
    out = {"data_until": pair(until_t), "data_age_h": num(age_h, 1), "stale": age_h * 60 > STALE_MIN}
    if out["stale"]:
        out["stale_note"] = (f"немає свіжих даних: останні дані на {loc(until_t)}, "
                             f"{age_text(age_h)} тому")
    return out


def metric_names(conn, server_id: int) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT metric FROM metrics_clean WHERE server_id = ? "
                                       "ORDER BY metric", (server_id,))]


def disks_of(metrics: list[str]) -> list[str]:
    return [m.split(":", 1)[1] for m in metrics if m.startswith("disk_used_pct:")]


# ---------------------------------------------------------------------------
# 0. Список серверів
# ---------------------------------------------------------------------------

@tool
def list_servers() -> dict:
    """Список серверів, які знає система.

    Коли викликати: щоб дізнатися назви серверів, їх диски, служби, пули IIS і сайти,
    а також за який період є дані.
    Повертає по кожному серверу: name, kind (sim — симуляція, real — справжній),
    description, disks, services (служби і пули IIS), sites (сайт -> пул),
    data_from і data_until (час перших і останніх даних, UTC і місцевий).
    """
    out = []
    clock = pd.Timestamp.now(tz="UTC")
    with connect() as conn:
        for s in conn.execute("SELECT * FROM servers ORDER BY id").fetchall():
            first, last = data_range(conn, s["id"])
            services = [{"name": r["name"], "kind": r["kind"]} for r in conn.execute(
                "SELECT DISTINCT name, kind FROM service_status WHERE server_id = ? ORDER BY kind DESC, name",
                (s["id"],))]
            inv = inventory(s)
            item = {"name": s["name"], "kind": s["kind"], "description": s["description"],
                    "disks": disks_of(metric_names(conn, s["id"])),
                    "services": services,
                    "sites": [{"site": x.get("name"), "app_pool": x.get("app_pool")}
                              for x in inv.get("sites", [])],
                    "data_from": pair(first), "data_until": pair(last)}
            if last is not None:      # свіжість — відносно справжнього поточного часу
                age_h = max((clock - parse_db(last)).total_seconds() / 3600, 0.0)
                item.update({"data_age_h": num(age_h, 1), "stale": age_h * 60 > STALE_MIN})
            out.append(item)
    return {"servers": out}


# ---------------------------------------------------------------------------
# 1. Стан сервера
# ---------------------------------------------------------------------------

def service_states(conn, server_id: int, now: pd.Timestamp) -> list[dict]:
    """Останній стан кожної служби / пулу і відколи він триває."""
    cut = cutoff(now)
    out = []
    names = conn.execute("SELECT DISTINCT name, kind FROM service_status WHERE server_id = ? AND ts < ? "
                         "ORDER BY kind DESC, name", (server_id, cut)).fetchall()
    for n in names:
        last = conn.execute("SELECT ts, status FROM service_status WHERE server_id = ? AND name = ? "
                            "AND ts < ? ORDER BY ts DESC LIMIT 1", (server_id, n["name"], cut)).fetchone()
        # «відколи» = перший запис поточного стану після останнього запису з іншим станом
        other = conn.execute("SELECT MAX(ts) FROM service_status WHERE server_id = ? AND name = ? "
                             "AND ts < ? AND status != ?",
                             (server_id, n["name"], cut, last["status"])).fetchone()[0]
        since = conn.execute("SELECT MIN(ts) FROM service_status WHERE server_id = ? AND name = ? "
                             "AND ts < ? AND ts > ?",
                             (server_id, n["name"], cut, other or "")).fetchone()[0]
        item = {"name": n["name"], "kind": n["kind"], "status": last["status"],
                "since_local": loc(since), "checked_local": loc(last["ts"])}
        if other is None:
            item["since_note"] = "стан не змінювався з початку даних"
        out.append(item)
    return out


def check_states(conn, server_id: int, now: pd.Timestamp) -> dict:
    """Останній результат кожної перевірки."""
    out = {}
    for t in ("http", "cert", "backup", "sql_port"):
        r = conn.execute("SELECT ts, ok, value, details FROM checks WHERE server_id = ? AND check_type = ? "
                         "AND ts < ? ORDER BY ts DESC LIMIT 1", (server_id, t, cutoff(now))).fetchone()
        if r is not None:
            out[t] = {"ok": bool(r["ok"]), CHECK_VALUE[t]: num(r["value"], 1),
                      "details": localize_text(r["details"]), "checked_local": loc(r["ts"])}
            if t == "backup" and r["value"] is not None:
                # перевірка щогодинна: age_h — вік на час перевірки, age_h_now — на момент now.
                # Якщо перевірка давня (даних немає), age_h_now не рахуємо: ми не знаємо,
                # чи був бекап після неї, — інакше це було б вигадане число
                passed_h = (now - parse_db(r["ts"])).total_seconds() / 3600
                if passed_h <= BACKUP_CHECK_MAX_H:
                    out[t]["age_h_now"] = num(r["value"] + max(passed_h, 0), 1)
                else:
                    out[t]["age_h_now"] = None
    return out


def server_status(conn, s, now) -> dict:
    t, live = resolve_now(conn, s, now)
    fresh = freshness(conn, s["id"], t)
    until = last_ts_before(conn, s["id"], t)
    # Кожна метрика — останнє значення не пізніше now (а не лише в хвилину data_until:
    # якщо якоїсь метрики в цю хвилину немає, беремо її попереднє значення)
    values, older = {}, []
    for m in metric_names(conn, s["id"]):
        r = conn.execute("SELECT ts, value FROM metrics_clean WHERE server_id = ? AND metric = ? AND ts < ? "
                         "ORDER BY ts DESC LIMIT 1", (s["id"], m, cutoff(t))).fetchone()
        if r is None:
            continue
        values[m] = r["value"]
        if r["ts"] != until:
            older.append(f"{m}: останнє значення о {loc(r['ts'])}")
    # метрики сервера — без «:» (disk_used_pct:C: і disk_free_gb:C: ідуть у disks)
    metrics = {m: num(v) for m, v in sorted(values.items()) if ":" not in m}
    disks = []
    for d in disks_of(sorted(values)):
        tot = conn.execute("SELECT total_gb FROM disk_usage WHERE server_id = ? AND drive = ? AND ts < ? "
                           "ORDER BY ts DESC LIMIT 1", (s["id"], d, cutoff(t))).fetchone()
        disks.append({"drive": d, "used_pct": num(values.get(f"disk_used_pct:{d}"), 1),
                      "free_gb": num(values.get(f"disk_free_gb:{d}"), 1),
                      "total_gb": num(tot["total_gb"], 1) if tot else None})
    services = service_states(conn, s["id"], t)
    checks = check_states(conn, s["id"], t)
    notes = older                       # ситуаційні примітки (показує й дашборд)
    if fresh["stale"]:
        notes.insert(0, f"усі значення — останні відомі на {fresh['data_until']['ts_local']}, "
                        f"{age_text(fresh['data_age_h'])} тому, а не поточні")
    about = []                          # постійні примітки про сервер (лише для LLM)
    missing = [c for c in CHECK_VALUE if c not in checks]
    if missing:
        about.append(f"перевірок {', '.join(missing)} на цьому сервері немає")
    inv = inventory(s)
    return {"server": s["name"], "now": pair(t), "live": live, **fresh,
            "metrics": metrics, "disks": disks, "services": services,
            "not_running": [x["name"] for x in services if x["status"] != "running"],
            "checks": checks,
            "sites": [{"site": x.get("name"), "app_pool": x.get("app_pool")} for x in inv.get("sites", [])],
            "notes": notes, "about": about}


@tool
def get_status(server: str | None = None, now: str | None = None) -> dict:
    """Поточний стан одного сервера або всіх.

    Коли викликати: «Що з серверами?», «Що з sim-1?», «Чи працює ApiPool?»,
    «Коли закінчується сертифікат?», «Коли був останній бекап?».
    Параметри: server — назва (None — усі сервери); now — момент «зараз»,
    місцевий час «2026-11-10 08:00» (None — поточний час).
    Повертає: data_until (час останніх даних), data_age_h (скільки годин тому),
    stale (true — даних немає довше 10 хв: тоді всі значення — останні відомі, а не
    поточні, див. stale_note), metrics (CPU, RAM, I/O, мережа, сайт), disks (used_pct,
    free_gb, total_gb), services (стан і відколи), not_running, checks (http:
    response_ms; cert: days_left; backup: age_h на час перевірки і age_h_now на now;
    sql_port: response_ms; ok і час перевірки), sites (сайт -> пул IIS),
    notes (ситуаційні примітки), about (постійні факти про сервер).
    Так само в усіх інструментах: data_until, data_age_h, stale, stale_note.
    """
    with connect() as conn:
        if server:
            return server_status(conn, find_server(conn, server), now)
        out = []
        for s in conn.execute("SELECT * FROM servers ORDER BY id").fetchall():
            try:
                out.append(server_status(conn, s, now))
            except ToolError as e:
                out.append({"server": s["name"], "error": str(e)})
        return {"servers": out}


# ---------------------------------------------------------------------------
# 2. Історія метрики
# ---------------------------------------------------------------------------

def window(t_now: pd.Timestamp, start, end, hours: float = DEFAULT_WINDOW_H) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Вікно (start, end]: за замовчуванням hours до now; end не пізніше now."""
    end_t = to_utc(end) or t_now
    end_t = min(end_t, t_now)
    start_t = to_utc(start) or end_t - pd.Timedelta(hours=hours)
    if start_t >= end_t:
        raise ToolError(f"початок періоду ({loc(start_t)}) не раніше за кінець ({loc(end_t)})")
    return start_t, end_t


def thin(s: pd.Series, max_points: int) -> tuple[pd.Series, int]:
    """Проріджує ряд до max_points: середнє по рівних відрізках часу.
    Повертає (ряд, крок у хвилинах); 1 — ряд без змін."""
    if len(s) <= max_points:
        return s, 1
    span_min = (s.index[-1] - s.index[0]).total_seconds() / 60 + 1
    step = math.ceil(span_min / max_points)
    return s.resample(f"{step}min", origin=s.index[0]).mean().dropna(), step


def pick_points(df: pd.DataFrame, mask: pd.Series, limit: int, median: float) -> list:
    """Позначені точки [час, значення]; якщо їх більше за limit — найдальші від медіани."""
    sel = df[mask]
    if len(sel) > limit:
        sel = sel.loc[(sel["value"] - median).abs().sort_values(ascending=False).index[:limit]].sort_index()
    return [[loc(ts), num(v)] for ts, v in sel["value"].items()]


@tool
def get_metric_history(server: str, metric: str, start: str | None = None, end: str | None = None,
                       now: str | None = None, max_points: int = 50) -> dict:
    """Історія однієї метрики за період (очищені дані після ETL).

    Коли викликати: «Що було вчора о 14:00 на sim-2?», «Який був максимум CPU за ніч?»,
    «Як змінювався диск E: за тиждень?».
    Параметри: metric — cpu_pct, ram_pct, disk_read_mbs, disk_write_mbs, net_in_mbs,
    net_out_mbs, app_response_ms, app_error_pct, disk_used_pct:<диск>, disk_free_gb:<диск>
    (наприклад disk_used_pct:E:); start, end — місцевий час (за замовчуванням 24 год до now);
    now — момент «зараз» (None — поточний час); max_points — скільки точок у series.
    Повертає: first, last, change, mean, min і max (з часом), points (скільки хвилин є),
    imputed (заповнено інтерполяцією), outliers (позначено викидами) і outlier_points,
    gaps (пропуски даних від–до), series — проріджений ряд [місцевий час, значення]
    (середнє по відрізках step_min хвилин).
    """
    max_points = as_number(max_points, "max_points", 2)
    with connect() as conn:
        s = find_server(conn, server)
        t_now, _ = resolve_now(conn, s, now)
        fresh = freshness(conn, s["id"], t_now)
        names = metric_names(conn, s["id"])
        if metric not in names:
            raise ToolError(f"невідома метрика «{metric}» для {server}", known_metrics=names)
        start_t, end_t = window(t_now, start, end)
        df = pd.read_sql_query(
            "SELECT ts, value, is_imputed, is_outlier FROM metrics_clean WHERE server_id = ? AND metric = ? "
            "AND ts > ? AND ts < ? ORDER BY ts", conn,
            params=[s["id"], metric, db_ts(start_t), cutoff(end_t)])
        if df.empty:
            near = conn.execute("SELECT MAX(ts) FROM metrics_clean WHERE server_id = ? AND metric = ? AND ts <= ?",
                                (s["id"], metric, db_ts(start_t))).fetchone()[0]
            raise ToolError(f"немає даних {metric} за період {loc(start_t)} – {loc(end_t)} (місц.); "
                            f"останні — {loc(near) or 'немає'}", nearest_data_local=loc(near), **fresh)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    v = df["value"]
    # Пропуски: from/to — останні дані перед пропуском і перші після; на краях вікна —
    # сама межа вікна (edge: start / end), щоб чесно сказати «немає даних за 14:00»
    minute = pd.Timedelta(minutes=1)
    gaps = []
    if v.index[0] > start_t + minute:
        gaps.append({"from_local": loc(start_t), "to_local": loc(v.index[0]),
                     "missing_min": int((v.index[0] - start_t) / minute) - 1, "edge": "start"})
    diffs = v.index.to_series().diff()
    for ts, d in diffs[diffs > minute].items():
        gaps.append({"from_local": loc(ts - d), "to_local": loc(ts), "missing_min": int(d / minute) - 1})
    notes = []
    if v.index[-1] < end_t:
        gaps.append({"from_local": loc(v.index[-1]), "to_local": loc(end_t),
                     "missing_min": int((end_t - v.index[-1]) / minute), "edge": "end"})
        if (end_t - v.index[-1]) / minute > STALE_MIN:
            notes.append(f"дані закінчуються о {loc(v.index[-1])}; до кінця періоду ({loc(end_t)}) "
                         f"даних немає — усі числа лише за наявні дані")
    median = float(v.median())
    mark_limit = max(10, max_points // 5)
    series, step_min = thin(v, max_points)
    return {"server": server, "metric": metric, "unit": unit_of(metric),
            "from_local": loc(start_t), "to_local": loc(end_t),
            "first": num(v.iloc[0]), "last": num(v.iloc[-1]), "change": num(v.iloc[-1] - v.iloc[0]),
            "mean": num(v.mean()),
            "min": {"value": num(v.min()), "ts_local": loc(v.idxmin())},
            "max": {"value": num(v.max()), "ts_local": loc(v.idxmax())},
            "points": int(len(v)), "imputed": int(df["is_imputed"].sum()),
            "outliers": int(df["is_outlier"].sum()),
            "outlier_points": pick_points(df, df["is_outlier"] == 1, mark_limit, median),
            "imputed_points": pick_points(df, df["is_imputed"] == 1, mark_limit, median),
            "gaps": gaps[:10], "gaps_total": len(gaps),
            "step_min": step_min,
            "series": [[loc(ts), num(x)] for ts, x in series.items()],
            "last_point_local": loc(v.index[-1]), **fresh, "notes": notes}


# ---------------------------------------------------------------------------
# 3. Процеси
# ---------------------------------------------------------------------------

@tool
def get_top_processes(server: str, now: str | None = None, by: str = "cpu", top: int = 5) -> dict:
    """Процеси з найбільшим навантаженням у знімку, найближчому до моменту now.

    Коли викликати: «Хто вантажить CPU на sim-1?», «Хто їсть пам'ять?», «Що вантажило о 14:07?».
    Параметри: now — момент, місцевий час (None — поточний час); by — "cpu" або "ram";
    top — скільки процесів. Знімок береться не пізніше now і не давніший за 10 хв;
    інакше — помилка «немає знімка».
    Повертає: snapshot_local (час знімка), cpu_server і ram_server_pct (CPU і RAM сервера
    в ту хвилину — щоб порівняти з процесом), processes: name, pid, cpu_pct (0–100 % від
    усього CPU), ram_mb, app_pool (для w3wp.exe), service (служба, якщо цей exe належить
    лише одній службі за інвентарем).
    """
    if by not in ("cpu", "ram"):
        raise ToolError(f"by має бути cpu або ram, а не «{by}»")
    top = as_number(top, "top", 1)
    with connect() as conn:
        s = find_server(conn, server)
        t, _ = resolve_now(conn, s, now)
        fresh = freshness(conn, s["id"], t)
        lo = db_ts(t - pd.Timedelta(minutes=SNAPSHOT_MAX_MIN))
        snap = conn.execute("SELECT MAX(ts) FROM process_snapshots WHERE server_id = ? AND ts < ? AND ts >= ?",
                            (s["id"], cutoff(t), lo)).fetchone()[0]
        if snap is None:
            near = conn.execute("SELECT MAX(ts) FROM process_snapshots WHERE server_id = ? AND ts < ?",
                                (s["id"], cutoff(t))).fetchone()[0]
            raise ToolError(f"немає знімка процесів за {SNAPSHOT_MAX_MIN} хв до {loc(t)} (місц.); "
                            f"останній — {loc(near) or 'немає'}", nearest_snapshot_local=loc(near), **fresh)
        order = "cpu_pct" if by == "cpu" else "ram_mb"
        rows = conn.execute(f"SELECT name, pid, cpu_pct, ram_mb, app_pool FROM process_snapshots "
                            f"WHERE server_id = ? AND ts = ? ORDER BY {order} DESC LIMIT ?",
                            (s["id"], snap, top)).fetchall()
        minute = db_ts(parse_db(snap).floor("min"))
        m = {r["metric"]: r["value"] for r in conn.execute(
            "SELECT metric, value FROM metrics_clean WHERE server_id = ? AND ts = ? "
            "AND metric IN ('cpu_pct', 'ram_pct')", (s["id"], minute))}
    # exe -> служба, лише якщо exe належить одній службі (svchost.exe — кільком)
    owners = {}
    for x in inventory(s).get("services", []):
        owners.setdefault(x["exe"].lower(), []).append(x["name"])
    procs = [{"name": r["name"], "pid": r["pid"], "cpu_pct": num(r["cpu_pct"], 1),
              "ram_mb": num(r["ram_mb"], 0), "app_pool": r["app_pool"],
              "service": owners[r["name"].lower()][0] if len(owners.get(r["name"].lower(), [])) == 1 else None}
             for r in rows]
    return {"server": server, "snapshot_local": loc(snap), "by": by,
            "cpu_server": num(m.get("cpu_pct"), 1), "ram_server_pct": num(m.get("ram_pct"), 1),
            "processes": procs, **fresh}


# ---------------------------------------------------------------------------
# 4. Ріст тек
# ---------------------------------------------------------------------------

def sizes_at(conn, server_id: int, t: pd.Timestamp) -> dict:
    """Останній розмір кожної теки не пізніше t: {шлях: (час, ГБ)}."""
    rows = conn.execute("SELECT path, MAX(ts) AS ts, size_gb FROM folder_sizes WHERE server_id = ? AND ts < ? "
                        "GROUP BY path", (server_id, cutoff(t))).fetchall()
    return {r["path"]: (r["ts"], r["size_gb"]) for r in rows}


def free_at(conn, server_id: int, drive: str, t: pd.Timestamp):
    r = conn.execute("SELECT value FROM metrics_clean WHERE server_id = ? AND metric = ? AND ts < ? "
                     "ORDER BY ts DESC LIMIT 1", (server_id, f"disk_free_gb:{drive}", cutoff(t))).fetchone()
    return r[0] if r else None


@tool
def get_folder_growth(server: str, period_h: float = 24, drive: str | None = None,
                      now: str | None = None) -> dict:
    """Як змінився розмір відстежуваних тек за період і що росте на дисках.

    Коли викликати: «Що там росте?», «Яка тека виросла за добу?», «Чим зайнятий диск D:?».
    Параметри: period_h — період у годинах до now (за замовчуванням 24); drive — лише
    один диск, наприклад "E:"; now — момент «зараз» (None — поточний час).
    Повертає: folders — по кожній теці size_start_gb, size_now_gb, growth_gb,
    growth_gb_per_day (від найбільшого росту); disks — по кожному диску growth_gb (ріст
    зайнятого місця зі зміни вільного), folders_gb (сума тек) і other_gb (росте щось
    поза відстежуваними теками).
    """
    period_h = as_number(period_h, "period_h", 0, integer=False, strict=True)
    with connect() as conn:
        s = find_server(conn, server)
        t, _ = resolve_now(conn, s, now)
        fresh = freshness(conn, s["id"], t)
        t0 = t - pd.Timedelta(hours=period_h)
        # «Зараз» для кожної теки — останній знімок усередині вікна (t0, t]. Знімок давніший
        # за t0 не годиться: інакше обидва знімки були б однакові і ріст «0» вводив би в оману
        now_sizes = {p: v for p, v in sizes_at(conn, s["id"], t).items() if parse_db(v[0]) > t0}
        if not now_sizes:
            last = conn.execute("SELECT MAX(ts) FROM folder_sizes WHERE server_id = ? AND ts < ?",
                                (s["id"], cutoff(t))).fetchone()[0]
            raise ToolError(f"немає даних про теки за період {loc(t0)} – {loc(t)} (місц.); "
                            f"останні — {loc(last) or 'немає'}", **fresh)
        start_sizes = sizes_at(conn, s["id"], t0)
        notes = []                       # ситуаційні примітки
        about = ["агент знає лише розмір відстежуваних тек, а не окремі файли"]
        last_snap = max(parse_db(v[0]) for v in now_sizes.values())
        if (t - last_snap).total_seconds() / 60 > STALE_MIN:
            notes.append(f"дані закінчуються о {loc(last_snap)}: ріст пораховано до цього часу, "
                         f"а не до {loc(t)}")
        folders = []
        for path, (ts1, size1) in now_sizes.items():
            d = path[:2].upper()
            if drive and d != drive.upper():
                continue
            if path in start_sizes:
                ts0, size0 = start_sizes[path]
            else:
                # період починається раніше за перші дані — беремо перший запис
                r = conn.execute("SELECT ts, size_gb FROM folder_sizes WHERE server_id = ? AND path = ? "
                                 "ORDER BY ts LIMIT 1", (s["id"], path)).fetchone()
                ts0, size0 = r["ts"], r["size_gb"]
                notes.append(f"{path}: дані лише з {loc(ts0)}")
            hours = (parse_db(ts1) - parse_db(ts0)).total_seconds() / 3600
            growth = size1 - size0
            folders.append({"path": path, "drive": d, "size_start_gb": num(size0), "size_now_gb": num(size1),
                            "growth_gb": num(growth), "measured_h": num(hours, 1),
                            "growth_gb_per_day": num(growth / hours * 24) if hours > 0 else None})
        folders.sort(key=lambda x: x["growth_gb"] or 0, reverse=True)
        disks = []
        for d in sorted({f["drive"] for f in folders} |
                        set(disks_of(metric_names(conn, s["id"])))):
            if drive and d != drive.upper():
                continue
            # ріст диска — лише якщо метрики диска є всередині вікна (інакше f0 = f1 і «0»)
            metrics_in_window = parse_db(last_ts_before(conn, s["id"], t)) > t0
            f0, f1 = free_at(conn, s["id"], d, t0), free_at(conn, s["id"], d, t)
            growth = (f0 - f1) if f0 is not None and f1 is not None and metrics_in_window else None
            in_folders = sum(f["growth_gb"] or 0 for f in folders if f["drive"] == d)
            disks.append({"drive": d, "growth_gb": num(growth), "folders_gb": num(in_folders),
                          "other_gb": num(growth - in_folders) if growth is not None else None})
    return {"server": server, "from_local": loc(t0), "to_local": loc(t),
            "folders": folders, "disks": disks, **fresh, "notes": notes, "about": about}


# ---------------------------------------------------------------------------
# 5. Прогноз диска (обгортка над core/forecast.py, метод не змінюємо)
# ---------------------------------------------------------------------------

def level(v) -> dict:
    """(значення, нижня, верхня межа) -> словник."""
    return {"pct": num(v[0], 1), "low": num(v[1], 1), "high": num(v[2], 1)}


def trend_line(tr: dict, now: pd.Timestamp) -> list:
    """Точки прямої тренду для графіка: від початку тренду (якщо не було зсуву) до +7 днів."""
    line, shift = tr["line"], tr.get("shift", 0.0)
    past_h = 0 if tr.get("stepped_rises") else -(now - tr["start"]).total_seconds() / 3600
    out = []
    for h in np.linspace(past_h, 168, 60):
        v, lo, hi = forecast.clip_pct(tuple(x + shift for x in line.at(h)))
        out.append([loc(now + pd.Timedelta(hours=h)), num(v, 2), num(lo, 2), num(hi, 2)])
    return out


def disk_answer(drive: str, r: dict, now: pd.Timestamp, cfg: dict, with_line: bool) -> dict:
    tr, fg, j = r["trend"], r["fast_growth"], r["jump"]
    trend_out = {"status": tr.get("status"), "from_local": loc(tr.get("start")),
                 "hours_to_threshold": num(tr.get("hours_to_threshold"), 1)}
    if tr.get("status") == "ok":
        trend_out.update({"pct_per_day": num(tr["speed_pct_day"]), "gb_per_day": num(tr["speed_gb_day"], 1),
                          "in_24h": level(tr["pred"][24]), "in_7d": level(tr["pred"][168])})
    notes = []
    if tr.get("excluded"):
        e = tr["excluded"]
        notes.append(f"у тренд не взято тимчасовий регулярний стрибок з {loc(e['from'])} "
                     f"(+{e['size_gb']:.0f} ГБ): зазвичай його видаляють до {loc(e['expected_removal_by'])}")
    if tr.get("excluded_growth_from") is not None:
        notes.append(f"у тренд не взято швидкий ріст з {loc(tr['excluded_growth_from'])} (див. fast_growth)")
    if tr.get("stepped_rises"):
        notes.append(f"завершені різкі підйоми ({tr['stepped_rises']}) враховано як сходинку, а не ріст")
    fg_out = {"status": fg.get("status")}
    if fg.get("status") == "ok":
        fg_out.update({"window_min": cfg["fast_growth"]["window_min"], "pct_per_h": num(fg["speed_pct_h"]),
                       "usual_max_pct_per_h": num(fg["usual_pct_h"]), "limit_pct_per_h": num(fg["limit_pct_h"]),
                       "unusual": fg["unusual"], "in_1h": level(fg["pred"][1]),
                       "hours_to_threshold": num(fg["hours_to_threshold"], 1)})
    if j["found"]:
        j_out = {"found": True, "next_local": loc(j["at"]), "hours_until": num(j["hours_until"], 1),
                 "size_gb": num(j["size_gb"], 1), "days_seen": j["days_seen"],
                 "lookback_days": cfg["jump"]["lookback_days"], "free_then_gb": num(j["free_at_gb"], 1),
                 "fits": j["fits"], "short_gb": num(j["short_gb"], 1),
                 "free_to_stay_below_gb": num(j["free_to_stay_below_gb"], 1),
                 "level_after_pct": num(j["level_after_pct"], 1), "text": j["text"]}
    else:
        j_out = {"found": False, "text": j["text"]}
    # Відстань до порогу: у п. п. і в ГБ (розмір диска = вільно / (1 − зайнято))
    thr, used, free = cfg["threshold_pct"], r["now_pct"], r["now_free_gb"]
    total = free / (1 - used / 100) if used < 99.9 else None
    out = {"drive": drive, "now_pct": num(used, 1), "now_free_gb": num(free, 1),
           "to_threshold_pp": num(thr - used, 1),
           "to_threshold_gb": num((thr - used) / 100 * total, 1) if total else None,
           "trend": trend_out, "fast_growth": fg_out, "jump": j_out, "notes": notes}
    if with_line and tr.get("status") == "ok":
        out["trend_line"] = trend_line(tr, now)
    return out


@tool
def get_disk_forecast(server: str, drive: str | None = None, now: str | None = None,
                      with_line: bool = False) -> dict:
    """Прогноз заповнення диска (лінійний метод, три питання).

    Коли викликати: «Коли заповниться E:?», «Чи поміститься бекап?», «На скільки вистачить диска?».
    Параметри: drive — диск, наприклад "E:" (None — усі диски); now — момент «зараз»
    (None — поточний час); with_line — додати точки прямої тренду для графіка.
    Повертає по кожному диску:
    - trend — повільний ріст від останнього прибирання: pct_per_day, gb_per_day,
      рівень in_24h і in_7d з інтервалом 10–90 % (low–high), hours_to_threshold
      (годин до порогу; 0 — вже вище; null — не досягне);
    - fast_growth — ріст за останні 90 хв: pct_per_h, звичайний максимум у ці хвилини
      доби за 7 днів, unusual (чи швидший за межу), hours_to_threshold;
    - jump — регулярний стрибок (напр. щоденний бекап): next_local, size_gb, free_then_gb,
      fits (поміститься), short_gb (фізично бракує), free_to_stay_below_gb (скільки звільнити,
      щоб лишитися нижче порогу).
    threshold_pct — поріг (90 %); to_threshold_pp і to_threshold_gb — скільки лишилось до порогу.
    Якщо даних свіжих немає (stale) — прогнозу на «зараз» немає (disks відсутнє), а
    forecast_at_last_data — прогноз на момент останніх даних, НЕ на зараз.
    """
    cfg = forecast.load_config()
    with connect() as conn:
        s = find_server(conn, server)
        t, _ = resolve_now(conn, s, now)
        fresh = freshness(conn, s["id"], t)
        drives = forecast.list_disks(conn, s["id"])
        if drive:
            if drive.upper() not in drives:
                raise ToolError(f"на {server} немає диска «{drive}»", known_disks=drives)
            drives = [drive.upper()]
        # Немає свіжих даних — прогноз рахуємо на момент останніх даних, а не на «зараз»:
        # інакше «+24 год» і «регулярних стрибків немає» стосувалися б часу, коли даних немає
        at = parse_db(last_ts_before(conn, s["id"], t)) if fresh["stale"] else t
        method = forecast.get_method(cfg["method"])
        out = []
        for d in drives:
            used, free = forecast.load_disk(conn, s["id"], d, at, cfg)
            if len(used) < 2 or free.empty:
                out.append({"drive": d, "error": "замало даних для прогнозу"})
                continue
            out.append(disk_answer(d, method.predict(used, free, at, cfg), at, cfg, with_line))
    result = {"server": server, "now": pair(t), "method": cfg["method"],
              "threshold_pct": cfg["threshold_pct"], **fresh}
    if fresh["stale"]:
        result["message"] = (f"немає свіжих даних; останні — {fresh['data_until']['ts_local']} "
                             f"({age_text(fresh['data_age_h'])} тому). Прогнозу на «зараз» немає")
        result["forecast_at_last_data"] = {
            "at": pair(at), "note": f"прогноз на момент останніх даних ({loc(at)}), а НЕ на зараз",
            "disks": out}
    else:
        result["disks"] = out
    return result


# ---------------------------------------------------------------------------
# 6. Хронологія
# ---------------------------------------------------------------------------

def state_changes(rows, key: str, value: str) -> list:
    """Зміни значення в записах, впорядкованих за часом: [(рядок, старе значення)]."""
    last, out = {}, []
    for r in rows:
        k = r[key]
        if k in last and last[k] != r[value]:
            out.append((r, last[k]))
        last[k] = r[value]
    return out


@tool
def get_timeline(server: str, start: str | None = None, end: str | None = None,
                 now: str | None = None, limit: int = 50) -> dict:
    """Хронологія подій сервера за період: журнал Windows, зміни стану служб і перевірок, алерти.

    Коли викликати: «Що сталося на sim-1 за ніч?», «Чому впав сайт?», «Що було перед падінням пулу?».
    Параметри: start, end — місцевий час (за замовчуванням 24 год до now); now — момент
    «зараз» (None — поточний час); limit — скільки записів (якщо більше — останні).
    Повертає items у хронологічному порядку: ts_local, type (event — подія журналу рівня
    Warning і вище, а також 7036 і 18264; service — зміна стану служби чи пулу; check —
    зміна результату перевірки; alert — алерт), source, code, level, text.
    noise — часті шумові події (1014, 37) лише кількістю; counts — кількість за типами.
    """
    limit = as_number(limit, "limit", 1)
    with connect() as conn:
        s = find_server(conn, server)
        t_now, _ = resolve_now(conn, s, now)
        fresh = freshness(conn, s["id"], t_now)
        start_t, end_t = window(t_now, start, end)
        # Дані сервера закінчились до початку вікна — порожній список вводив би в оману
        # («нічого не сталося»), тому кажемо прямо: немає даних
        last_raw = last_ts_before(conn, s["id"], end_t)
        if last_raw is None or parse_db(last_raw) <= start_t:
            raise ToolError(f"немає даних за період {loc(start_t)} – {loc(end_t)} (місц.); "
                            f"останні — {loc(last_raw) or 'немає'}", **fresh)
        last = parse_db(last_raw)
        notes = []
        if (end_t - last).total_seconds() / 60 > STALE_MIN:
            notes.append(f"дані закінчуються о {loc(last)}: після цього подій не видно не тому, "
                         f"що їх не було, а тому, що даних немає")
        lo, hi = db_ts(start_t), cutoff(end_t)
        items, noise = [], {}
        for r in conn.execute("SELECT ts, log, source, event_id, level, message FROM events "
                              "WHERE server_id = ? AND ts > ? AND ts < ? ORDER BY ts", (s["id"], lo, hi)):
            if r["event_id"] in NOISE_EVENTS:
                noise[str(r["event_id"])] = noise.get(str(r["event_id"]), 0) + 1
                continue
            if r["level"] in TIMELINE_LEVELS or r["event_id"] in INFO_EVENTS:
                items.append({"ts": r["ts"], "type": "event", "source": r["source"], "code": r["event_id"],
                              "level": r["level"], "text": (r["message"] or "")[:TEXT_MAX]})
        # Зміни стану: спершу останній запис перед вікном (prev), щоб побачити зміну на його початку
        rows = conn.execute("SELECT ts, name, kind, status FROM service_status WHERE server_id = ? "
                            "AND ts > ? AND ts < ? ORDER BY ts", (s["id"], lo, hi)).fetchall()
        prev = conn.execute("SELECT name, kind, status, MAX(ts) AS ts FROM service_status WHERE server_id = ? "
                            "AND ts <= ? GROUP BY name", (s["id"], lo)).fetchall()
        for r, old in state_changes(list(prev) + list(rows), "name", "status"):
            items.append({"ts": r["ts"], "type": "service", "source": r["name"], "code": r["kind"],
                          "level": None, "text": f"{r['name']} ({r['kind']}): {old} -> {r['status']}"})
        rows = conn.execute("SELECT ts, check_type, ok, value, details FROM checks WHERE server_id = ? "
                            "AND ts > ? AND ts < ? ORDER BY ts", (s["id"], lo, hi)).fetchall()
        prev = conn.execute("SELECT check_type, ok, value, details, MAX(ts) AS ts FROM checks "
                            "WHERE server_id = ? AND ts <= ? GROUP BY check_type", (s["id"], lo)).fetchall()
        for r, old in state_changes(list(prev) + list(rows), "check_type", "ok"):
            state = "ok" if r["ok"] else "не пройшла"
            items.append({"ts": r["ts"], "type": "check", "source": r["check_type"], "code": None,
                          "level": None,
                          "text": f"перевірка {r['check_type']}: {state} ({localize_text(r['details'])}, "
                                  f"{CHECK_VALUE[r['check_type']]} = {num(r['value'], 1)})"})
        for r in conn.execute("SELECT opened_at, rule_id, severity, status, message FROM alerts "
                              "WHERE server_id = ? AND opened_at > ? AND opened_at < ? ORDER BY opened_at",
                              (s["id"], lo, hi)):
            items.append({"ts": r["opened_at"], "type": "alert", "source": r["rule_id"], "code": r["severity"],
                          "level": r["severity"], "text": (r["message"] or "")[:TEXT_MAX]})
    items.sort(key=lambda x: x["ts"])
    counts = {k: sum(1 for x in items if x["type"] == k) for k in ("event", "service", "check", "alert")}
    total = len(items)
    items = items[-limit:]
    # час у відповіді — місцевий, першим полем
    items = [{"ts_local": loc(x["ts"]), **{k: v for k, v in x.items() if k != "ts"}} for x in items]
    if total > len(items):
        notes.append(f"показано останні {len(items)} з {total} записів")
    return {"server": server, "from_local": loc(start_t), "to_local": loc(end_t),
            "items": items, "noise": noise, "counts": counts, **fresh, "notes": notes}


# ---------------------------------------------------------------------------
# Запуск з консолі
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Функції-інструменти: друк JSON")
    parser.add_argument("function", choices=sorted(TOOLS))
    parser.add_argument("--server")
    parser.add_argument("--now", help="момент, місцевий час: \"2026-11-10 08:00\"")
    parser.add_argument("--metric")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--drive")
    parser.add_argument("--by")
    parser.add_argument("--top", type=int)
    parser.add_argument("--period-h", dest="period_h", type=float)
    parser.add_argument("--max-points", dest="max_points", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--with-line", dest="with_line", action="store_true", default=None)
    args = vars(parser.parse_args())
    func = TOOLS[args.pop("function")]
    kwargs = {k: v for k, v in args.items() if v is not None}
    result = func(**kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
