"""Симулятор серверів sim-1 (вебсервер) і sim-2 (сервер БД).

Пише дані в ті самі таблиці і в тому самому форматі, що й агент
(agent/agent.py), тому решта системи не розрізняє реальні й симульовані дані.

Запуск:
    python -m simulator.simulator --history 30            # 30 днів історії до поточного моменту
    python -m simulator.simulator --history 30 --reset    # спершу видалити дані sim-*
    python -m simulator.simulator --plot                  # контрольний графік у docs/experiments/
    python -m simulator.simulator --incidents             # список інцидентів з sim_incidents
    python -m simulator.simulator --plot-incident 5       # графік інциденту №5 (±2 год, по хвилинах)

Живий потік (--live) і ручний запуск сценаріїв (--scenario) — кроки 1.6–1.7.

Профілі серверів — у simulator/profiles.yaml, сценарії — у simulator/scenarios.py.
"""

import argparse
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from core.db import db_session, init_db
from simulator.scenarios import SCENARIOS, event

PROFILES_PATH = Path(__file__).resolve().parent / "profiles.yaml"
PLOT_PATH = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "sim_history.png"

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Недосконалості даних (щоб ETL було що виправляти)
MISSING_PROB = 0.005                    # ~0,5 % поодиноких пропущених хвилин
LONG_GAPS = {"sim-1": 1, "sim-2": 2}    # довгі пропуски по 5–15 хв («агент перезапускався»)
CORRUPT_PER_SERVER = 2                  # неможливі значення («збій агента»)

# Фонові «хибні тривоги» без мітки: короткі сплески CPU, у середньому разів на добу
BURST_PER_DAY = 1.5

# Фонові події журналу, що не є інцидентами: (разів на добу, подія)
NOISE_EVENTS = [
    (2.0, event("System", "Microsoft-Windows-DNS-Client", 1014, "Warning",
                "Name resolution for the name updates.example.com timed out "
                "after none of the configured DNS servers responded.")),
    (1.0, event("System", "Microsoft-Windows-Time-Service", 37, "Information",
                "The time provider NtpClient is currently receiving valid time data "
                "from time.example.com.")),
]


def fmt(t: datetime) -> str:
    return t.strftime(TS_FORMAT)


def r(value, digits):
    """Округлення, яке пропускає None."""
    return None if value is None else round(value, digits)


def load_profiles() -> dict:
    with open(PROFILES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def local_day_start(t: datetime, tz: timedelta) -> datetime:
    """Початок місцевої доби (00:00 за місцевим часом), виражений в UTC."""
    local = t + tz
    return datetime(local.year, local.month, local.day, tzinfo=timezone.utc) - tz


# ---------------------------------------------------------------------------
# Стан сервера: усе, що «пам'ятається» між хвилинами
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self, name: str, profile: dict, tz: timedelta, rng: random.Random):
        self.name = name
        self.profile = profile
        self.tz = tz
        self.now: datetime | None = None   # поточна хвилина симуляції (UTC)

        self.service_status = {s["name"]: "running" for s in profile["services"]}
        self.folder_gb = {f["path"]: float(f.get("start_gb", 0)) for f in profile["folders"]}
        self.drive_other_gb = {d: float(v["other_gb"]) for d, v in profile["drives"].items()}

        cert = profile.get("checks", {}).get("cert")
        self.cert_days = float(cert["normal_days_left"]) if cert else None

        # Бекапи (лише для сервера БД)
        self.backups: list[tuple[datetime, float]] = []   # (коли закінчився, розмір ГБ)
        self.last_backup_end: datetime | None = None
        self.last_backup_ok = True
        self.backup_fail_next = False
        self.backup_running_until: datetime | None = None

        # Короткий сплеск CPU (не інцидент)
        self.burst_left = 0
        self.burst_cpu = 0.0

        # Множник навантаження на кожен день (дні трохи відрізняються)
        self.daily_mult: dict = {}

        # PID кожного процесу; при перезапуску — новий
        self.pids = {self.proc_key(p): rng.randint(1000, 30000) for p in profile["processes"]}

    # --- процеси ---
    @staticmethod
    def proc_key(proc: dict) -> tuple:
        return (proc["name"], proc.get("app_pool") or proc.get("service") or "")

    def find_process(self, target: dict) -> dict | None:
        """Процес профілю за ціллю сценарію: {service} або {process, app_pool}."""
        for p in self.profile["processes"]:
            if target.get("service") and p.get("service") == target["service"]:
                return p
            if (target.get("process") == p["name"]
                    and (target.get("app_pool") is None or target.get("app_pool") == p.get("app_pool"))):
                return p
        return None

    def pid_for_target(self, target: dict) -> int | None:
        proc = self.find_process(target)
        return self.pids.get(self.proc_key(proc)) if proc else None

    def restart_target(self, target: dict, rng: random.Random) -> None:
        proc = self.find_process(target)
        if proc:
            self.pids[self.proc_key(proc)] = rng.randint(1000, 30000)

    # --- диски і теки ---
    def folder_drive(self, path: str) -> str:
        for f in self.profile["folders"]:
            if f["path"] == path:
                return f["drive"]
        raise KeyError(f"Теки {path} немає в профілі {self.name}")

    def drive_total_gb(self, drive: str) -> float:
        return float(self.profile["drives"][drive]["total_gb"])

    def drive_used_gb(self, drive: str) -> float:
        folders = sum(size for path, size in self.folder_gb.items() if self.folder_drive(path) == drive)
        return self.drive_other_gb[drive] + folders

    def drive_used_pct(self, drive: str) -> float:
        return 100 * self.drive_used_gb(drive) / self.drive_total_gb(drive)

    # --- бекапи ---
    def sync_backup_folder(self) -> None:
        b = self.profile["backup"]
        self.folder_gb[b["folder"]] = sum(size for _, size in self.backups)

    def complete_backup(self) -> list[dict]:
        """Успішний бекап щойно закінчився: новий файл у теці бекапів."""
        b = self.profile["backup"]
        size = self.folder_gb[b["data_folder"]] * b["size_ratio"]
        self.backups.append((self.now, size))
        self.last_backup_end = self.now
        self.last_backup_ok = True
        self.backup_running_until = None
        self.sync_backup_folder()
        return [event("Application", "MSSQLSERVER", 18264, "Information",
                      f"Database backed up. Database: {b['database']}, "
                      f"size: {size:.1f} GB, file: {b['folder']}\\{b['database']}.bak")]


# ---------------------------------------------------------------------------
# Нормальна поведінка
# ---------------------------------------------------------------------------

def load_factor(state: ServerState, local: datetime, rng: random.Random) -> float:
    """Навантаження 0..1: вночі 0, у робочий час до 1 (пік ~14:30), вихідні нижче."""
    hour = local.hour + local.minute / 60
    day = math.sin(math.pi * (hour - 7) / 15) if 7 <= hour <= 22 else 0.0
    if local.weekday() >= 5:
        day *= state.profile["cpu"]["weekend_factor"]
    date = local.date()
    if date not in state.daily_mult:
        state.daily_mult[date] = rng.uniform(0.9, 1.1)
    return day * state.daily_mult[date]


def normal_values(state: ServerState, t: datetime, rng: random.Random) -> dict:
    """Значення хвилини t без інцидентів."""
    p = state.profile
    load = load_factor(state, t + state.tz, rng)

    cpu = p["cpu"]["night"] + (p["cpu"]["day_peak"] - p["cpu"]["night"]) * load
    cpu += rng.gauss(0, p["cpu"]["noise"])

    # Короткі сплески CPU на 1–3 хв — нормальне явище, не інцидент
    if state.burst_left > 0:
        cpu = max(cpu, state.burst_cpu)
        state.burst_left -= 1
    elif rng.random() < BURST_PER_DAY / 1440:
        state.burst_left = rng.randint(1, 3) - 1
        state.burst_cpu = rng.uniform(70, 97)
        cpu = max(cpu, state.burst_cpu)

    ram = p["ram"]["base"] + p["ram"]["load_gain"] * load + rng.gauss(0, p["ram"]["noise"])
    read = 0.3 + p["disk_io"]["read_peak"] * load * rng.uniform(0.7, 1.3)
    write = 0.3 + p["disk_io"]["write_peak"] * load * rng.uniform(0.7, 1.3)
    net_in = 0.2 + p["net"]["in_peak"] * load * rng.uniform(0.8, 1.2)
    net_out = 0.2 + p["net"]["out_peak"] * load * rng.uniform(0.8, 1.2)

    # Нічний бекап: більше CPU і дискових операцій
    if state.backup_running_until is not None:
        b = p["backup"]
        cpu += b["cpu_add"] * rng.uniform(0.8, 1.2)
        read += b["read_mbs"] * rng.uniform(0.8, 1.2)
        write += b["write_mbs"] * rng.uniform(0.8, 1.2)

    values = {
        "cpu_pct": cpu, "ram_pct": ram,
        "disk_read_mbs": read, "disk_write_mbs": write,
        "net_in_mbs": net_in, "net_out_mbs": net_out,
        "app_response_ms": None, "app_error_pct": None,
        "http": None,
        "process_overrides": [],   # зміни процесів від сценаріїв
    }
    app = p.get("app")
    if app:
        values["app_response_ms"] = app["response_base_ms"] + app["response_load_ms"] * load + rng.gauss(0, 10)
        values["app_error_pct"] = app["error_pct"] * rng.uniform(0.3, 1.7)
    if p.get("checks", {}).get("http"):
        base = values["app_response_ms"] or 150
        values["http"] = {"code": 200, "response_ms": base * rng.uniform(0.9, 1.1)}
    return values


def backup_routine(state: ServerState, t: datetime) -> list[dict]:
    """Нічний бекап о 02:00 і видалення старих бекапів (лише для сервера БД)."""
    b = state.profile.get("backup")
    if not b:
        return []
    events = []
    local = t + state.tz

    if local.hour == b["hour"] and local.minute == 0:
        if state.backup_fail_next:
            # Бекап зламано сценарієм backup_stale (подію 3041 записав сценарій)
            state.backup_fail_next = False
        else:
            state.backup_running_until = t + timedelta(minutes=b["duration_min"])

    if state.backup_running_until is not None and t >= state.backup_running_until:
        events += state.complete_backup()

    # Окреме завдання прибирання: видаляє бекапи, старші за строк зберігання
    if local.hour == b["cleanup_hour"] and local.minute == 0:
        keep_after = t - timedelta(days=b["retention_days"])
        state.backups = [(end, size) for end, size in state.backups if end > keep_after]
        state.sync_backup_folder()
    return events


def advance_state(state: ServerState, t: datetime) -> None:
    """Повільні зміни за хвилину: ріст тек і дисків, прибирання логів, строк сертифіката."""
    local = t + state.tz
    weekend = local.weekday() >= 5
    for f in state.profile["folders"]:
        rate = f.get("weekend_gb_per_day" if weekend else "weekday_gb_per_day", 0.0)
        state.folder_gb[f["path"]] += rate / 1440
        cleanup = f.get("weekly_cleanup")
        if (cleanup and local.weekday() == cleanup["weekday"]
                and local.hour == cleanup["hour"] and local.minute == 0):
            state.folder_gb[f["path"]] = min(state.folder_gb[f["path"]], cleanup["keep_gb"])
    for drive, cfg in state.profile["drives"].items():
        state.drive_other_gb[drive] += cfg.get("growth_gb_per_day", 0.0) / 1440
    if state.cert_days is not None:
        state.cert_days -= 1 / 1440


def build_processes(state: ServerState, values: dict, rng: random.Random) -> list[dict]:
    """Топ-5 процесів за CPU з урахуванням зупинених служб і змін від сценаріїв."""
    total = values["cpu_pct"]
    procs = []
    for p in state.profile["processes"]:
        owner = p.get("service") or p.get("app_pool")
        if owner and state.service_status.get(owner) != "running":
            continue   # служба / пул зупинені — процесу немає
        procs.append({
            "name": p["name"], "app_pool": p.get("app_pool"),
            "pid": state.pids[state.proc_key(p)],
            "cpu_pct": total * p["cpu_share"] * rng.uniform(0.8, 1.2),
            "ram_mb": p["ram_mb"] * rng.uniform(0.97, 1.03),
        })
    for o in values["process_overrides"]:
        for proc in list(procs):
            if proc["name"] != o.get("name"):
                continue
            if o.get("app_pool") is not None and proc["app_pool"] != o["app_pool"]:
                continue
            if o.get("hide"):
                procs.remove(proc)
            if "cpu_pct" in o:
                proc["cpu_pct"] = o["cpu_pct"]
            if "ram_add_mb" in o:
                proc["ram_mb"] += o["ram_add_mb"]
    for proc in procs:
        proc["cpu_pct"] = min(proc["cpu_pct"], total)
    procs.sort(key=lambda pr: pr["cpu_pct"], reverse=True)
    return procs[:5]


# ---------------------------------------------------------------------------
# Робота з базою
# ---------------------------------------------------------------------------

# Таблиці, де є server_id (їх чистить --reset для sim-*)
SERVER_TABLES = ["metrics", "disk_usage", "process_snapshots", "folder_sizes", "service_status",
                 "events", "checks", "sim_incidents", "metrics_clean", "anomalies",
                 "predictions", "alerts"]

INSERT_SQL = {
    "metrics": "INSERT INTO metrics (ts, server_id, cpu_pct, ram_pct, disk_read_mbs, disk_write_mbs, "
               "net_in_mbs, net_out_mbs, app_response_ms, app_error_pct, source) "
               "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
    "disk_usage": "INSERT INTO disk_usage (ts, server_id, drive, used_pct, free_gb, total_gb) "
                  "VALUES (?,?,?,?,?,?)",
    "process_snapshots": "INSERT INTO process_snapshots (ts, server_id, name, pid, cpu_pct, ram_mb, app_pool) "
                         "VALUES (?,?,?,?,?,?,?)",
    "folder_sizes": "INSERT INTO folder_sizes (ts, server_id, path, size_gb) VALUES (?,?,?,?)",
    "service_status": "INSERT INTO service_status (ts, server_id, name, kind, status) VALUES (?,?,?,?,?)",
    "events": "INSERT INTO events (ts, server_id, log, source, event_id, level, message) "
              "VALUES (?,?,?,?,?,?,?)",
    "checks": "INSERT INTO checks (ts, server_id, check_type, ok, value, details) VALUES (?,?,?,?,?,?)",
}


def ensure_servers(profiles: dict) -> dict[str, int]:
    """Створює записи серверів sim-* (якщо їх немає) і повертає {назва: id}."""
    ids = {}
    with db_session() as conn:
        for name, prof in profiles["servers"].items():
            conn.execute("INSERT OR IGNORE INTO servers (name, kind, description) VALUES (?, 'sim', ?)",
                         (name, prof["description"]))
            conn.execute("UPDATE servers SET description = ? WHERE name = ?", (prof["description"], name))
            ids[name] = conn.execute("SELECT id FROM servers WHERE name = ?", (name,)).fetchone()["id"]
    return ids


def sim_data_count() -> int:
    """Скільки записів метрик уже є для серверів sim-*."""
    with db_session() as conn:
        return conn.execute("SELECT COUNT(*) FROM metrics WHERE server_id IN "
                            "(SELECT id FROM servers WHERE kind = 'sim')").fetchone()[0]


def reset_sim_data() -> None:
    """Видаляє всі дані серверів sim-* (дані реальних серверів, напр. my-pc, не чіпає)."""
    with db_session() as conn:
        for table in SERVER_TABLES:
            deleted = conn.execute(f"DELETE FROM {table} WHERE server_id IN "
                                   "(SELECT id FROM servers WHERE kind = 'sim')").rowcount
            if deleted:
                print(f"  видалено з {table}: {deleted}")


# ---------------------------------------------------------------------------
# Симуляція: обробка однієї хвилини (спільна для історії і, на кроці 1.6,
# для догенерації та живого режиму)
# ---------------------------------------------------------------------------

class Simulation:
    def __init__(self, profiles: dict, server_ids: dict[str, int], rng: random.Random):
        self.profiles = profiles
        self.tz = timedelta(hours=profiles["timezone_offset_hours"])
        self.rng = rng
        self.server_ids = server_ids
        self.states = {name: ServerState(name, prof, self.tz, rng)
                       for name, prof in profiles["servers"].items()}
        self.active = {name: [] for name in self.states}     # активні інциденти
        self.buffers = {table: [] for table in INSERT_SQL}   # рядки, що чекають запису
        self.last_status = {name: {} for name in self.states}
        self.stats = defaultdict(int)

    def flush(self) -> None:
        with db_session() as conn:
            for table, rows in self.buffers.items():
                if rows:
                    conn.executemany(INSERT_SQL[table], rows)
                    rows.clear()

    def step(self, name: str, t: datetime, minute_index: int,
             starting=(), ending=(), skip=False, corrupt=None) -> None:
        """Одна хвилина одного сервера.

        starting / ending — інциденти, що починаються / закінчуються в цю хвилину;
        skip — агент «пропустив» цю хвилину (пишемо лише події журналу);
        corrupt — вид неможливого значення («збій агента») або None.
        """
        state, rng = self.states[name], self.rng
        sid = self.server_ids[name]
        state.now = t
        events = []

        # 1. Початок інцидентів
        for inc in starting:
            events += SCENARIOS[inc["scenario"]].start(state, inc["params"], rng)
            self.active[name].append(inc)

        # 2. Завершення інцидентів. Інцидент діє з started_at включно до ended_at
        #    не включно: у хвилину ended_at сервер уже в нормі
        for inc in ending:
            events += SCENARIOS[inc["scenario"]].end(state, inc["params"], rng)
            self.active[name].remove(inc)

        # 3. Нічний бекап; 4. нормальні значення; 5. вплив активних інцидентів
        events += backup_routine(state, t)
        values = normal_values(state, t, rng)
        for inc in self.active[name]:
            minute = int((t - inc["start"]).total_seconds() // 60)
            events += SCENARIOS[inc["scenario"]].apply(state, values, minute, inc["params"], rng)

        # 6. Повільні зміни і фонові події
        advance_state(state, t)
        for per_day, ev in NOISE_EVENTS:
            if rng.random() < per_day / 1440:
                events.append(ev)

        # Події пишемо завжди: навіть якщо агент пропустив хвилину,
        # подія лишається в журналі Windows і буде зібрана пізніше
        ts = fmt(t + timedelta(seconds=rng.randint(0, 5)))   # зсув часу запису 0–5 с
        for ev in events:
            self.buffers["events"].append((ts, sid, ev["log"], ev["source"], ev["event_id"],
                                           ev["level"], ev["message"]))
        if skip:
            self.stats["пропущено хвилин"] += 1
            return

        # 7. Метрики (з можливим «збоєм агента»)
        row = [min(max(values["cpu_pct"], 0.5), 100), min(max(values["ram_pct"], 1), 99.5),
               values["disk_read_mbs"], values["disk_write_mbs"], values["net_in_mbs"], values["net_out_mbs"]]
        if corrupt:
            self.stats["неможливих значень"] += 1
            if corrupt == "cpu_over_100":
                row[0] = rng.uniform(101, 110)
            elif corrupt == "ram_over_100":
                row[1] = rng.uniform(100.5, 105)
            elif corrupt == "negative_read":
                row[2] = -rng.uniform(0.1, 2)
            else:
                row[4] = -rng.uniform(0.1, 3)
        self.buffers["metrics"].append((ts, sid, r(row[0], 1), r(row[1], 1), r(row[2], 2), r(row[3], 2),
                                        r(row[4], 2), r(row[5], 2), r(values["app_response_ms"], 0),
                                        r(values["app_error_pct"], 2), "simulator"))

        # 8. Диски — щохвилини
        for drive in state.profile["drives"]:
            total = state.drive_total_gb(drive)
            used = min(state.drive_used_gb(drive), total)
            self.buffers["disk_usage"].append((ts, sid, drive, round(100 * used / total, 1),
                                               round(total - used, 1), total))

        # 9. Процеси — раз на 5 хв; служби — раз на 5 хв і одразу при зміні
        if minute_index % 5 == 0:
            for p in build_processes(state, values, rng):
                self.buffers["process_snapshots"].append((ts, sid, p["name"], p["pid"], round(p["cpu_pct"], 1),
                                                          round(p["ram_mb"], 1), p["app_pool"]))
        for svc in state.profile["services"]:
            status = state.service_status[svc["name"]]
            if minute_index % 5 == 0 or self.last_status[name].get(svc["name"]) != status:
                self.buffers["service_status"].append((ts, sid, svc["name"], svc["kind"], status))
                self.last_status[name][svc["name"]] = status

        # 10. Теки — щогодини
        if minute_index % 60 == 0:
            for path, size in state.folder_gb.items():
                self.buffers["folder_sizes"].append((ts, sid, path, round(size, 2)))

        # 11. Перевірки: HTTP і порт SQL — щохвилини, сертифікат і бекап — щогодини
        checks = state.profile.get("checks", {})
        if values["http"]:
            http, timeout = values["http"], checks["http"]["timeout_ms"]
            ok = http["code"] == 200 and http["response_ms"] <= timeout
            details = f"HTTP {http['code']}"
            if http["response_ms"] > timeout:
                details += f", {http['response_ms']:.0f} мс > таймаут {timeout} мс"
            self.buffers["checks"].append((ts, sid, "http", int(ok), round(http["response_ms"]), details))
        if "sql_port" in checks:
            ok = state.service_status.get("MSSQLSERVER") == "running"
            self.buffers["checks"].append((ts, sid, "sql_port", int(ok), round(rng.uniform(1, 3), 1),
                                           f"TCP {checks['sql_port']['port']}"))
        if minute_index % 60 == 0:
            if "cert" in checks and state.cert_days is not None:
                self.buffers["checks"].append((ts, sid, "cert", int(state.cert_days > 0),
                                               round(state.cert_days, 1), checks["cert"]["subject"]))
            if "backup" in checks and state.last_backup_end is not None:
                age_h = (t - state.last_backup_end).total_seconds() / 3600
                self.buffers["checks"].append((ts, sid, "backup", int(state.last_backup_ok), round(age_h, 1),
                                               f"останній успішний: {fmt(state.last_backup_end)}"))


# ---------------------------------------------------------------------------
# Розклад інцидентів в історії
# ---------------------------------------------------------------------------

def schedule_history_incidents(rng, profiles, start, end) -> list[dict]:
    """Розставляє інциденти так, щоб на одному сервері вони не перекривалися,
    крім ланцюжків «причина → наслідок», які накладаються навмисно.
    """
    tz = timedelta(hours=profiles["timezone_offset_hours"])
    days = (end - start).days
    total_min = int((end - start).total_seconds() // 60)
    incidents = []

    def minutes(m):
        return timedelta(minutes=m)

    for server, prof in profiles["servers"].items():
        busy: list[tuple[datetime, datetime]] = []

        def is_free(s, e, margin=60):
            return all(e + minutes(margin) <= bs or s >= be + minutes(margin) for bs, be in busy)

        def backup_night(day_index):
            """Час нічного бекапу (02:00 місцевого) у день day_index від початку історії."""
            return local_day_start(start, tz) + timedelta(days=day_index, hours=prof["backup"]["hour"])

        def place(propose):
            """Пробує варіанти, поки не знайде вільний проміжок."""
            for _ in range(2000):
                plan = propose()   # список (сценарій, початок, кінець, параметри)
                s = min(item[1] for item in plan)
                e = max(item[2] for item in plan)
                if start + timedelta(hours=12) <= s and e <= end - timedelta(hours=6) and is_free(s, e):
                    for scenario, ps, pe, params in plan:
                        incidents.append({
                            "server": server, "scenario": scenario, "start": ps, "end": pe,
                            "params": dict(params or {}, duration_min=int((pe - ps).total_seconds() // 60)),
                        })
                    busy.append((s, e))
                    return
            raise RuntimeError(f"Не вдалося розмістити інциденти на {server}: забагато для {days} днів")

        counts = dict(prof.get("history_incidents", {}))

        # 1) Сертифікат — найдовший (кілька днів), ставимо першим
        for _ in range(counts.pop("cert_expiring", 0)):
            def propose_cert():
                s = start + minutes(rng.randint(5 * 1440, (days - 10) * 1440))
                return [("cert_expiring", s, s + minutes(rng.randint(3 * 1440, 5 * 1440)), {})]
            place(propose_cert)

        # 2) Ланцюжки «причина → наслідок», що накладаються
        for chain in prof.get("history_chains", []):
            for _ in range(chain["count"]):
                if chain["effect"] == "backup_stale":
                    def propose_chain(chain=chain):
                        night = backup_night(rng.randint(1, days - 3))
                        cause_s = night - minutes(rng.randint(3 * 60, 6 * 60))   # почалось увечері
                        cause_e = night + minutes(rng.randint(4 * 60, 7 * 60))   # прибрали вранці
                        stale_e = night + timedelta(days=1, minutes=prof["backup"]["duration_min"])
                        return [(chain["cause"], cause_s, cause_e, chain.get("cause_params")),
                                ("backup_stale", night, stale_e, chain.get("effect_params"))]
                else:
                    def propose_chain(chain=chain):
                        lo, hi = SCENARIOS[chain["cause"]].duration_min
                        cause_s = start + minutes(rng.randint(0, total_min))
                        cause_e = cause_s + minutes(rng.randint(lo, hi))
                        effect_s = cause_e - minutes(2)   # наслідок починається ще до кінця причини
                        lo, hi = SCENARIOS[chain["effect"]].duration_min
                        effect_e = effect_s + minutes(rng.randint(lo, hi))
                        return [(chain["cause"], cause_s, cause_e, chain.get("cause_params")),
                                (chain["effect"], effect_s, effect_e, chain.get("effect_params"))]
                place(propose_chain)

        # 3) Окремі нічні збої бекапу
        for _ in range(counts.pop("backup_stale", 0)):
            def propose_stale():
                night = backup_night(rng.randint(1, days - 3))
                return [("backup_stale", night,
                         night + timedelta(days=1, minutes=prof["backup"]["duration_min"]), {})]
            place(propose_stale)

        # 4) Решта — у випадковий час, довші першими
        for scenario, n in sorted(counts.items(), key=lambda kv: -SCENARIOS[kv[0]].duration_min[1]):
            for _ in range(n):
                def propose_other(scenario=scenario):
                    lo, hi = SCENARIOS[scenario].duration_min
                    s = start + minutes(rng.randint(0, total_min))
                    return [(scenario, s, s + minutes(rng.randint(lo, hi)), {})]
                place(propose_other)

    incidents.sort(key=lambda inc: inc["start"])
    return incidents


# ---------------------------------------------------------------------------
# Генерація історії
# ---------------------------------------------------------------------------

def generate_history(days: int, seed: int) -> None:
    rng = random.Random(seed)
    profiles = load_profiles()
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    total_min = days * 1440

    init_db()
    server_ids = ensure_servers(profiles)
    incidents = schedule_history_incidents(rng, profiles, start, end)
    sim = Simulation(profiles, server_ids, rng)
    tz = sim.tz

    # --- Початковий стан ---
    for state in sim.states.values():
        state.now = start
        # Сертифікат: строк підібрано так, щоб на початку інциденту він був ~30 днів
        cert_inc = [i for i in incidents if i["server"] == state.name and i["scenario"] == "cert_expiring"]
        if cert_inc and state.cert_days is not None:
            state.cert_days = 30 + (cert_inc[0]["start"] - start).total_seconds() / 86400
        # Бекапи за попередні ночі
        b = state.profile.get("backup")
        if b:
            last = local_day_start(start, tz) + timedelta(hours=b["hour"], minutes=b["duration_min"])
            if last > start:
                last -= timedelta(days=1)
            size = state.folder_gb[b["data_folder"]] * b["size_ratio"]
            state.backups = [(last - timedelta(days=k), size) for k in range(b["retention_days"] - 1, -1, -1)]
            state.last_backup_end = last
            state.sync_backup_folder()

    # --- Пропуски і неможливі значення (номери хвилин) ---
    skip = {name: {i for i in range(total_min + 1) if rng.random() < MISSING_PROB} for name in sim.states}
    for name, n in LONG_GAPS.items():
        for _ in range(n):
            g_start = rng.randint(1440, total_min - 1440)
            skip[name].update(range(g_start, g_start + rng.randint(5, 15)))
    corrupt = {}
    kinds = ["cpu_over_100", "ram_over_100", "negative_read", "negative_net"]
    for name in sim.states:
        placed = 0
        while placed < CORRUPT_PER_SERVER:
            i = rng.randint(1440, total_min - 1)
            if i not in skip[name] and (name, i) not in corrupt:
                corrupt[(name, i)] = rng.choice(kinds)
                placed += 1

    # --- Інциденти за хвилиною початку і кінця ---
    def minute_of(t):
        return int((t - start).total_seconds() // 60)
    starts, ends = defaultdict(list), defaultdict(list)
    for inc in incidents:
        starts[(inc["server"], minute_of(inc["start"]))].append(inc)
        ends[(inc["server"], minute_of(inc["end"]))].append(inc)

    print(f"Генерую {days} днів: {fmt(start)} — {fmt(end)} (UTC), seed={seed}")
    for i in range(total_min + 1):
        t = start + timedelta(minutes=i)
        for name in sim.states:
            sim.step(name, t, i,
                     starting=starts.get((name, i), ()), ending=ends.get((name, i), ()),
                     skip=i in skip[name], corrupt=corrupt.get((name, i)))
        if i % 1440 == 0 and i > 0:
            sim.flush()
            print(f"  день {i // 1440} з {days}", end="\r")
    sim.flush()

    # --- Мітки інцидентів («правильні відповіді» для оцінки ML) ---
    with db_session() as conn:
        for inc in incidents:
            conn.execute("INSERT INTO sim_incidents (server_id, scenario, started_at, ended_at) VALUES (?,?,?,?)",
                         (server_ids[inc["server"]], inc["scenario"], fmt(inc["start"]), fmt(inc["end"])))

    long_gaps = sum(LONG_GAPS.values())
    print_summary(incidents, sim.stats, long_gaps, tz)


def print_summary(incidents, stats, long_gaps, tz) -> None:
    print("\nГотово.")
    with db_session() as conn:
        print("Рядків у таблицях (лише sim-*):")
        for table in INSERT_SQL:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE server_id IN "
                             "(SELECT id FROM servers WHERE kind = 'sim')").fetchone()[0]
            print(f"  {table:<18} {n:>8}")
    print(f"Недосконалості: пропущено хвилин {stats['пропущено хвилин']} "
          f"(серед них {long_gaps} довгі пропуски), неможливих значень {stats['неможливих значень']}")

    print(f"\nІнциденти в sim_incidents: {len(incidents)} "
          f"(різних типів: {len({i['scenario'] for i in incidents})})")
    print(f"  {'сервер':<7}{'сценарій':<15}{'початок (місц.)':<18}{'тривалість':>12}")
    for inc in incidents:
        dur = inc["end"] - inc["start"]
        dur_txt = (f"{dur.days} д {dur.seconds // 3600} год" if dur.days
                   else f"{int(dur.total_seconds() // 60)} хв")
        local = (inc["start"] + tz).strftime("%d.%m %a %H:%M")
        overlap = any(o is not inc and o["server"] == inc["server"]
                      and o["start"] < inc["end"] and inc["start"] < o["end"] for o in incidents)
        print(f"  {inc['server']:<7}{inc['scenario']:<15}{local:<18}{dur_txt:>12}"
              f"{'   <- ланцюжок' if overlap else ''}")


# ---------------------------------------------------------------------------
# Контрольний графік
# ---------------------------------------------------------------------------

def plot_history() -> None:
    import matplotlib
    matplotlib.use("Agg")   # малюємо у файл, без вікна
    import matplotlib.pyplot as plt
    import pandas as pd

    tz = timedelta(hours=load_profiles()["timezone_offset_hours"])
    with db_session() as conn:
        # Неможливі значення («збій агента») на графік не беремо — їх відкине ETL
        metrics = pd.read_sql(
            "SELECT m.ts, s.name AS server, m.cpu_pct, m.ram_pct FROM metrics m "
            "JOIN servers s ON s.id = m.server_id "
            "WHERE s.kind = 'sim' AND m.cpu_pct BETWEEN 0 AND 100 AND m.ram_pct BETWEEN 0 AND 100", conn)
        disks = pd.read_sql(
            "SELECT d.ts, s.name AS server, d.drive, d.used_pct FROM disk_usage d "
            "JOIN servers s ON s.id = d.server_id WHERE s.kind = 'sim'", conn)
        incidents = pd.read_sql(
            "SELECT s.name AS server, i.scenario, i.started_at, i.ended_at FROM sim_incidents i "
            "JOIN servers s ON s.id = i.server_id WHERE s.kind = 'sim'", conn)
    if metrics.empty:
        print("Немає даних sim-*. Спершу: python -m simulator.simulator --history 30")
        return

    for df in (metrics, disks):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("started_at", "ended_at"):
        incidents[col] = pd.to_datetime(incidents[col], utc=True)

    def hourly(df, server, column, drive=None):
        part = df[df["server"] == server]
        if drive:
            part = part[part["drive"] == drive]
        return part.set_index("ts")[column].resample("1h").mean()

    panels = [
        ("sim-1: CPU, % (середнє за годину)", hourly(metrics, "sim-1", "cpu_pct"), "sim-1", True),
        ("sim-1: RAM, % (середнє за годину; витоки пам'яті)", hourly(metrics, "sim-1", "ram_pct"), "sim-1", True),
        ("sim-1: диск D:, % зайнято (логи IIS, щотижневе прибирання)",
         hourly(disks, "sim-1", "used_pct", "D:"), "sim-1", False),
        ("sim-2: CPU, % (середнє за годину; нічний бекап о 02:00)", hourly(metrics, "sim-2", "cpu_pct"), "sim-2", True),
        ("sim-2: RAM, % (середнє за годину; SQL Server тримає пам'ять)",
         hourly(metrics, "sim-2", "ram_pct"), "sim-2", True),
        ("sim-2: диск E:, % зайнято (бекапи і вивантаження)", hourly(disks, "sim-2", "used_pct", "E:"), "sim-2", False),
    ]
    colors = scenario_colors()

    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 19), sharex=True)
    start, end = metrics["ts"].min(), metrics["ts"].max()
    for ax, (title, series, server, show_weekends) in zip(axes, panels):
        ax.plot(series.index, series.values, color="#333333", linewidth=0.9)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        if show_weekends:   # вихідні за місцевим часом — сірим фоном
            day = (start + tz).normalize() - tz
            while day < end:
                if (day + tz).weekday() >= 5:
                    ax.axvspan(day, day + pd.Timedelta(days=1), color="#000000", alpha=0.06, linewidth=0)
                day += pd.Timedelta(days=1)
        for _, inc in incidents[incidents["server"] == server].iterrows():
            ax.axvspan(inc["started_at"], inc["ended_at"], color=colors[inc["scenario"]],
                       alpha=0.35, label=inc["scenario"], linewidth=0)
    handles = {}
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=4, fontsize=9,
               title="Інциденти (sim_incidents); сірий фон — вихідні")
    axes[-1].set_xlabel("час (UTC)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=110)
    plt.close(fig)
    print(f"Графік збережено: {PLOT_PATH}")


def scenario_colors() -> dict:
    """Сталий колір для кожного типу інциденту (однаковий на всіх графіках)."""
    import matplotlib.pyplot as plt
    return dict(zip(sorted(SCENARIOS), plt.get_cmap("tab10").colors))


# ---------------------------------------------------------------------------
# Перегляд окремих інцидентів
# ---------------------------------------------------------------------------

def list_incidents() -> None:
    """Друкує всі інциденти з sim_incidents."""
    tz = timedelta(hours=load_profiles()["timezone_offset_hours"])
    with db_session() as conn:
        rows = conn.execute(
            "SELECT i.id, s.name AS server, i.scenario, i.started_at, i.ended_at "
            "FROM sim_incidents i JOIN servers s ON s.id = i.server_id ORDER BY i.started_at").fetchall()
    if not rows:
        print("У sim_incidents немає записів.")
        return

    def local(ts):
        if ts is None:
            return "триває"
        t = datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc) + tz
        return t.strftime("%d.%m %a %H:%M")

    print(f"Інцидентів: {len(rows)}. Час місцевий (UTC+{tz.seconds // 3600}); у базі — UTC.")
    print(f"  {'id':>4}  {'сервер':<7}{'тип':<15}{'початок':<17}{'кінець':<17}{'тривалість':>11}")
    for row in rows:
        if row["ended_at"]:
            s = datetime.strptime(row["started_at"], TS_FORMAT)
            e = datetime.strptime(row["ended_at"], TS_FORMAT)
            minutes = int((e - s).total_seconds() // 60)
            duration = f"{minutes // 1440} д {minutes % 1440 // 60} год" if minutes >= 1440 else f"{minutes} хв"
        else:
            duration = "—"
        print(f"  {row['id']:>4}  {row['server']:<7}{row['scenario']:<15}"
              f"{local(row['started_at']):<17}{local(row['ended_at']):<17}{duration:>11}")
    print("\nГрафік інциденту: python -m simulator.simulator --plot-incident <id>")


def plot_incident(incident_id: int) -> None:
    """Графік одного інциденту по хвилинах (±2 год): CPU, RAM, диски, сайт / перевірки, події."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    tz = timedelta(hours=load_profiles()["timezone_offset_hours"])
    with db_session() as conn:
        inc = conn.execute(
            "SELECT i.*, s.name AS server FROM sim_incidents i JOIN servers s ON s.id = i.server_id "
            "WHERE i.id = ?", (incident_id,)).fetchone()
        if inc is None:
            print(f"Інциденту з id = {incident_id} немає. Список: python -m simulator.simulator --incidents")
            return
        start = pd.Timestamp(inc["started_at"])
        end = pd.Timestamp(inc["ended_at"]) if inc["ended_at"] else pd.Timestamp.now(tz="UTC").floor("min")
        win_from, win_to = start - pd.Timedelta(hours=2), end + pd.Timedelta(hours=2)
        params = (inc["server_id"], win_from.strftime(TS_FORMAT), win_to.strftime(TS_FORMAT))

        metrics = pd.read_sql(
            "SELECT ts, cpu_pct, ram_pct, app_response_ms, app_error_pct FROM metrics "
            "WHERE server_id = ? AND ts BETWEEN ? AND ? "
            "AND cpu_pct BETWEEN 0 AND 100 AND ram_pct BETWEEN 0 AND 100", conn, params=params)
        disks = pd.read_sql("SELECT ts, drive, used_pct FROM disk_usage WHERE server_id = ? AND ts BETWEEN ? AND ?",
                            conn, params=params)
        checks = pd.read_sql("SELECT ts, check_type, ok, value FROM checks WHERE server_id = ? AND ts BETWEEN ? AND ?",
                             conn, params=params)
        events = pd.read_sql("SELECT ts, event_id, level, source FROM events WHERE server_id = ? AND ts BETWEEN ? AND ?",
                             conn, params=params)
        # Інші інциденти на цьому сервері, що потрапляють у вікно (ланцюжки)
        others = pd.read_sql(
            "SELECT id, scenario, started_at, ended_at FROM sim_incidents WHERE server_id = ? AND id != ? "
            "AND started_at <= ? AND COALESCE(ended_at, ?) >= ?", conn,
            params=(inc["server_id"], incident_id, params[2], params[2], params[1]))

    def to_local(series):
        """UTC-рядки -> місцевий час без часового поясу (для підписів осі)."""
        return pd.to_datetime(series, utc=True).dt.tz_convert(None) + tz

    def per_minute(df, column):
        """Ряд з кроком 1 хв: пропущені хвилини стають розривом лінії."""
        s = df.assign(t=to_local(df["ts"]).dt.floor("min")).groupby("t")[column].mean()
        return s.resample("1min").mean()

    loc_start, loc_end = start.tz_convert(None) + tz, end.tz_convert(None) + tz

    # --- Які панелі малювати ---
    panels = [("CPU, %", [("CPU", per_minute(metrics, "cpu_pct"))]),
              ("RAM, %", [("RAM", per_minute(metrics, "ram_pct"))]),
              ("Диски, % зайнято", [(d, per_minute(disks[disks["drive"] == d], "used_pct"))
                                     for d in sorted(disks["drive"].unique())])]
    http = checks[checks["check_type"] == "http"]
    if not http.empty:
        # Відповідь сайту — з HTTP-перевірки; невдалі перевірки — червоні точки
        http_fail = http[http["ok"] == 0]
        panels.append(("Сайт: час відповіді HTTP-перевірки, мс",
                       [("відповідь", per_minute(http, "value"))], http_fail))
    if metrics["app_error_pct"].notna().any():
        panels.append(("Сайт: помилки, % запитів", [("помилки", per_minute(metrics, "app_error_pct"))]))
    for check_type, title in (("backup", "Бекап: годин від останнього успішного"),
                              ("cert", "Сертифікат: днів до кінця строку")):
        part = checks[checks["check_type"] == check_type]
        if not part.empty and (check_type == "backup" or inc["scenario"] == "cert_expiring"):
            s = part.assign(t=to_local(part["ts"])).set_index("t")["value"]
            panels.append((title, [(check_type, s)]))

    colors = scenario_colors()
    level_color = {"Error": "#c0392b", "Warning": "#e67e22", "Information": "#7f8c8d"}

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 2.6 * len(panels) + 1), sharex=True)
    for ax, panel in zip(axes, panels):
        title, lines = panel[0], panel[1]
        for label, series in lines:
            ax.plot(series.index, series.values, linewidth=1, label=label,
                    drawstyle="steps-post" if label in ("backup", "cert") else "default")
        if len(panel) > 2 and not panel[2].empty:   # невдалі HTTP-перевірки — червоні точки
            fails = panel[2]
            fail_t = to_local(fails["ts"]).dt.floor("min")
            y = lines[0][1].reindex(fail_t).values
            ax.scatter(fail_t, y, color="#c0392b", s=10, zorder=3, label="HTTP-перевірка не пройшла")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        if len(lines) > 1 or len(panel) > 2:
            ax.legend(loc="upper left", fontsize=8)
        # Сам інцидент і пов'язані з ним
        ax.axvspan(loc_start, loc_end, color=colors[inc["scenario"]], alpha=0.25, linewidth=0)
        for _, o in others.iterrows():
            o_end = pd.Timestamp(o["ended_at"]) if o["ended_at"] else end
            ax.axvspan(pd.Timestamp(o["started_at"]).tz_convert(None) + tz, o_end.tz_convert(None) + tz,
                       color=colors[o["scenario"]], alpha=0.18, linewidth=0, hatch="//")
        # Події журналу — вертикальні позначки
        for _, ev in events.iterrows():
            ev_t = pd.Timestamp(ev["ts"]).tz_convert(None) + tz
            ax.axvline(ev_t, color=level_color.get(ev["level"], "#7f8c8d"), linewidth=0.8, alpha=0.8)

    # Вісь — рівно вікно ±2 год (смуги пов'язаних інцидентів можуть бути довшими)
    axes[0].set_xlim(win_from.tz_convert(None) + tz, win_to.tz_convert(None) + tz)

    # Підписи подій над першою панеллю
    top = axes[0]
    for _, ev in events.iterrows():
        ev_t = pd.Timestamp(ev["ts"]).tz_convert(None) + tz
        top.annotate(str(ev["event_id"]), (ev_t, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 2), textcoords="offset points", rotation=90, fontsize=7,
                     ha="center", va="bottom", color=level_color.get(ev["level"], "#7f8c8d"))

    chain = ", ".join(f"#{o['id']} {o['scenario']}" for _, o in others.iterrows())
    fig.suptitle(f"Інцидент #{incident_id}: {inc['scenario']} на {inc['server']}, "
                 f"{loc_start:%d.%m %H:%M} – {loc_end:%d.%m %H:%M} (місцевий час)"
                 + (f"\nУ вікні також (штриховка): {chain}" if chain else "")
                 + "\nВертикальні лінії — події журналу: червоні — Error, помаранчеві — Warning, сірі — Information",
                 fontsize=10, x=0.01, ha="left")
    axes[-1].set_xlabel(f"місцевий час (UTC+{tz.seconds // 3600})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = PLOT_PATH.parent / f"incident_{incident_id}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Графік збережено: {path}")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Симулятор серверів sim-1 і sim-2")
    parser.add_argument("--history", type=int, metavar="ДНІВ", help="згенерувати історію за N днів")
    parser.add_argument("--reset", action="store_true", help="спершу видалити всі дані sim-*")
    parser.add_argument("--seed", type=int, default=42, help="зерно випадковості (за замовчуванням 42)")
    parser.add_argument("--plot", action="store_true", help="контрольний графік у docs/experiments/")
    parser.add_argument("--incidents", action="store_true", help="список інцидентів з sim_incidents")
    parser.add_argument("--plot-incident", type=int, metavar="ID",
                        help="графік одного інциденту (±2 год) у docs/experiments/incident_<id>.png")
    parser.add_argument("--live", action="store_true", help="живий потік (крок 1.6)")
    parser.add_argument("--scenario", help="запустити сценарій (крок 1.7)")
    args = parser.parse_args()

    if args.live or args.scenario:
        print("Живий режим і ручні сценарії з'являться на кроках 1.6–1.7.")
        return 1

    init_db()
    if args.reset:
        print("Видаляю дані sim-*...")
        reset_sim_data()

    if args.history:
        existing = sim_data_count()
        if existing:
            print(f"У базі вже є {existing} записів метрик sim-*. Повторна генерація задвоїла б дані.\n"
                  "Щоб згенерувати заново: python -m simulator.simulator --history 30 --reset")
            return 1
        generate_history(args.history, args.seed)

    if args.plot:
        plot_history()
    if args.incidents:
        list_incidents()
    if args.plot_incident is not None:
        plot_incident(args.plot_incident)

    if not (args.history or args.plot or args.reset or args.incidents or args.plot_incident is not None):
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
