"""Короткі незалежні прогони одного диска для сценарного стенду прогнозу (крок 2.2b).

Кожен прогін — ~10–13 днів норми, потім одна ситуація, потім кілька днів після.
Використовує ті самі класи, що й симулятор (ServerState, розклад бекапу і
прибирання, сценарій disk_fill з причинами), але лише для одного диска і в пам'яті —
основна база не змінюється. Параметри кожного прогону випадкові в розумних межах.

Ситуації:
  S1 стабільний диск; S2 «пилка» (ріст логів + щотижневе прибирання);
  S3 щоденні стрибки (бекап і видалення старих); S4 зламане прибирання
  (retention_stopped); S5 log_burst; S6 швидке заповнення (fast);
  S7 регулярний стрибок поміщається впритул / не поміщається;
  S8 короткий різкий пік (тимчасовий файл на 10–30 хв).

Прогін визначається трійкою (seed, ситуація, номер) — його можна відтворити.
"""
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from simulator.scenarios import SCENARIOS
from simulator.simulator import ServerState, advance_state, backup_routine

SITUATIONS = {
    "S1": "стабільний диск",
    "S2": "«пилка»: ріст + щотижневе прибирання",
    "S3": "щоденні стрибки: бекап і видалення старих",
    "S4": "зламане прибирання (retention_stopped)",
    "S5": "log_burst: логи в 5–10 разів швидше",
    "S6": "швидке заповнення (fast)",
    "S7": "регулярний стрибок: впритул / не поміщається",
    "S8": "короткий різкий пік (тимчасовий файл)",
}

TZ = timedelta(0)                                        # у стенді місцевий час = UTC
START = datetime(2026, 1, 5, tzinfo=timezone.utc)        # понеділок (дата умовна)
MIN = timedelta(minutes=1)


def run_rng(seed: int, situation: str, k: int) -> random.Random:
    """Генератор випадкових чисел саме цього прогону (відтворюваний)."""
    return random.Random(f"{seed}-{situation}-{k}")


# ---------------------------------------------------------------------------
# Шаблони диска (профіль у форматі profiles.yaml, лише потрібні поля)
# ---------------------------------------------------------------------------

def empty_profile() -> dict:
    return {"services": [], "processes": [], "folders": [], "drives": {}, "checks": {}}


def template_stable(rng, level: float) -> tuple[dict, str, dict]:
    """Стабільний диск: майже не росте (0–0,2 % на добу)."""
    total = rng.choice([80, 120, 250, 500])
    p = empty_profile()
    p["drives"]["C:"] = {"total_gb": total, "other_gb": level / 100 * total - 1,
                         "growth_gb_per_day": rng.uniform(0, 0.002) * total}
    p["folders"] = [{"path": "C:\\Temp", "drive": "C:", "start_gb": 1.0},
                    {"path": "C:\\Tmp", "drive": "C:", "start_gb": 0.0}]
    return p, "C:", {"total_gb": total}


def template_logs(rng, level: float, rate_pct: float, cleanup_weekday: int | None = None
                  ) -> tuple[dict, str, dict]:
    """Логи: ростуть rate_pct % диска на добу в будні (у вихідні менше),
    щотижневе прибирання в певний день і годину."""
    total = rng.choice([100, 200, 400, 800])
    wd = rate_pct / 100 * total
    folder = "D:\\Logs"
    start_logs = rng.uniform(0, 3) * wd
    p = empty_profile()
    p["drives"]["D:"] = {"total_gb": total, "other_gb": max(level / 100 * total - start_logs, 1),
                         "growth_gb_per_day": 0.0}
    p["folders"] = [
        {"path": folder, "drive": "D:", "start_gb": start_logs,
         "weekday_gb_per_day": wd, "weekend_gb_per_day": wd * rng.uniform(0.3, 0.7),
         "weekly_cleanup": {"weekday": rng.randint(0, 6) if cleanup_weekday is None else cleanup_weekday,
                            "hour": rng.randint(0, 23), "keep_gb": 0.01 * total}},
        {"path": "D:\\Tmp", "drive": "D:", "start_gb": 0.0},
    ]
    return p, "D:", {"total_gb": total, "logs_pct_day": rate_pct, "folder": folder}


def template_backup(rng, base_level: float | None = None) -> tuple[dict, str, dict]:
    """Бекапи на E:: щоночі (у випадкову годину) бекап size_pct % диска,
    окреме завдання видаляє старіші за retention_days."""
    total = rng.choice([200, 500, 1000])
    size_pct = rng.uniform(5, 12)
    retention = rng.randint(4, 7)
    lo = retention * size_pct + 2
    hi = max(lo + 1, 85 - size_pct)
    base = base_level if base_level is not None else rng.uniform(lo, hi)
    base = min(max(base, lo), hi)
    data_gb = size_pct / 100 * total / 0.15
    hour = rng.randint(0, 23)
    p = empty_profile()
    p["drives"]["E:"] = {"total_gb": total, "other_gb": base / 100 * total - retention * size_pct / 100 * total,
                         "growth_gb_per_day": 0.0}
    p["drives"]["D:"] = {"total_gb": data_gb * 2, "other_gb": 1, "growth_gb_per_day": 0.0}
    p["folders"] = [{"path": "E:\\Backup", "drive": "E:", "start_gb": 0.0},
                    {"path": "E:\\Exports", "drive": "E:", "start_gb": 0.0},
                    {"path": "E:\\Tmp", "drive": "E:", "start_gb": 0.0},
                    {"path": "D:\\Data", "drive": "D:", "start_gb": data_gb}]
    p["backup"] = {"database": "Db", "folder": "E:\\Backup", "hour": hour,
                   "duration_min": rng.randint(20, 60), "size_ratio": 0.15, "data_folder": "D:\\Data",
                   "retention_days": retention, "cleanup_hour": (hour + rng.randint(2, 8)) % 24}
    return p, "E:", {"total_gb": total, "backup_pct": size_pct, "backup_hour": hour,
                     "retention_days": retention}


def init_backups(state: ServerState) -> None:
    """Бекапи за попередні ночі — як у generate_history симулятора."""
    b = state.profile["backup"]
    day0 = START.replace(hour=0, minute=0)
    last = day0 + timedelta(hours=b["hour"], minutes=b["duration_min"])
    if last > START:
        last -= timedelta(days=1)
    size = state.folder_gb[b["data_folder"]] * b["size_ratio"]
    state.backups = [(last - timedelta(days=k), size) for k in range(b["retention_days"] - 1, -1, -1)]
    state.last_backup_end = last
    state.sync_backup_folder()


# ---------------------------------------------------------------------------
# Прогін
# ---------------------------------------------------------------------------

def generate(seed: int, situation: str, k: int) -> dict:
    """Один прогін. Повертає ряди «зайнято, %» і «вільно, ГБ» (похвилинно),
    час подій ситуації і дій людини / щотижневого прибирання (для «MAE до втручання»)."""
    rng = run_rng(seed, situation, k)
    normal_days = rng.randint(10, 13)
    inc_start = START + timedelta(days=normal_days, minutes=rng.randint(0, 1439))
    params, info = {}, {}
    incident = None          # (cause, params) для disk_fill
    max_inc_days, end_level = None, None
    fixed_end = None
    export, spike = None, None

    if situation == "S1":
        prof, drive, info = template_stable(rng, rng.uniform(30, 85))
    elif situation == "S2":
        l0 = rng.uniform(20, 60)
        rate = min(rng.uniform(1, 5), (88 - l0) / 7.7)
        prof, drive, info = template_logs(rng, l0, rate)
    elif situation == "S3":
        prof, drive, info = template_backup(rng)
    elif situation == "S4":
        if rng.random() < 0.5:
            # у звичайний тиждень (з прибиранням) пік ≤ 85 %: рівень після прибирання + 7,7 дня росту
            rate = rng.uniform(3, 8)
            l0 = rng.uniform(15, max(16, 85 - 7.7 * rate))
            prof, drive, info = template_logs(rng, l0, rate)
        else:
            prof, drive, info = template_backup(rng)
        incident = ("retention_stopped", {})
        max_inc_days, end_level = 12, 92
    elif situation == "S5":
        factor = rng.uniform(5, 10)
        target_pct_day = rng.uniform(6, 25)              # швидкість під час burst, % на добу
        # прибирання — за добу до початку, щоб у burst не потрапило;
        # у звичайний тиждень пік ≤ 85 %
        rate = target_pct_day / factor
        l0 = rng.uniform(15, max(16, min(55, 85 - 7.7 * rate)))
        prof, drive, info = template_logs(rng, l0, rate,
                                          cleanup_weekday=(inc_start.weekday() - 1) % 7)
        incident = ("log_burst", {"burst_factor": round(factor, 1)})
        max_inc_days, end_level = 5, 92
    elif situation == "S6":
        if rng.random() < 0.5:
            prof, drive, info = template_stable(rng, rng.uniform(30, 80))
        else:
            l0 = rng.uniform(20, 55)
            prof, drive, info = template_logs(rng, l0, min(rng.uniform(1, 4), (85 - l0) / 7.7))
        fill = rng.randint(60, 360)
        incident = ("fast", {"fill_minutes": fill})
        fixed_end = inc_start + timedelta(minutes=fill + rng.randint(30, 360))
    elif situation == "S7":
        prof, drive, info = template_backup(rng)
        b = prof["backup"]
        day = START + timedelta(days=normal_days)
        jump_at = day.replace(hour=b["hour"], minute=0)
        lead_min = rng.randint(120, 480)                  # вивантаження починається за 2–8 год до бекапу
        fill_min = rng.randint(30, lead_min - 10)
        fits_target = rng.random() < 0.4
        k_free = rng.uniform(1.03, 1.25) if fits_target else rng.uniform(0.6, 0.95)
        export = {"start": jump_at - timedelta(minutes=lead_min), "fill_min": fill_min, "k_free": k_free,
                  "jump_at": jump_at, "clean_at": jump_at + timedelta(minutes=rng.randint(6 * 60, 30 * 60))}
        lead_h = lead_min / 60
        inc_start = export["start"]
        info.update({"k_free": round(k_free, 2), "lead_h": round(lead_h, 1)})
    elif situation == "S8":
        if rng.random() < 0.5:
            level = rng.uniform(30, 70)
            prof, drive, info = template_stable(rng, level)
        else:
            level = rng.uniform(20, 50)
            prof, drive, info = template_logs(rng, level, min(rng.uniform(1, 4), (80 - level) / 7.7))
        spike = {"start": inc_start, "minutes": rng.randint(10, 30), "pct": None}
    else:
        raise ValueError(situation)

    folder_of = {"C:": "C:\\Temp", "D:": "D:\\Logs", "E:": "E:\\Backup"}
    prof["scenario_targets"] = {"disk_fill": {"folder": folder_of[drive] if drive != "C:" else "C:\\Temp",
                                              "retention_folder": folder_of[drive],
                                              "log_folder": folder_of[drive]}}
    state = ServerState("bench", prof, TZ, rng)
    if prof.get("backup"):
        init_backups(state)
    sc = SCENARIOS["disk_fill"]
    total = state.drive_total_gb(drive)
    noise = rng.uniform(0, 0.15)
    after_days = rng.randint(2, 4)
    # Кінець: для фіксованих — заздалегідь; для S4/S5 — коли «адміністратор помітить»
    end_limit = START + timedelta(days=normal_days + (max_inc_days or 2) + after_days + 1)

    used_l, free_l, idx = [], [], []
    actions = []                         # (час, вид) — дії людини і щотижневе прибирання
    events = {"start": inc_start}
    inc_active, inc_end_at = False, fixed_end
    t = START
    weekly = [f for f in prof["folders"] if f.get("weekly_cleanup")]
    while t < end_limit:
        state.now = t
        if incident and not inc_active and t == inc_start:
            cause, extra = incident
            params = dict(extra, cause=cause, duration_min=60 * 24 * (max_inc_days or 1))
            if cause == "fast":
                params["duration_min"] = int((fixed_end - inc_start).total_seconds() // 60)
            sc.start(state, params, rng)
            inc_active = True
            inc_minute = 0
        if inc_active and inc_end_at is not None and t >= inc_end_at:
            sc.end(state, params, rng)
            inc_active = False
            actions.append((t, "адміністратор почистив"))
            events["end"] = t
            incident = None
            end_limit = min(end_limit, t + timedelta(days=after_days))
        if state.profile.get("backup"):
            backup_routine(state, t)
            if export and t == export["jump_at"]:
                events["jump_fits"] = state.backup_running_until is not None
        values = {"disk_write_mbs": 0.0, "process_overrides": []}
        if inc_active:
            sc.apply(state, values, inc_minute, params, rng)
            inc_minute += 1
            if max_inc_days and inc_end_at is None and t >= inc_start + timedelta(days=max_inc_days):
                inc_end_at = t + timedelta(hours=rng.uniform(2, 12))
        # S7: одноразове вивантаження, яке заповнює місце перед бекапом
        if export:
            if t == export["start"]:
                free_now = total - state.drive_used_gb(drive)
                size = state.folder_gb[prof["backup"]["data_folder"]] * prof["backup"]["size_ratio"]
                export["add_gb"] = max(free_now - export["k_free"] * size, 0.5)
                export["rate"] = export["add_gb"] / export["fill_min"]
            if export["start"] <= t < export["start"] + timedelta(minutes=export["fill_min"]):
                state.folder_gb["E:\\Exports"] += export["rate"]
            if t == export["clean_at"]:
                state.folder_gb["E:\\Exports"] = 0.0
                actions.append((t, "адміністратор почистив"))
                events["end"] = t
                end_limit = min(end_limit, t + timedelta(days=after_days))
        # S8: тимчасовий файл
        if spike:
            tmp = f"{drive}\\Tmp"
            if t == spike["start"]:
                room = 88 - state.drive_used_pct(drive)
                spike["pct"] = min(rng.uniform(5, 25), max(room, 2))
                state.folder_gb[tmp] += spike["pct"] / 100 * total
            if t == spike["start"] + timedelta(minutes=spike["minutes"]):
                state.folder_gb[tmp] = 0.0
                events["end"] = t
                end_limit = min(end_limit, t + timedelta(days=after_days))
        before = [state.folder_gb[f["path"]] for f in weekly]
        advance_state(state, t)
        if any(state.folder_gb[f["path"]] < b - 1e-9 for f, b in zip(weekly, before)):
            actions.append((t, "щотижневе прибирання"))

        used_gb = state.drive_used_gb(drive)
        pct = 100 * used_gb / total + rng.gauss(0, noise)
        if inc_active and end_level and inc_end_at is None and pct >= end_level:
            inc_end_at = t + timedelta(hours=rng.uniform(2, 12))      # адміністратор помітив
        idx.append(t)
        used_l.append(round(min(max(pct, 0), 100), 1))
        free_l.append(round(total - used_gb, 1))
        t += MIN

    index = pd.DatetimeIndex(idx)
    used = pd.Series(used_l, index=index)
    free = pd.Series(free_l, index=index)
    if export:
        events["jump_at"] = export["jump_at"]
    if situation in ("S1", "S2", "S3"):
        events.pop("start", None)          # ситуації немає — лише звичайна поведінка
    if spike:
        info["spike_pct"] = round(spike["pct"], 1)
        info["spike_min"] = spike["minutes"]
    info.update({"normal_days": normal_days, "noise_pp": round(noise, 3)})
    info.update({k: v for k, v in params.items() if k in ("burst_factor", "fill_minutes")})
    return {"id": f"{situation}-{k:02d}", "seed": seed, "situation": situation, "drive": drive,
            "used": pd.Series(used.to_numpy(), index=index.tz_convert("UTC")),
            "free": pd.Series(free.to_numpy(), index=index.tz_convert("UTC")),
            "events": {k2: pd.Timestamp(v) for k2, v in events.items() if isinstance(v, datetime)},
            "jump_fits": events.get("jump_fits"),
            "actions": [(pd.Timestamp(a), kind) for a, kind in actions],
            "eval_from": pd.Timestamp(START + timedelta(days=7)),
            "info": info}
