"""Прогноз заповнення диска.

Прогноз універсальний: він бачить лише ряд «зайнятість диска, %»
(disk_used_pct:<диск>) і «вільно, ГБ» (disk_free_gb:<диск>) з metrics_clean.
Нічого не знає про сервер, теки чи бекапи — усе знаходить у самих даних.
Прогноз лише рахує числа; коли це алерт — вирішують правила (крок 2.3).

Три питання:
  1. trend       — тренд: пряма від останнього різкого падіння (прибирання).
                   Швидкість %/добу, рівень на +24 год і +7 днів, годин до порогу.
  2. fast_growth — незвичний ріст: пряма за останні 90 хв. Незвичний — лише якщо
                   помітно швидший, ніж зазвичай у ці ж хвилини доби за 7 днів.
  3. jump        — регулярний стрибок (бекап, нічний експорт...): чи поміститься
                   наступний.

Метод — налаштування (core/forecast.yaml): linear — робочий;
naive_last і naive_daily — лише для порівняння на кроці 2.2b.

Запуск:
  python -m core.forecast                               # прогноз «зараз» -> predictions
  python -m core.forecast --server sim-2                # лише один сервер
  python -m core.forecast --at "2026-11-12 03:00" --server sim-2
                     # прогноз на минулий момент (UTC): бачить лише дані до нього,
                     # у predictions не пише — лише друкує
"""
import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.db import DB_PATH, db_session, init_db

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
CONFIG_PATH = Path(__file__).with_name("forecast.yaml")
Z_10_90 = 1.2816      # інтервал 10–90 %: ± 1,28 стандартного відхилення


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Допоміжне: пряма і час до порогу
# ---------------------------------------------------------------------------

class Line:
    """Пряма y = a + b·t (метод найменших квадратів) з інтервалом прогнозу."""

    def __init__(self, t: np.ndarray, y: np.ndarray):
        self.n = len(t)
        self.b, self.a = np.polyfit(t, y, 1)
        resid = y - (self.a + self.b * t)
        self.sigma = float(np.sqrt((resid ** 2).sum() / (self.n - 2))) if self.n > 2 else 0.0
        self.t_mean = float(t.mean())
        self.sxx = float(((t - self.t_mean) ** 2).sum()) or 1.0

    def at(self, t: float) -> tuple[float, float, float]:
        """Значення на момент t і межі інтервалу 10–90 %."""
        v = self.a + self.b * t
        se = self.sigma * np.sqrt(1 + 1 / self.n + (t - self.t_mean) ** 2 / self.sxx)
        return float(v), float(v - Z_10_90 * se), float(v + Z_10_90 * se)


def clip_pct(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Диск не буває заповнений менше ніж на 0 % чи більше ніж на 100 %."""
    return tuple(float(min(max(x, 0.0), 100.0)) for x in v)


def hours_to(level: float, speed_per_h: float, threshold: float, max_hours: float) -> float | None:
    """Скільки годин до порогу при такій швидкості; None — не досягне."""
    if level >= threshold:
        return 0.0
    if speed_per_h <= 1e-9:
        return None
    h = (threshold - level) / speed_per_h
    return float(h) if h <= max_hours else None


def hours_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a).total_seconds() / 3600


def tod_min(t: pd.Timestamp) -> int:
    """Хвилина доби (UTC), 0..1439."""
    return t.hour * 60 + t.minute


def tod_diff(a: int, b: int) -> int:
    """Різниця між двома часами доби з урахуванням переходу через північ."""
    d = abs(a - b) % 1440
    return min(d, 1440 - d)


# ---------------------------------------------------------------------------
# Різкі зміни: підйоми (стрибки) і падіння (прибирання)
# ---------------------------------------------------------------------------

def find_steps(used: pd.Series, free: pd.Series, cfg: dict) -> list[dict]:
    """Різкі підйоми і падіння. Кожна подія: start, end, size_pp (+ підйом, − падіння),
    size_gb (наскільки змінилось вільне місце, завжди > 0)."""
    c = cfg["steps"]
    x = used.resample(f"{c['bin_min']}min").last().dropna()
    if len(x) < 2:
        return []
    d = x.diff().to_numpy()
    sign = np.where(d >= c["step_rate_pp"], 1, np.where(d <= -c["step_rate_pp"], -1, 0))
    steps, i = [], 1
    while i < len(x):
        if sign[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and sign[j + 1] == sign[i]:
            j += 1
        start, end = x.index[i - 1], x.index[j]
        size_pp = float(x.iloc[j] - x.iloc[i - 1])
        minutes = hours_between(start, end) * 60
        if abs(size_pp) >= c["step_min_pp"] and minutes <= c["step_max_min"] + c["bin_min"]:
            f0 = free.asof(start + pd.Timedelta(minutes=c["bin_min"] - 1))
            f1 = free.asof(end + pd.Timedelta(minutes=c["bin_min"] - 1))
            size_gb = abs(float(f0 - f1)) if pd.notna(f0) and pd.notna(f1) else float("nan")
            steps.append({"start": start, "end": end, "size_pp": size_pp, "size_gb": size_gb})
        i = j + 1
    return steps


def similar_size(a: float, b: float, tol: float) -> bool:
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol * max(a, b)


def removal_after(rise: dict, steps: list[dict], cfg: dict) -> dict | None:
    """Падіння того самого розміру після підйому (не пізніше max_pair_hours) — «пара»."""
    c = cfg["jump"]
    for s in steps:
        if (s["size_pp"] < 0 and s["start"] >= rise["end"]
                and hours_between(rise["end"], s["start"]) <= c["max_pair_hours"]
                and similar_size(s["size_gb"], rise["size_gb"], c["size_tolerance"])):
            return s
    return None


def find_regular_rise(steps: list[dict], now: pd.Timestamp, cfg: dict) -> dict | None:
    """Регулярний стрибок: підйоми приблизно в той самий час доби і того самого
    розміру щонайменше в min_days різні дні за lookback_days. None — немає."""
    c = cfg["jump"]
    rises = [s for s in steps if s["size_pp"] > 0
             and s["start"] >= now - pd.Timedelta(days=c["lookback_days"])]
    best = None
    for r in sorted(rises, key=lambda s: s["start"], reverse=True):
        same = [q for q in rises
                if tod_diff(tod_min(q["start"]), tod_min(r["start"])) <= c["time_tolerance_min"]
                and similar_size(q["size_gb"], r["size_gb"], c["size_tolerance"])]
        days = {q["start"].date() for q in same}
        if len(days) >= c["min_days"] and (best is None or len(days) > best["days"]):
            best = {"days": len(days), "matches": sorted(same, key=lambda s: s["start"])}
    if best is None:
        return None

    m = best["matches"]
    base = tod_min(m[-1]["start"])
    # середній час доби (зсуви відносно останнього — щоб не ламалось біля півночі)
    offsets = [((tod_min(q["start"]) - base + 720) % 1440) - 720 for q in m]
    tod = int(round(base + float(np.median(offsets)))) % 1440
    size_gb = float(np.median([q["size_gb"] for q in m[-3:]]))
    size_pp = float(np.median([q["size_pp"] for q in m[-3:]]))
    # чи зазвичай стрибок прибирають (видалення старого бекапу) і через скільки
    gaps = [hours_between(q["start"], p["start"]) for q in m if (p := removal_after(q, steps, cfg))]
    usually_removed = len(gaps) >= max(len(m) / 2, 1)
    return {"tod_min": tod, "size_gb": size_gb, "size_pp": size_pp, "days": best["days"],
            "usually_removed": usually_removed,
            "removal_after_h": float(np.median(gaps)) if usually_removed else None,
            "matches": m}


def matches_regular(rise: dict, regular: dict | None, cfg: dict) -> bool:
    c = cfg["jump"]
    return (regular is not None
            and tod_diff(tod_min(rise["start"]), regular["tod_min"]) <= c["time_tolerance_min"]
            and similar_size(rise["size_gb"], regular["size_gb"], c["size_tolerance"]))


# ---------------------------------------------------------------------------
# Три питання (лінійний метод)
# ---------------------------------------------------------------------------

def trend(used: pd.Series, free: pd.Series, now: pd.Timestamp, steps: list[dict],
          regular: dict | None, cfg: dict) -> dict:
    """Питання 1: тренд від останнього різкого падіння."""
    c, thr = cfg["trend"], cfg["threshold_pct"]
    drops = [s for s in steps if s["size_pp"] < 0]
    start = drops[-1]["end"] if drops else used.index[0]

    # Тимчасовий регулярний стрибок, який зазвичай прибирають (бекап, що буде
    # видалено), у пряму не беремо — поки не минув звичний час видалення.
    # Минув, а видалення не було — стрибок лишається в даних і тренд його бачить.
    excluded = None
    open_rises = [s for s in steps if s["size_pp"] > 0 and s["start"] >= start
                  and removal_after(s, steps, cfg) is None]
    if open_rises and regular and regular["usually_removed"]:
        r = open_rises[-1]
        deadline = r["start"] + pd.Timedelta(hours=regular["removal_after_h"]) \
            + pd.Timedelta(minutes=cfg["jump"]["time_tolerance_min"])
        if matches_regular(r, regular, cfg) and now < deadline:
            excluded = {"from": r["start"], "size_pp": r["size_pp"], "size_gb": r["size_gb"],
                        "expected_removal_by": deadline}

    data = used[used.index > start]
    if excluded:
        data = data[data.index < excluded["from"]]
    hourly = data.resample("1h").mean().dropna()
    result = {"start": start, "excluded": excluded, "points": len(hourly)}
    if len(hourly) < c["min_hours"]:
        result["status"] = "мало даних"
        return result

    t = np.array([hours_between(now, ts + pd.Timedelta(minutes=30)) for ts in hourly.index])
    line = Line(t, hourly.to_numpy())
    level_now = line.at(0)[0]
    speed_h = line.b
    total_gb = total_size_gb(used, free)
    result.update({
        "status": "ok",
        "level_now": level_now,
        "speed_pct_day": speed_h * 24,
        "speed_gb_day": speed_h * 24 * total_gb / 100 if total_gb else None,
        "pred": {24: clip_pct(line.at(24)), 168: clip_pct(line.at(168))},
        "hours_to_threshold": hours_to(level_now, speed_h, thr, c["max_hours_to_threshold"]),
    })
    return result


def fast_growth(used: pd.Series, now: pd.Timestamp, cfg: dict) -> dict:
    """Питання 2: швидкість за останні window_min хвилин проти звичайної в ці ж хвилини доби."""
    c, thr = cfg["fast_growth"], cfg["threshold_pct"]
    w = pd.Timedelta(minutes=c["window_min"])

    def slope(t_end):
        seg = used[(used.index > t_end - w) & (used.index <= t_end)]
        if len(seg) < c["window_min"] / 2:
            return None
        t = np.array([hours_between(t_end, ts) for ts in seg.index])
        return Line(t, seg.to_numpy())

    line = slope(now)
    if line is None:
        return {"status": "мало даних"}
    usual = [ln.b for k in range(1, c["baseline_days"] + 1)
             if (ln := slope(now - pd.Timedelta(days=k))) is not None]
    usual_p90 = float(np.percentile(usual, 90)) if usual else None
    limit = max(c["min_pct_per_h"], c["factor"] * max(usual_p90 or 0.0, 0.0))
    level_now = line.at(0)[0]
    return {
        "status": "ok",
        "level_now": level_now,
        "speed_pct_h": line.b,
        "usual_pct_h": usual_p90,
        "usual_days": len(usual),
        "limit_pct_h": limit,
        "unusual": bool(line.b > limit),
        "pred": {1: clip_pct(line.at(1))},
        "hours_to_threshold": hours_to(level_now, line.b, thr, 24 * 365),
    }


def speed_without_jumps(used: pd.Series, start: pd.Timestamp, now: pd.Timestamp,
                        steps: list[dict], regular: dict, cfg: dict) -> float:
    """Швидкість росту (%/год) без регулярних стрибків: з ряду віднімаємо кожен
    регулярний стрибок (і додаємо назад його видалення). Інакше на «сходах»
    тренд уже містить бекапи, і наступний бекап порахувався б двічі."""
    data = used[(used.index > start) & (used.index <= now)].copy()
    for s in steps:
        if s["size_pp"] > 0 and s["start"] >= start and matches_regular(s, regular, cfg):
            data[data.index > s["end"]] -= s["size_pp"]
            removal = removal_after(s, steps, cfg)
            if removal:
                data[data.index > removal["end"]] -= removal["size_pp"]   # size_pp < 0 -> додаємо
    hourly = data.resample("1h").mean().dropna()
    if len(hourly) < cfg["trend"]["min_hours"]:
        return 0.0
    t = np.array([hours_between(now, ts + pd.Timedelta(minutes=30)) for ts in hourly.index])
    return max(Line(t, hourly.to_numpy()).b, 0.0)


def next_jump(used: pd.Series, free: pd.Series, now: pd.Timestamp, steps: list[dict],
              regular: dict | None, trend_res: dict, cfg: dict) -> dict:
    """Питання 3: чи поміститься наступний регулярний стрибок."""
    if regular is None:
        return {"found": False, "text": "регулярних стрибків немає"}
    thr = cfg["threshold_pct"]
    day = now.normalize()
    at = day + pd.Timedelta(minutes=regular["tod_min"])
    if at <= now:
        at += pd.Timedelta(days=1)
    hours_until = hours_between(now, at)
    free_now = float(free.iloc[-1])
    total_gb = total_size_gb(used, free)
    # до стрибка вільне місце зменшиться на повільний ріст (без самих стрибків)...
    start = trend_res.get("start", used.index[0])
    grow_pct = speed_without_jumps(used, start, now, steps, regular, cfg) * hours_until
    grow_gb = grow_pct * total_gb / 100 if total_gb else 0.0
    # ...і збільшиться, якщо попередній стрибок (бекап) ще мають видалити
    back_gb = trend_res["excluded"]["size_gb"] if trend_res.get("excluded") else 0.0
    free_at = free_now - grow_gb + (back_gb if not np.isnan(back_gb) else 0.0)
    size_gb = regular["size_gb"]
    # Два різні «бракує»: фізично (до 100 %) і щоб лишитися нижче порогу
    short_gb = max(size_gb - free_at, 0.0)
    keep_free_gb = (100 - thr) / 100 * total_gb if total_gb else 0.0
    free_to_stay_below_gb = max(size_gb + keep_free_gb - free_at, 0.0)
    fits = short_gb == 0
    level_after = 100 * (1 - (free_at - size_gb) / total_gb) if total_gb else None
    if not fits:
        text = (f"не поміститься: фізично бракує {short_gb:.0f} ГБ; щоб лишитися нижче {thr} % — "
                f"звільнити {free_to_stay_below_gb:.0f} ГБ")
    elif free_to_stay_below_gb > 0:
        text = (f"поміститься, але диск буде вище {thr} % ({level_after:.0f} %); щоб лишитися "
                f"нижче — звільнити {free_to_stay_below_gb:.0f} ГБ")
    else:
        text = "поміститься"
    return {"found": True, "at": at, "hours_until": hours_until, "size_gb": size_gb,
            "free_now_gb": free_now, "free_at_gb": free_at, "fits": bool(fits),
            "short_gb": short_gb, "free_to_stay_below_gb": free_to_stay_below_gb,
            "above_threshold_after": bool(level_after is not None and level_after > thr),
            "level_after_pct": level_after, "days_seen": regular["days"], "text": text}


def total_size_gb(used: pd.Series, free: pd.Series) -> float | None:
    """Розмір диска з останніх значень: вільно / (1 − зайнято)."""
    u, f = float(used.iloc[-1]), float(free.iloc[-1])
    return f / (1 - u / 100) if u < 99.9 else None


# ---------------------------------------------------------------------------
# Методи: кожен відповідає на ті самі питання (наївні — лише рівні)
# ---------------------------------------------------------------------------

class LinearMethod:
    """Робочий метод: лінійна регресія (три питання)."""
    name = "linear"

    def predict(self, used: pd.Series, free: pd.Series, now: pd.Timestamp, cfg: dict) -> dict:
        steps = find_steps(used, free, cfg)
        regular = find_regular_rise(steps, now, cfg)
        tr = trend(used, free, now, steps, regular, cfg)
        return {"now_pct": float(used.iloc[-1]), "now_free_gb": float(free.iloc[-1]),
                "trend": tr, "fast_growth": fast_growth(used, now, cfg),
                "jump": next_jump(used, free, now, steps, regular, tr, cfg)}


class NaiveLastMethod:
    """Лінійка для порівняння: «як зараз» — на будь-який горизонт рівень не зміниться."""
    name = "naive_last"

    def predict(self, used, free, now, cfg):
        v = float(used.iloc[-1])
        thr = cfg["threshold_pct"]
        h = 0.0 if v >= thr else None
        return {"trend": {"status": "ok", "level_now": v, "pred": {24: (v, v, v), 168: (v, v, v)},
                          "hours_to_threshold": h},
                "fast_growth": {"status": "ok", "level_now": v, "pred": {1: (v, v, v)},
                                "hours_to_threshold": h, "unusual": False},
                "jump": {"found": False, "text": "метод не відповідає на це питання"}}


class NaiveDailyMethod:
    """Лінійка для порівняння: «як учора о цій порі» (+1 год — як було 23 год тому)."""
    name = "naive_daily"

    def predict(self, used, free, now, cfg):
        def at(h):
            # значення на добу раніше за цільовий момент (для +7 днів — те саме, що «як зараз»)
            back = now + pd.Timedelta(hours=h) - pd.Timedelta(days=int(np.ceil(h / 24)))
            v = used.asof(back)
            return float(v) if pd.notna(v) else float(used.iloc[-1])
        v = float(used.iloc[-1])
        thr = cfg["threshold_pct"]
        h = 0.0 if v >= thr else None
        p1, p24, p168 = at(1), at(24), at(168)
        return {"trend": {"status": "ok", "level_now": v,
                          "pred": {24: (p24, p24, p24), 168: (p168, p168, p168)},
                          "hours_to_threshold": h},
                "fast_growth": {"status": "ok", "level_now": v, "pred": {1: (p1, p1, p1)},
                                "hours_to_threshold": h, "unusual": False},
                "jump": {"found": False, "text": "метод не відповідає на це питання"}}


METHODS = {m.name: m for m in (LinearMethod, NaiveLastMethod, NaiveDailyMethod)}


def get_method(name: str):
    if name not in METHODS:
        raise SystemExit(f"Невідомий метод прогнозу «{name}». Є: {', '.join(METHODS)}")
    return METHODS[name]()


# ---------------------------------------------------------------------------
# Рядки для таблиці predictions
# ---------------------------------------------------------------------------

def fmt_ts(t) -> str | None:
    return t.strftime(TS_FORMAT) if t is not None else None


def to_rows(result: dict, method: str) -> list[dict]:
    """Відповіді на три питання -> рядки predictions (без ts, server_id, metric)."""
    rows = []
    tr = result["trend"]
    tr_details = {"status": tr.get("status"), "start": fmt_ts(tr.get("start")),
                  "speed_pct_day": tr.get("speed_pct_day"), "speed_gb_day": tr.get("speed_gb_day"),
                  "points_h": tr.get("points")}
    if tr.get("excluded"):
        tr_details["excluded_jump"] = {"from": fmt_ts(tr["excluded"]["from"]),
                                       "size_gb": tr["excluded"]["size_gb"],
                                       "expected_removal_by": fmt_ts(tr["excluded"]["expected_removal_by"])}
    for h in (24, 168):
        v = tr.get("pred", {}).get(h, (None, None, None))
        rows.append({"model": method, "kind": "trend", "horizon_h": h, "predicted_value": v[0],
                     "lower_value": v[1], "upper_value": v[2],
                     "hours_to_threshold": tr.get("hours_to_threshold"), "details": tr_details})

    fg = result["fast_growth"]
    v = fg.get("pred", {}).get(1, (None, None, None))
    rows.append({"model": method, "kind": "fast_growth", "horizon_h": 1, "predicted_value": v[0],
                 "lower_value": v[1], "upper_value": v[2],
                 "hours_to_threshold": fg.get("hours_to_threshold"),
                 "details": {"status": fg.get("status"), "speed_pct_h": fg.get("speed_pct_h"),
                             "usual_pct_h": fg.get("usual_pct_h"), "limit_pct_h": fg.get("limit_pct_h"),
                             "unusual": fg.get("unusual")}})

    j = result["jump"]
    rows.append({"model": method, "kind": "jump", "horizon_h": j.get("hours_until"),
                 "predicted_value": j.get("level_after_pct"), "lower_value": None, "upper_value": None,
                 "hours_to_threshold": None,
                 "details": {k: (fmt_ts(v) if isinstance(v, pd.Timestamp) else v)
                             for k, v in j.items()}})
    return rows


# ---------------------------------------------------------------------------
# Робота з базою
# ---------------------------------------------------------------------------

def list_disks(conn: sqlite3.Connection, server_id: int) -> list[str]:
    rows = conn.execute("SELECT DISTINCT metric FROM metrics_clean WHERE server_id = ? "
                        "AND metric LIKE 'disk_used_pct:%' ORDER BY metric", (server_id,)).fetchall()
    return [r[0].split(":", 1)[1] for r in rows]


def load_disk(conn: sqlite3.Connection, server_id: int, drive: str, now: pd.Timestamp,
              cfg: dict) -> tuple[pd.Series, pd.Series]:
    """Ряди «зайнято, %» і «вільно, ГБ» за history_days до now (без викидів)."""
    since = now - pd.Timedelta(days=cfg["history_days"])
    df = pd.read_sql_query(
        "SELECT ts, metric, value FROM metrics_clean WHERE server_id = ? AND metric IN (?, ?) "
        "AND ts > ? AND ts <= ? AND is_outlier = 0 ORDER BY ts", conn,
        params=[server_id, f"disk_used_pct:{drive}", f"disk_free_gb:{drive}",
                since.strftime(TS_FORMAT), now.strftime(TS_FORMAT)])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    used = df[df["metric"] == f"disk_used_pct:{drive}"].set_index("ts")["value"]
    free = df[df["metric"] == f"disk_free_gb:{drive}"].set_index("ts")["value"]
    return used, free


def forecast_server(conn: sqlite3.Connection, server_id: int, now: pd.Timestamp | None,
                    cfg: dict, method_name: str | None = None) -> tuple[pd.Timestamp, dict]:
    """Прогноз для всіх дисків сервера. now=None — останні дані. Повертає (момент, {диск: відповіді})."""
    if now is None:
        last = conn.execute("SELECT MAX(ts) FROM metrics_clean WHERE server_id = ? "
                            "AND metric LIKE 'disk_used_pct:%'", (server_id,)).fetchone()[0]
        if last is None:
            return None, {}
        now = pd.Timestamp(last)
    method = get_method(method_name or cfg["method"])
    out = {}
    for drive in list_disks(conn, server_id):
        used, free = load_disk(conn, server_id, drive, now, cfg)
        if len(used) < 2 or free.empty:
            continue
        out[drive] = method.predict(used, free, now, cfg)
    return now, out


def rounded(value):
    """Округлює числа в details_json до 3 знаків (замість -6.35e-14 — 0.0)."""
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return round(float(value), 3)
    return value


def save_predictions(conn: sqlite3.Connection, server_id: int, now: pd.Timestamp,
                     results: dict, method: str) -> int:
    rows = []
    for drive, result in results.items():
        for r in to_rows(result, method):
            rows.append((now.strftime(TS_FORMAT), server_id, f"disk_used_pct:{drive}", r["model"],
                         r["kind"], r["horizon_h"], r["predicted_value"], r["lower_value"],
                         r["upper_value"], r["hours_to_threshold"],
                         json.dumps(rounded(r["details"]), ensure_ascii=False, default=float)))
    conn.executemany("INSERT INTO predictions (ts, server_id, metric, model, kind, horizon_h, "
                     "predicted_value, lower_value, upper_value, hours_to_threshold, details_json) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Друк
# ---------------------------------------------------------------------------

def f1(v, unit="") -> str:
    return "—" if v is None else f"{v:.1f}{unit}"


def hours_text(h) -> str:
    if h is None:
        return "не досягне"
    return f"{h:.0f} год" if h < 72 else f"{h / 24:.1f} дн"


def print_result(server: str, now: pd.Timestamp, results: dict, cfg: dict) -> None:
    thr = cfg["threshold_pct"]
    print(f"\n{server}, момент {now:%Y-%m-%d %H:%M} UTC")
    for drive, r in results.items():
        tr, fg, j = r["trend"], r["fast_growth"], r["jump"]
        print(f"  Диск {drive}: зараз {f1(r.get('now_pct'), ' %')}, вільно {f1(r.get('now_free_gb'), ' ГБ')}")
        if tr.get("status") == "ok":
            p24, p168 = tr["pred"][24], tr["pred"][168]
            print(f"    1. Тренд з {tr['start']:%d.%m %H:%M}: {tr['speed_pct_day']:+.2f} %/добу"
                  f" ({f1(tr['speed_gb_day'], ' ГБ/добу')}); +24 год {p24[0]:.1f} % ({p24[1]:.1f}–{p24[2]:.1f}),"
                  f" +7 дн {p168[0]:.1f} % ({p168[1]:.1f}–{p168[2]:.1f}); до {thr} %: "
                  f"{hours_text(tr['hours_to_threshold'])}")
        else:
            print(f"    1. Тренд: {tr.get('status')} (з {tr['start']:%d.%m %H:%M}, {tr['points']} год)")
        if tr.get("excluded"):
            e = tr["excluded"]
            print(f"       без тимчасового стрибка з {e['from']:%d.%m %H:%M} (+{e['size_gb']:.0f} ГБ),"
                  f" видалення очікується до {e['expected_removal_by']:%d.%m %H:%M}")
        if fg.get("status") == "ok":
            usual = "—" if fg["usual_pct_h"] is None else f"{fg['usual_pct_h']:.2f}"
            print(f"    2. Ріст за {cfg['fast_growth']['window_min']} хв: {fg['speed_pct_h']:+.2f} %/год "
                  f"(зазвичай до {usual}, межа {fg['limit_pct_h']:.2f}) -> "
                  f"{'НЕЗВИЧНИЙ' if fg['unusual'] else 'звичайний'}; +1 год {fg['pred'][1][0]:.1f} %; "
                  f"до {thr} %: {hours_text(fg['hours_to_threshold'])}")
        else:
            print(f"    2. Ріст: {fg.get('status')}")
        if j["found"]:
            print(f"    3. Регулярний стрибок ~{j['size_gb']:.0f} ГБ ({j['days_seen']} дн. з 7), наступний"
                  f" {j['at']:%d.%m %H:%M}: вільно буде {j['free_at_gb']:.0f} ГБ -> {j['text']}")
        else:
            print(f"    3. {j['text']}")


def run(server: str | None = None, at: str | None = None, method: str | None = None,
        db_path: Path = DB_PATH) -> None:
    cfg = load_config()
    init_db(db_path)
    now = pd.Timestamp(at, tz="UTC") if at else None
    with db_session(db_path) as conn:
        servers = conn.execute("SELECT id, name FROM servers ORDER BY id").fetchall()
        for row in servers:
            if server and row["name"] != server:
                continue
            moment, results = forecast_server(conn, row["id"], now, cfg, method)
            if not results:
                continue
            print_result(row["name"], moment, results, cfg)
            if at is None:
                n = save_predictions(conn, row["id"], moment, results, method or cfg["method"])
                print(f"  -> записано в predictions: {n} рядків")


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогноз заповнення дисків")
    parser.add_argument("--server", help="лише цей сервер")
    parser.add_argument("--at", help="момент у минулому, UTC (\"YYYY-MM-DD HH:MM\"); лише друк")
    parser.add_argument("--method", help="метод з core/forecast.yaml за замовчуванням")
    args = parser.parse_args()
    run(args.server, args.at, args.method)


if __name__ == "__main__":
    main()
