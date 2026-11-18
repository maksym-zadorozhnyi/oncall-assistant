"""ETL: сирі дані (metrics, disk_usage) -> очищені дані metrics_clean.

Що робить для кожного сервера і кожного показника:
  1. Сітка 1 хв: секунди відкидаються (12:21:58 -> 12:21), час — UTC.
     Два записи в одній хвилині усереднюються.
  2. Неможливі значення (CPU > 100 %, від'ємний диск...) відкидаються.
  3. Пропуски до 2 хв заповнюються інтерполяцією (is_imputed = 1),
     довші лишаються пропусками (рядків немає).
  4. Викиди позначаються is_outlier = 1, але не видаляються.
     Викид — короткий сплеск: значення далеко і від рівня «до»
     (медіана 30 хв перед ним), і від рівня «після» (медіана 30 хв після),
     з одного боку від обох. Сходинка і повільний ріст — не викиди.
     Для дисків викидів не шукаємо.
Сирі таблиці не змінюються.

Запуск:
  python -m core.etl            # обробити нові дані (і перерахувати останню годину)
  python -m core.etl --reset    # перебудувати metrics_clean з нуля
  python -m core.etl --demo     # штучний ряд -> docs/experiments/etl_demo.png і etl_demo.md
"""
import argparse
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.db import DB_PATH, PROJECT_ROOT, db_session, init_db

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
EXPERIMENTS_DIR = PROJECT_ROOT / "docs" / "experiments"

# Колонки таблиці metrics, які стають окремими показниками в metrics_clean
METRIC_COLUMNS = ["cpu_pct", "ram_pct", "disk_read_mbs", "disk_write_mbs",
                  "net_in_mbs", "net_out_mbs", "app_response_ms", "app_error_pct"]

# Показники у відсотках: можливі значення лише 0..100. Решта — лише >= 0.
PERCENT_METRICS = {"cpu_pct", "ram_pct", "app_error_pct", "disk_used_pct"}

# Пропуск такої довжини (хв) і коротший заповнюємо, довший — лишаємо
MAX_IMPUTE_GAP_MIN = 2

# Пошук викидів
OUTLIER_WINDOW_MIN = 30      # вікно рівнів «до» і «після», хв
OUTLIER_MIN_POINTS = 10      # менше точок у вікні — рівень невідомий, викид не ставимо
OUTLIER_IQR_K = 3            # «далеко» = більше за 3 звичайні розкиди (IQR вікна «до»)
# ...але не менше за мінімальне відхилення: (абсолютне, частка від рівня «до»)
OUTLIER_MIN_DELTA = {
    "cpu_pct": (15, 0),          # п. п.
    "ram_pct": (10, 0),          # п. п.
    "app_error_pct": (5, 0),     # п. п.
    "app_response_ms": (300, 0.5),
    "disk_read_mbs": (1, 0.5),
    "disk_write_mbs": (1, 0.5),
    "net_in_mbs": (1, 0.5),
    "net_out_mbs": (1, 0.5),
}
# Для цих показників викидів не шукаємо: диск міняється сходинками
NO_OUTLIER_PREFIXES = ("disk_used_pct", "disk_free_gb")

# Скільки останніх хвилин перераховувати при кожному запуску:
# позначка викиду стає остаточною, коли після точки є 30 хв даних
RECALC_MIN = 60
# Скільки хвилин сирих даних брати перед перерахунком — для рівня «до» і пропусків
CONTEXT_MIN = OUTLIER_WINDOW_MIN + MAX_IMPUTE_GAP_MIN + 1


# ---------------------------------------------------------------------------
# Очищення одного ряду
# ---------------------------------------------------------------------------

def base_metric(metric: str) -> str:
    """'disk_used_pct:C:' -> 'disk_used_pct'."""
    return metric.split(":")[0]


def possible_mask(metric: str, values: pd.Series) -> pd.Series:
    """True для можливих значень, False для неможливих (і для порожніх)."""
    ok = values.notna() & (values >= 0)
    if base_metric(metric) in PERCENT_METRICS:
        ok &= values <= 100
    return ok


def find_outliers(s: pd.Series, metric: str) -> pd.Series:
    """Позначає короткі сплески. s — ряд на 1-хв сітці (пропуски = NaN)."""
    no = pd.Series(False, index=s.index)
    if metric.startswith(NO_OUTLIER_PREFIXES):
        return no

    w, mp = OUTLIER_WINDOW_MIN, OUTLIER_MIN_POINTS
    # Рівень «до»: медіана 30 хв перед точкою (shift(1) — без самої точки)
    roll = s.rolling(w, min_periods=mp)
    before = roll.median().shift(1)
    spread_before = (roll.quantile(0.75) - roll.quantile(0.25)).shift(1)
    # Рівень «після»: те саме, але ряд перевернуто в часі
    roll_rev = s[::-1].rolling(w, min_periods=mp)
    after = roll_rev.median()[::-1].shift(-1)
    spread_after = (roll_rev.quantile(0.75) - roll_rev.quantile(0.25))[::-1].shift(-1)
    # Звичайний розкид — менший з двох: наприкінці сплеску вікно «до» вже містить
    # сам сплеск і розкид там завеликий, а вікно «після» чисте (і навпаки на початку)
    spread = np.fmin(spread_before, spread_after)

    abs_min, rel = OUTLIER_MIN_DELTA.get(base_metric(metric), (0, 0.5))
    # Відносний поріг — від більшого з рівнів: інакше на початку тривалого
    # високого рівня (нічний бекап, 0,3 -> 120 МБ/с) поріг рахувався б від 0,3
    # і звичайні коливання навколо 120 ставали б «викидами»
    level = np.fmax(before.abs(), after.abs())
    threshold = np.maximum(OUTLIER_IQR_K * spread, np.maximum(abs_min, rel * level))

    # Вище за обидва рівні або нижче за обидва — це сплеск.
    # Сходинка: значення збігається з рівнем «після» -> не викид.
    # Повільний ріст: вище за «до», але нижче за «після» -> не викид.
    up = (s - before > threshold) & (s - after > threshold)
    down = (before - s > threshold) & (after - s > threshold)
    # Порівняння з NaN дає False: на краю ряду (немає «після») викид не ставимо
    return (up | down).fillna(False)


def clean_series(raw: pd.Series, metric: str) -> tuple[pd.DataFrame, dict]:
    """Очищає один ряд.

    raw — значення з індексом-часом (pandas Timestamp, UTC), можна з секундами
    і з повторами. Повертає таблицю на 1-хв сітці (value, is_imputed, is_outlier;
    хвилин довгих пропусків у ній немає) і лічильники.
    """
    stats = {"dropped": 0, "dropped_minutes": [], "long_gaps": []}
    if raw.empty:
        return pd.DataFrame(columns=["value", "is_imputed", "is_outlier"]), stats

    # 1. Сітка: відкидаємо секунди
    raw = raw.copy()
    raw.index = raw.index.floor("min")

    # 2. Неможливі значення (NULL у сирих даних — це «немає значення», не помилка)
    present = raw.notna()
    ok = possible_mask(metric, raw)
    bad = present & ~ok
    stats["dropped"] = int(bad.sum())
    stats["dropped_minutes"] = list(raw.index[bad])
    values = raw[ok]
    if values.empty:
        return pd.DataFrame(columns=["value", "is_imputed", "is_outlier"]), stats

    # Два записи в одній хвилині — середнє
    s = values.groupby(level=0).mean()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="min")
    s = s.reindex(grid)

    # 3. Пропуски: довжина кожного пропуску (серії NaN поспіль)
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    short = missing & (run_len <= MAX_IMPUTE_GAP_MIN)
    for _, run in s[missing & ~short].groupby(run_id[missing & ~short]):
        stats["long_gaps"].append((run.index[0], len(run)))

    # 4. Викиди шукаємо на справжніх значеннях (до заповнення пропусків)
    outlier = find_outliers(s, metric)

    filled = s.interpolate(limit_area="inside")
    value = s.where(~short, filled)

    result = pd.DataFrame({
        "value": value,
        "is_imputed": short.astype(int),
        "is_outlier": (outlier & ~missing).astype(int),
    })
    return result.dropna(subset=["value"]), stats


# ---------------------------------------------------------------------------
# Читання сирих даних і запис у metrics_clean
# ---------------------------------------------------------------------------

def read_raw(conn: sqlite3.Connection, server_id: int, since: str | None) -> tuple[dict, int]:
    """Сирі ряди сервера: {показник: pd.Series} і кількість прочитаних записів."""
    where, params = "server_id = ?", [server_id]
    if since:
        where += " AND ts >= ?"
        params.append(since)

    series, records = {}, 0
    m = pd.read_sql_query(f"SELECT ts, {', '.join(METRIC_COLUMNS)} FROM metrics WHERE {where}",
                          conn, params=params)
    records += len(m)
    if not m.empty:
        idx = pd.to_datetime(m["ts"], utc=True)
        for col in METRIC_COLUMNS:
            col_values = pd.Series(m[col].to_numpy(dtype=float), index=idx)
            if col_values.notna().any():
                series[col] = col_values

    d = pd.read_sql_query(f"SELECT ts, drive, used_pct, free_gb FROM disk_usage WHERE {where}",
                          conn, params=params)
    records += len(d)
    for drive, part in d.groupby("drive"):
        idx = pd.to_datetime(part["ts"], utc=True)
        series[f"disk_used_pct:{drive}"] = pd.Series(part["used_pct"].to_numpy(dtype=float), index=idx)
        series[f"disk_free_gb:{drive}"] = pd.Series(part["free_gb"].to_numpy(dtype=float), index=idx)
    return series, records


def process_server(conn: sqlite3.Connection, server_id: int) -> dict:
    """Обробляє нові дані одного сервера. Повертає лічильники."""
    last = conn.execute("SELECT MAX(ts) FROM metrics_clean WHERE server_id = ?",
                        (server_id,)).fetchone()[0]
    start = read_from = None
    if last:
        # Перераховуємо останню годину, а сирі дані беремо ще трохи раніше (контекст)
        start = pd.Timestamp(last) - pd.Timedelta(minutes=RECALC_MIN)
        read_from = (start - pd.Timedelta(minutes=CONTEXT_MIN)).strftime(TS_FORMAT)

    series, records = read_raw(conn, server_id, read_from)
    stats = {"records": records, "rows": 0, "dropped": 0, "imputed": 0, "outliers": 0,
             "long_gaps": set(), "dropped_at": []}
    rows = []
    for metric, raw in series.items():
        clean, st = clean_series(raw, metric)
        if start is not None:
            clean = clean[clean.index >= start]
            st["dropped_minutes"] = [t for t in st["dropped_minutes"] if t >= start]
            st["long_gaps"] = [g for g in st["long_gaps"] if g[0] >= start]
        stats["dropped"] += len(st["dropped_minutes"])
        stats["dropped_at"] += [(t, metric) for t in st["dropped_minutes"]]
        stats["long_gaps"].update(st["long_gaps"])
        stats["imputed"] += int(clean["is_imputed"].sum())
        stats["outliers"] += int(clean["is_outlier"].sum())
        for ts, r in clean.iterrows():
            rows.append((ts.strftime(TS_FORMAT), server_id, metric, float(r["value"]),
                         int(r["is_imputed"]), int(r["is_outlier"])))

    # Старі рядки періоду перерахунку видаляємо і пишемо заново
    if start is not None:
        conn.execute("DELETE FROM metrics_clean WHERE server_id = ? AND ts >= ?",
                     (server_id, start.strftime(TS_FORMAT)))
    conn.executemany("INSERT OR REPLACE INTO metrics_clean "
                     "(ts, server_id, metric, value, is_imputed, is_outlier) "
                     "VALUES (?, ?, ?, ?, ?, ?)", rows)
    stats["rows"] = len(rows)
    stats["long_gaps"] = sorted(stats["long_gaps"])
    return stats


def run_etl(db_path: Path = DB_PATH, reset: bool = False) -> dict:
    """Запускає ETL для всіх серверів. Повертає {ім'я сервера: лічильники}."""
    init_db(db_path)
    result = {}
    with db_session(db_path) as conn:
        if reset:
            conn.execute("DELETE FROM metrics_clean")
        for row in conn.execute("SELECT id, name FROM servers ORDER BY id").fetchall():
            result[row["name"]] = process_server(conn, row["id"])
    return result


def print_stats(result: dict) -> None:
    for name, st in result.items():
        print(f"{name}: прочитано записів {st['records']}, записано рядків {st['rows']}, "
              f"відкинуто {st['dropped']}, заповнено {st['imputed']}, "
              f"викидів {st['outliers']}, довгих пропусків {len(st['long_gaps'])}")
        for t, n in st["long_gaps"]:
            print(f"    пропуск {t:%Y-%m-%d %H:%M} UTC, {n} хв")


# ---------------------------------------------------------------------------
# Демо: штучний ряд ~2 год з відомими відповідями (його ж використовують тести)
# ---------------------------------------------------------------------------

DEMO_START = pd.Timestamp("2026-01-05T10:00:00Z")
DEMO_MINUTES = 120
# Правильні відповіді: номери хвилин від початку ряду
DEMO_EXPECT = {
    "impossible": [10],                 # CPU 130 % -> відкинуто, хвилину заповнено
    "short_gap": [20],                  # пропуск 1 хв -> заповнено
    "long_gap": list(range(30, 40)),    # пропуск 10 хв -> лишився пропуском
    "spike": list(range(55, 60)),       # сплеск 5 хв до ~85 % -> викид
    "step": list(range(70, 90)),        # сходинка 30 -> 55 % -> не викид
    "growth": list(range(90, 120)),     # повільний ріст 55 -> 76 % -> не викид
}


def demo_raw() -> pd.DataFrame:
    """Сирі записи штучного ряду CPU: хвилина, час із секундами, значення."""
    rng = np.random.default_rng(7)
    rows = []
    for m in range(DEMO_MINUTES):
        if m in DEMO_EXPECT["short_gap"] or m in DEMO_EXPECT["long_gap"]:
            continue                                   # запису немає
        value = 30.0
        if m in DEMO_EXPECT["spike"]:
            value = 85.0
        elif m in DEMO_EXPECT["step"]:
            value = 55.0
        elif m in DEMO_EXPECT["growth"]:
            value = 55.0 + (m - 89) * 0.7
        value += rng.uniform(-1.5, 1.5)                # звичайні коливання
        if m in DEMO_EXPECT["impossible"]:
            value = 130.0                              # «збій агента»
        # Як у справжньому агенті: запис приходить на 0-5 с пізніше початку хвилини
        ts = DEMO_START + pd.Timedelta(minutes=m, seconds=float(rng.uniform(0, 5)))
        rows.append({"minute": m, "ts": ts.strftime(TS_FORMAT), "value": value})
    return pd.DataFrame(rows)


def run_demo_db(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Пише штучний ряд у базу db_path (сервер demo-1), запускає ETL.
    Повертає (сирі записи, очищений ряд з колонкою minute)."""
    raw = demo_raw()
    init_db(db_path)
    with db_session(db_path) as conn:
        conn.execute("INSERT INTO servers (name, kind, description) VALUES ('demo-1', 'sim', 'демо ETL')")
        sid = conn.execute("SELECT id FROM servers WHERE name = 'demo-1'").fetchone()[0]
        conn.executemany("INSERT INTO metrics (ts, server_id, cpu_pct, source) VALUES (?, ?, ?, 'demo')",
                         [(r.ts, sid, r.value) for r in raw.itertuples()])
    run_etl(db_path)
    with db_session(db_path) as conn:
        clean = pd.read_sql_query("SELECT ts, value, is_imputed, is_outlier FROM metrics_clean "
                                  "WHERE metric = 'cpu_pct' ORDER BY ts", conn)
    clean["minute"] = ((pd.to_datetime(clean["ts"], utc=True) - DEMO_START)
                       .dt.total_seconds() // 60).astype(int)
    return raw, clean


def demo() -> None:
    """Проганяє штучний ряд через ETL, зберігає графік і таблицю «було -> стало»."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        raw, clean = run_demo_db(Path(tmp) / "demo.db")

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_by_min = raw.set_index("minute")["value"]
    clean_by_min = clean.set_index("minute")

    # --- Графік ---
    fig, ax = plt.subplots(figsize=(11, 4.8))
    gap = DEMO_EXPECT["long_gap"]
    ax.axvspan(gap[0] - 0.5, gap[-1] + 0.5, color="#d9d9d9", alpha=0.6, lw=0,
               label="довгий пропуск (лишився)")
    ax.plot(raw_by_min.index, raw_by_min.values, "o", ms=4, color="#9a9a9a",
            label="сирі записи")
    # Очищений ряд — лінія; розрив на довгому пропуску
    line = clean_by_min["value"].reindex(range(DEMO_MINUTES))
    ax.plot(line.index, line.values, "-", lw=2, color="#2a6fdb", label="metrics_clean")
    bad = raw[raw["minute"].isin(DEMO_EXPECT["impossible"])]
    ax.plot(bad["minute"], bad["value"], "x", ms=10, mew=2.5, color="#c0392b",
            label="неможливе значення (відкинуто)")
    imp = clean_by_min[clean_by_min["is_imputed"] == 1]
    ax.plot(imp.index, imp["value"], "s", ms=8, mfc="white", mew=2, color="#e08a00",
            label="заповнено (is_imputed = 1)")
    out = clean_by_min[clean_by_min["is_outlier"] == 1]
    ax.plot(out.index, out["value"], "^", ms=9, color="#c0392b",
            label="викид (is_outlier = 1)")
    for x, text in [(57, "сплеск"), (80, "сходинка"), (105, "повільний ріст")]:
        ax.annotate(text, (x, 92 if text == "сплеск" else 66 if text == "сходинка" else 80),
                    ha="center", fontsize=9, color="#444444")
    ax.set_xlim(-1, DEMO_MINUTES)
    ax.set_ylim(0, 140)
    ax.set_xlabel("хвилина від початку ряду")
    ax.set_ylabel("CPU, %")
    ax.set_title("ETL на штучному ряді: сирі записи і результат", loc="left")
    ax.grid(axis="y", color="#eeeeee")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    png = EXPERIMENTS_DIR / "etl_demo.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    # --- Таблиця «було -> стало» ---
    def state(m: int) -> str:
        if m not in clean_by_min.index:
            return "пропуск лишився"
        r = clean_by_min.loc[m]
        if m in raw_by_min.index and m in DEMO_EXPECT["impossible"]:
            return "відкинуто, заповнено"
        if r["is_imputed"]:
            return "заповнено"
        if r["is_outlier"]:
            return "викид"
        return "без змін"

    show = [9, 10, 11, 19, 20, 21, 29, 30, 39, 40, 54, 55, 59, 60, 69, 70, 71, 90, 105, 119]
    lines = ["# ETL на штучному ряді: було → стало", "",
             "Ряд CPU на 2 год, згенерований `python -m core.etl --demo`; на ньому ж працюють",
             "тести `tests/test_etl.py`. Графік: `etl_demo.png`.", "",
             "## Очікувано і отримано", "",
             "| Що в ряді | Хвилини | Очікуємо | Отримано |", "|---|---|---|---|"]
    expect_text = {"impossible": ("неможливе значення 130 %", "відкинуто, хвилину заповнено"),
                   "short_gap": ("пропуск 1 хв", "заповнено, is_imputed = 1"),
                   "long_gap": ("пропуск 10 хв", "лишився пропуском"),
                   "spike": ("сплеск до ~85 % на 5 хв", "is_outlier = 1"),
                   "step": ("сходинка 30 → 55 %", "is_outlier = 0"),
                   "growth": ("повільний ріст 55 → 76 %", "is_outlier = 0")}
    for key, minutes in DEMO_EXPECT.items():
        what, expected = expect_text[key]
        got = sorted({state(m) for m in minutes})
        rng_text = f"{minutes[0]}–{minutes[-1]}" if len(minutes) > 1 else str(minutes[0])
        lines.append(f"| {what} | {rng_text} | {expected} | {', '.join(got)} |")
    lines += ["", f"Усього викидів: {int(clean['is_outlier'].sum())} "
              f"(очікуємо {len(DEMO_EXPECT['spike'])}).", "",
              "## Окремі хвилини", "",
              "| Хвилина | Було (сирий запис) | Стало (metrics_clean) | Що сталося |",
              "|---|---|---|---|"]
    for m in show:
        before = f"{raw_by_min[m]:.1f}" if m in raw_by_min.index else "—"
        after = f"{clean_by_min.loc[m, 'value']:.1f}" if m in clean_by_min.index else "—"
        lines.append(f"| {m} | {before} | {after} | {state(m)} |")
    md = EXPERIMENTS_DIR / "etl_demo.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Збережено: {png}\n           {md}")
    print("\n".join(lines[6:16]))


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL: сирі дані -> metrics_clean")
    parser.add_argument("--reset", action="store_true", help="перебудувати metrics_clean з нуля")
    parser.add_argument("--demo", action="store_true", help="демо на штучному ряді (тимчасова база)")
    args = parser.parse_args()

    if args.demo:
        demo()
        return
    result = run_etl(reset=args.reset)
    print_stats(result)


if __name__ == "__main__":
    main()
