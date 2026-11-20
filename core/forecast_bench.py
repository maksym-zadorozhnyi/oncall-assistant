"""Перевірка прогнозу диска (крок 2.2b): сценарний стенд, історія, звіт, графіки.

  python -m core.forecast_bench --run --seed 1          # стенд: прогони S1–S8 (налаштування)
  python -m core.forecast_bench --run --seed 2          # стенд: інші прогони (цифри для звіту)
  python -m core.forecast_bench --history               # 30-денна історія з oncall.db
  python -m core.forecast_bench --report --seed 2       # звіт і графіки з backtest.csv
  python -m core.forecast_bench --plot sim-2 E: "2026-11-10 06:00"     # момент історії (UTC)
  python -m core.forecast_bench --plot-run S4-03 --seed 2 [--at "..."] # прогін стенду

Прогноз у кожен момент бачить лише дані до нього. Методи: linear (робочий),
naive_last і naive_daily (лінійки). Результати: docs/experiments/forecast/backtest.csv
(по рядку на момент і метод) і runs.csv (опис прогонів).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core import forecast as fc
from core.db import DB_PATH, PROJECT_ROOT, db_session

OUT_DIR = PROJECT_ROOT / "docs" / "experiments" / "forecast"
BACKTEST_CSV = OUT_DIR / "backtest.csv"
RUNS_CSV = OUT_DIR / "runs.csv"
REPORT_MD = PROJECT_ROOT / "docs" / "experiments" / "forecast_report.md"
METHODS = ("linear", "naive_last", "naive_daily")
HORIZONS = (1, 24, 168)
RUNS_PER_SITUATION = 25
WORKERS = 4                     # процесів паралельно (пам'ять: ~150 МБ кожен)

# Кольори графіків
C_BEFORE = "#222222"     # дані до моменту (їх бачив прогноз)
C_AFTER = "#e08a00"      # що було насправді після
C_TREND = "#2a6fdb"      # тренд (питання 1)
C_FAST = "#8e44ad"       # швидкий ріст (питання 2)
C_JUMP = "#2e8b57"       # наступний стрибок (питання 3)
C_THR = "#c0392b"        # поріг 90 %


def load_history_disk(server: str, drive: str, moment: pd.Timestamp, cfg: dict,
                      db_path: Path = DB_PATH) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(зайнято до моменту, вільно до моменту, зайнято після моменту — до +7 днів)."""
    with db_session(db_path) as conn:
        row = conn.execute("SELECT id FROM servers WHERE name = ?", (server,)).fetchone()
        if row is None:
            raise SystemExit(f"Сервера {server} немає в базі")
        used, free = fc.load_disk(conn, row["id"], drive, moment, cfg)
        after, _ = fc.load_disk(conn, row["id"], drive, moment + pd.Timedelta(days=7),
                                dict(cfg, history_days=7))
    if used.empty:
        raise SystemExit(f"Немає даних {server} {drive} до {moment}")
    return used, free, after[after.index > moment]


def answers_text(r: dict, cfg: dict) -> str:
    """Три відповіді прогнозу коротким текстом для графіка."""
    thr = cfg["threshold_pct"]
    tr, fg, j = r["trend"], r["fast_growth"], r["jump"]
    lines = [f"Зараз {r['now_pct']:.1f} %, вільно {r['now_free_gb']:.0f} ГБ"]
    if tr.get("status") == "ok":
        lines.append(f"1. Тренд {tr['speed_pct_day']:+.2f} %/добу; "
                     f"{fc.until_text(tr['hours_to_threshold'], thr)}")
    else:
        lines.append(f"1. Тренд: {tr.get('status')}")
    if tr.get("excluded"):
        lines.append(f"   (без тимчасового стрибка +{tr['excluded']['size_gb']:.0f} ГБ — "
                     f"видалення очікується до {tr['excluded']['expected_removal_by']:%d.%m %H:%M})")
    if tr.get("excluded_growth_from") is not None:
        lines.append(f"   (без незвичного росту з {tr['excluded_growth_from']:%d.%m %H:%M} — його описує п. 2)")
    if fg.get("status") == "ok":
        after = (f" (лише після сходинки о {fg['after_step_from']:%H:%M})"
                 if fg.get("after_step_from") is not None else "")
        lines.append(f"2. Ріст {fg['speed_pct_h']:+.2f} %/год{after} -> "
                     f"{'НЕЗВИЧНИЙ' if fg['unusual'] else 'звичайний'}; "
                     f"{fc.until_text(fg['hours_to_threshold'], thr)}")
    if j["found"]:
        lines.append(f"3. Стрибок ~{j['size_gb']:.0f} ГБ о {j['at']:%d.%m %H:%M}:")
        lines.append(f"   {j['text']}")
    else:
        lines.append(f"3. {j['text']}")
    return "\n".join(lines)


def plot_moment(used: pd.Series, free: pd.Series, after: pd.Series, moment: pd.Timestamp,
                cfg: dict, title: str, png: Path) -> dict:
    """Малює прогноз у момент moment. Повертає відповіді прогнозу."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr = cfg["threshold_pct"]
    r = fc.LinearMethod().predict(used, free, moment, cfg)
    tr, fg, j = r["trend"], r["fast_growth"], r["jump"]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 9.5),
                                        gridspec_kw={"height_ratios": [0.5, 3, 2]})
    # Окремий рядок зверху — відповіді прогнозу текстом (щоб не закривали дані)
    ax0.axis("off")
    ax0.text(0.0, 1.0, answers_text(r, cfg), transform=ax0.transAxes, va="top", fontsize=9,
             family="monospace")

    # --- Верхня панель: дні ---
    left = moment - pd.Timedelta(days=4)
    if tr.get("start") is not None and tr["start"] > moment - pd.Timedelta(days=10):
        left = min(left, tr["start"] - pd.Timedelta(hours=12))
    right = moment + pd.Timedelta(days=7)
    before = used[used.index >= left]
    ax1.plot(before.index, before.values, "-", lw=1.2, color=C_BEFORE, label="дані до моменту (бачив прогноз)")
    ax1.plot(after.index, after.values, "-", lw=1.2, color=C_AFTER, label="що було насправді")
    if tr.get("status") == "ok":
        line = tr["line"]
        past_t = np.linspace(fc.hours_between(moment, max(tr["start"], left)), 0, 50)
        fut_t = np.linspace(0, 168, 200)
        to_time = lambda ts: [moment + pd.Timedelta(hours=float(h)) for h in ts]
        ax1.plot(to_time(past_t), [line.at(h)[0] for h in past_t], "--", lw=1.2, color=C_TREND)
        sh = tr.get("shift", 0.0)          # завершені різкі підйоми — сходинка вгору
        fut = np.array([fc.clip_pct(tuple(v + sh for v in line.at(h))) for h in fut_t])
        ax1.plot(to_time(fut_t), fut[:, 0], "-", lw=2, color=C_TREND, label="тренд (питання 1)")
        ax1.fill_between(to_time(fut_t), fut[:, 1], fut[:, 2], color=C_TREND, alpha=0.15, lw=0,
                         label="інтервал 10–90 %")
        for h in (24, 168):
            v, lo, hi = tr["pred"][h]
            ax1.errorbar([moment + pd.Timedelta(hours=h)], [v], yerr=[[v - lo], [hi - v]], fmt="o",
                         ms=6, color=C_TREND, capsize=4)
    if tr.get("excluded"):
        e = tr["excluded"]
        ax1.axvspan(e["from"], min(e["expected_removal_by"], right), color=C_JUMP, alpha=0.10, lw=0,
                    label="тимчасовий стрибок (не в тренді)")
    if j["found"]:
        ax1.axvline(j["at"], color=C_JUMP, lw=1.2, ls=":")
        ax1.plot([j["at"]], [min(j["level_after_pct"], 100)], "D", ms=8, color=C_JUMP,
                 mfc="white" if j["fits"] else C_JUMP, label="наступний стрибок (питання 3)")
    ax1.axhline(thr, color=C_THR, lw=1, ls="--", label=f"{thr} %")
    ax1.axvline(moment, color="#888888", lw=1)
    ax1.set_xlim(left, right)
    ax1.set_ylim(max(0, min(before.min(), after.min() if not after.empty else 100) - 8), 102)
    ax1.set_ylabel("зайнято, %")
    ax1.set_title("Дні: тренд і наступний стрибок", loc="left", fontsize=10)
    ax1.legend(loc="lower left", fontsize=8, ncol=3, frameon=False)

    # --- Нижня панель: години ---
    l2, r2 = moment - pd.Timedelta(hours=6), moment + pd.Timedelta(hours=3)
    b2 = used[used.index >= l2]
    a2 = after[after.index <= r2]
    ax2.plot(b2.index, b2.values, "-", lw=1.2, color=C_BEFORE)
    ax2.plot(a2.index, a2.values, "-", lw=1.2, color=C_AFTER)
    if fg.get("status") == "ok":
        line = fg["line"]
        w = cfg["fast_growth"]["window_min"] / 60
        ts = np.linspace(-w, 1, 60)
        ax2.plot([moment + pd.Timedelta(hours=float(h)) for h in ts],
                 [fc.clip_pct(line.at(h))[0] for h in ts], "-", lw=2, color=C_FAST,
                 label=f"ріст за {cfg['fast_growth']['window_min']} хв і прогноз на +1 год (питання 2)")
        v, lo, hi = fg["pred"][1]
        ax2.errorbar([moment + pd.Timedelta(hours=1)], [v], yerr=[[v - lo], [hi - v]], fmt="o", ms=6,
                     color=C_FAST, capsize=4)
    ax2.axhline(thr, color=C_THR, lw=1, ls="--")
    ax2.axvline(moment, color="#888888", lw=1)
    ax2.set_xlim(l2, r2)
    lo_y = min(b2.min(), a2.min() if not a2.empty else 100)
    ax2.set_ylim(max(0, lo_y - 5), 102)
    ax2.set_ylabel("зайнято, %")
    ax2.set_xlabel("час (UTC); сіра вертикаль — момент прогнозу")
    ax2.set_title("Години: незвичний ріст", loc="left", fontsize=10)
    ax2.legend(loc="lower left", fontsize=8, frameon=False)

    for ax in (ax1, ax2):
        ax.grid(color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return r


def plot_history_moment(server: str, drive: str, moment_text: str) -> Path:
    cfg = fc.load_config()
    moment = pd.Timestamp(moment_text, tz="UTC")
    used, free, after = load_history_disk(server, drive, moment, cfg)
    png = OUT_DIR / f"moment_{server}_{drive.rstrip(':')}_{moment:%Y%m%d-%H%M}.png"
    plot_moment(used, free, after, moment, cfg,
                f"Прогноз: {server}, диск {drive}, момент {moment:%Y-%m-%d %H:%M} UTC", png)
    return png


# ---------------------------------------------------------------------------
# Оцінка: моменти, попередження, рядки результатів
# ---------------------------------------------------------------------------

HOLD_MIN = 5          # «дійшов до 90 %» — тримається щонайменше 5 хв поспіль (як у правилах)


def sustained_above(used: pd.Series, thr: float) -> pd.Series:
    """True у хвилину, коли рівень ≥ thr уже HOLD_MIN хвилин поспіль (кінець такої серії)."""
    return (used >= thr).astype(int).rolling(HOLD_MIN).sum().eq(HOLD_MIN)


def first_reach(used: pd.Series, thr: float, after: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Момент, коли диск дійшов до порогу (початок першої серії ≥ thr завдовжки HOLD_MIN хв),
    не раніше after. Одна хвилина шуму на 90,0 % — не «дійшов»."""
    s = sustained_above(used, thr)
    if after is not None:
        s = s[used.index >= after + pd.Timedelta(minutes=HOLD_MIN - 1)]
    if not s.any():
        return None
    return s[s].index[0] - pd.Timedelta(minutes=HOLD_MIN - 1)


def moments_for(index: pd.DatetimeIndex, eval_from: pd.Timestamp, dense_events: list) -> list:
    """Моменти перевірки: кожні 6 год; у ±6 год від подій — кожні 5 хв."""
    end = index[-1] - pd.Timedelta(hours=1)
    out = set(pd.date_range(eval_from.ceil("6h"), end, freq="6h"))
    for ev in dense_events:
        if ev is None or pd.isna(ev):
            continue
        for t in pd.date_range((ev - pd.Timedelta(hours=6)).floor("5min"), ev + pd.Timedelta(hours=6),
                               freq="5min"):
            if eval_from <= t <= end:
                out.add(t)
    return sorted(out)


def warn_info(r: dict, cfg: dict) -> tuple[bool, str, float | None]:
    """Попередження: прогнозований час до 90 % < 72 год — за трендом, за незвичним
    ростом або «після наступного стрибка диск буде вище 90 %». Повертає
    (є попередження, причини, обіцяний час у годинах)."""
    reasons, hours = [], []
    h = r["trend"].get("hours_to_threshold")
    if h is not None and h < 72:
        reasons.append("тренд")
        hours.append(h)
    fg = r["fast_growth"]
    h = fg.get("hours_to_threshold")
    if fg.get("status") == "ok" and h is not None and h < 72 and (fg.get("unusual") or h == 0):
        reasons.append("ріст")
        hours.append(h)
    j = r["jump"]
    if j.get("found") and j.get("above_threshold_after") and j["hours_until"] < 72:
        reasons.append("стрибок")
        hours.append(j["hours_until"])
    return bool(reasons), "+".join(reasons), (min(hours) if hours else None)


def evaluate_series(used: pd.Series, free: pd.Series, moments: list, actions: list, cfg: dict,
                    meta: dict) -> list[dict]:
    """Проганяє всі методи по моментах. actions — часи втручань (прибирання людиною
    і щотижневе): горизонт, через який пройшло втручання, у MAE не входить."""
    thr = cfg["threshold_pct"]
    hist = pd.Timedelta(days=cfg["history_days"])
    last = used.index[-1]
    reach = sustained_above(used, thr).to_numpy()     # «дійшов до 90 %» (≥ 5 хв поспіль)
    # часи втручань як numpy datetime (UTC без поясу) — для швидкого порівняння
    acts = (pd.DatetimeIndex(sorted(actions)).tz_convert(None).to_numpy() if actions
            else np.array([], dtype="datetime64[ns]"))
    values = used.to_numpy()
    rows = []
    for m in moments:
        u = used[(used.index > m - hist) & (used.index <= m)]
        f = free[(free.index > m - hist) & (free.index <= m)]
        if len(u) < 60:
            continue
        now_pct = float(u.iloc[-1])
        for name in METHODS:
            t0 = time.perf_counter()
            r = fc.METHODS[name]().predict(u, f, m, cfg)
            ms = (time.perf_counter() - t0) * 1000
            warn, reasons, promised = warn_info(r, cfg)
            # чи дійде диск до 90 % (≥ 5 хв поспіль) за обіцяний час (але не менше 72 год)
            window = pd.Timedelta(hours=max(promised or 0, 72))
            fut = reach[(used.index > m) & (used.index <= m + window + pd.Timedelta(minutes=HOLD_MIN))]
            m_np = np.datetime64(m.tz_convert(None))
            row = {**meta, "moment": m, "method": name, "now_pct": now_pct, "warn": warn,
                   "reasons": reasons, "promised_h": promised,
                   "reached_in_window": bool(fut.any()),
                   # втручання у вікні: якщо почистили раніше, ніж диск дійшов до 90 %,
                   # попередження не можна назвати хибним — невідомо, що було б далі
                   "action_in_window": bool(((acts > m_np)
                                             & (acts <= np.datetime64((m + window).tz_convert(None)))).any()),
                   "ms": round(ms, 2)}
            preds = {1: r["fast_growth"].get("pred", {}).get(1),
                     24: r["trend"].get("pred", {}).get(24),
                     168: r["trend"].get("pred", {}).get(168)}
            for h in HORIZONS:
                target = m + pd.Timedelta(hours=h)
                p = preds[h]
                row[f"p{h}"], row[f"lo{h}"], row[f"hi{h}"] = (p if p else (np.nan,) * 3)
                row[f"a{h}"] = float(used.asof(target)) if target <= last else np.nan
                row[f"x{h}"] = bool(((acts > np.datetime64(m.tz_convert(None)))
                                     & (acts <= np.datetime64(target.tz_convert(None)))).any())
            j = r["jump"]
            row["jump_found"] = bool(j.get("found"))
            row["jump_fits"] = j.get("fits") if j.get("found") else None
            row["jump_at"] = j.get("at") if j.get("found") else None
            rows.append(row)
    return rows


def evaluate_bench_run(args: tuple) -> tuple[list[dict], dict]:
    """Один прогін стенду (окремий процес): генерація + оцінка."""
    seed, situation, k = args
    from simulator.disk_runs import generate
    cfg = fc.load_config()
    run = generate(seed, situation, k)
    used, free = run["used"], run["free"]
    # дійшов до 90 % — лише після початку перевірки (у тижні розігріву моментів немає)
    t90 = first_reach(used, cfg["threshold_pct"], after=run["eval_from"])
    ev = run["events"]
    dense = [ev.get("start"), ev.get("end"), ev.get("jump_at"), t90]
    moments = moments_for(used.index, run["eval_from"], dense)
    meta = {"source": "bench", "seed": seed, "run": run["id"], "situation": situation,
            "drive": run["drive"]}
    # Втручання — лише позапланові дії адміністратора. Щотижневе прибирання — звичайна,
    # передбачувана поведінка: не виключаємо ні з MAE, ні з хибних тривог
    admin = [a for a, kind in run["actions"] if kind == "адміністратор почистив"]
    rows = evaluate_series(used, free, moments, admin, cfg, meta)
    info = {"source": "bench", "seed": seed, "run": run["id"], "situation": situation,
            "drive": run["drive"], "t90": t90, "start": ev.get("start"), "end": ev.get("end"),
            "jump_at": ev.get("jump_at"), "jump_fits": run["jump_fits"],
            "actions": len(admin), "weekly_cleanups": run["info"].get("weekly_cleanups", 0),
            "info": json.dumps(run["info"], ensure_ascii=False)}
    return rows, info


def save_results(rows: list[dict], runs: list[dict], replace: dict) -> None:
    """Дописує результати в backtest.csv і runs.csv, замінюючи старі рядки
    з тими самими значеннями полів replace (напр. source=bench, seed=2)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, new in ((BACKTEST_CSV, pd.DataFrame(rows)), (RUNS_CSV, pd.DataFrame(runs))):
        if path.exists():
            old = pd.read_csv(path)
            # старі рядки цього ж прогону (той самий source і seed) замінюємо новими
            same = np.ones(len(old), dtype=bool)
            for col, val in replace.items():
                same &= (old[col].astype(str) == str(val)).to_numpy()
            new = pd.concat([old[~same], new], ignore_index=True)
        new.to_csv(path, index=False)


def run_bench(seed: int, runs_per_situation: int = RUNS_PER_SITUATION, workers: int = WORKERS) -> None:
    from simulator.disk_runs import SITUATIONS
    jobs = [(seed, s, k) for s in SITUATIONS for k in range(runs_per_situation)]
    print(f"Стенд: seed {seed}, {len(jobs)} прогонів ({runs_per_situation} на ситуацію), "
          f"процесів {workers}...")
    t0 = time.time()
    all_rows, all_runs = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, info) in enumerate(pool.map(evaluate_bench_run, jobs, chunksize=2), 1):
            all_rows += rows
            all_runs.append(info)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i} з {len(jobs)} ({time.time() - t0:.0f} с)")
    save_results(all_rows, all_runs, {"source": "bench", "seed": seed})
    print(f"Готово за {time.time() - t0:.0f} с: {len(all_rows)} рядків -> {BACKTEST_CSV}")


# ---------------------------------------------------------------------------
# 30-денна історія (oncall.db) — реалістичний змішаний приклад
# ---------------------------------------------------------------------------

# Звичайна поведінка дисків історії (без інциденту): яка це ситуація
HISTORY_BASELINE = {("sim-1", "D:"): "S2", ("sim-2", "E:"): "S3"}      # решта — S1
CAUSE_TO_SITUATION = {"fast": "S6", "retention_stopped": "S4", "log_burst": "S5"}


def load_full_disk(conn, server_id: int, drive: str) -> tuple[pd.Series, pd.Series]:
    df = pd.read_sql_query("SELECT ts, metric, value FROM metrics_clean WHERE server_id = ? "
                           "AND metric IN (?, ?) AND is_outlier = 0 ORDER BY ts", conn,
                           params=[server_id, f"disk_used_pct:{drive}", f"disk_free_gb:{drive}"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    used = df[df["metric"] == f"disk_used_pct:{drive}"].set_index("ts")["value"]
    free = df[df["metric"] == f"disk_free_gb:{drive}"].set_index("ts")["value"]
    return used, free


def history_incidents(conn) -> list[dict]:
    """Інциденти диска з ключа відповідей: disk_fill (диск — з параметрів) і збій бекапу
    через брак місця (стрибок, що не помістився, — S7)."""
    out = []
    for r in conn.execute("SELECT i.id, s.name, i.scenario, i.started_at, i.ended_at, i.params_json "
                          "FROM sim_incidents i JOIN servers s ON s.id = i.server_id "
                          "WHERE i.scenario IN ('disk_fill', 'backup_stale') ORDER BY i.id"):
        p = json.loads(r["params_json"] or "{}")
        start, end = pd.Timestamp(r["started_at"]), pd.Timestamp(r["ended_at"])
        if r["scenario"] == "disk_fill":
            drive = p.get("drive") or p["folder"][:2]
            out.append({"id": r["id"], "server": r["name"], "drive": drive, "start": start, "end": end,
                        "situation": CAUSE_TO_SITUATION[p.get("cause", "fast")], "jump_at": None})
        elif p.get("reason") == "no_space":
            out.append({"id": r["id"], "server": r["name"], "drive": "E:", "start": start, "end": end,
                        "situation": "S7", "jump_at": start, "jump_fits": False})
    return out


def run_history() -> None:
    cfg = fc.load_config()
    thr = cfg["threshold_pct"]
    t_start = time.time()
    all_rows, all_runs = [], []
    with db_session() as conn:
        incidents = history_incidents(conn)
        servers = conn.execute("SELECT id, name FROM servers WHERE kind = 'sim' ORDER BY id").fetchall()
        disks = [(s["id"], s["name"], d) for s in servers for d in fc.list_disks(conn, s["id"])]
        series = {(name, d): load_full_disk(conn, sid, d) for sid, name, d in disks}
    for (name, drive), (used, free) in series.items():
        incs = [i for i in incidents if i["server"] == name and i["drive"] == drive]
        # Втручання — лише позапланові дії адміністратора: кінець інциденту диска
        # з ключа відповідей («помітив і почистив»); щотижневе прибирання — ні
        actions = [i["end"] for i in incs if i["situation"] != "S7"]
        above = sustained_above(used, thr)
        crossings = list(used.index[above & ~above.shift(fill_value=False)])
        dense = [i["start"] for i in incs] + [i["end"] for i in incs] + \
                [i["jump_at"] for i in incs if i["jump_at"] is not None] + crossings
        eval_from = used.index[0] + pd.Timedelta(days=7)
        meta = {"source": "history", "seed": 0, "run": f"{name} {drive}", "drive": drive}
        rows = evaluate_series(used, free, moments_for(used.index, eval_from, dense), actions, cfg, meta)
        baseline = HISTORY_BASELINE.get((name, drive), "S1")
        for row in rows:
            row["situation"] = next((i["situation"] for i in incs if i["start"] <= row["moment"] < i["end"]),
                                    baseline)
        all_rows += rows
        for i in incs:
            win = used[(used.index >= i["start"]) & (used.index <= i["end"])]
            t90 = first_reach(win, thr)
            all_runs.append({"source": "history", "seed": 0, "run": f"{name} {drive}",
                             "situation": i["situation"], "drive": drive, "t90": t90,
                             "start": i["start"], "end": i["end"], "jump_at": i["jump_at"],
                             "jump_fits": i.get("jump_fits"), "actions": len(actions),
                             "info": json.dumps({"incident": i["id"]})})
        print(f"  {name} {drive}: {len(rows)} рядків, втручань {len(actions)}, інцидентів {len(incs)}")
    save_results(all_rows, all_runs, {"source": "history"})
    print(f"Історія: {len(all_rows)} рядків за {time.time() - t_start:.0f} с")


# ---------------------------------------------------------------------------
# Підрахунок результатів
# ---------------------------------------------------------------------------

def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    bt = pd.read_csv(BACKTEST_CSV, low_memory=False)
    for c in ("moment", "jump_at"):
        bt[c] = pd.to_datetime(bt[c], utc=True, errors="coerce")
    runs = pd.read_csv(RUNS_CSV)
    for c in ("t90", "start", "end", "jump_at"):
        runs[c] = pd.to_datetime(runs[c], utc=True, errors="coerce")
    return bt, runs


def streak_start(g: pd.DataFrame, flag: pd.Series, last_moment) -> pd.Timestamp:
    """Початок безперервної серії True (за порядком моментів), що закінчується в last_moment."""
    g = g.reset_index(drop=True)
    flag = flag.reset_index(drop=True)
    i = int(np.flatnonzero((g["moment"] == last_moment).to_numpy())[0])
    while i > 0 and flag.iloc[i - 1]:
        i -= 1
    return g["moment"].iloc[i]


def episodes(flag: pd.Series) -> int:
    """Скільки окремих серій True (епізодів) у ряду."""
    f = flag.to_numpy(dtype=bool)
    return int((f & ~np.r_[False, f[:-1]]).sum())


def run_outcomes(bt: pd.DataFrame, runs: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Для кожного прогону (інциденту історії) і методу: чи попередив вчасно,
    за скільки, хибні тривоги, «не поміститься» для стрибка."""
    out = []
    for _, run in runs.iterrows():
        sel = (bt["source"] == run["source"]) & (bt["seed"] == run["seed"]) & (bt["run"] == run["run"])
        for method in METHODS:
            g = bt[sel & (bt["method"] == method)].sort_values("moment")
            if run["source"] == "history":            # лише моменти навколо цього інциденту
                lo = run["start"] - pd.Timedelta(days=3)
                hi = (run["end"] if pd.notna(run["end"]) else run["start"]) + pd.Timedelta(days=1)
                g = g[(g["moment"] >= lo) & (g["moment"] <= hi)]
            if g.empty:
                continue
            res = {"key": run.name, "source": run["source"], "seed": run["seed"], "run": run["run"],
                   "situation": run["situation"], "method": method, "reached": pd.notna(run["t90"]),
                   "lead_h": np.nan, "miss": False}
            if pd.notna(run["t90"]):
                pre = g[g["moment"] < run["t90"]]
                w = pre[pre["warn"] & (pre["moment"] >= run["t90"] - pd.Timedelta(hours=72))]
                if w.empty:
                    res["miss"] = True
                else:
                    first = streak_start(pre, pre["warn"], w["moment"].max())
                    res["lead_h"] = (run["t90"] - first).total_seconds() / 3600
                    # Похибка прогнозованого моменту 90 %: що обіцяв прогноз («до 90 % — N год»)
                    # і скільки було насправді — у першому попередженні і по всій серії
                    streak = pre[(pre["moment"] >= first) & pre["warn"]]
                    actual = (run["t90"] - streak["moment"]).dt.total_seconds() / 3600
                    err = streak["promised_h"] - actual
                    res["t90_promised_first_h"] = float(streak["promised_h"].iloc[0])
                    res["t90_actual_first_h"] = float(actual.iloc[0])
                    res["t90_err_h"] = float(err.iloc[0])
                    res["t90_err_med_h"] = float(err.abs().median())
            false = (g["warn"] & (g["now_pct"] < thr) & ~g["reached_in_window"]
                     & ~g["action_in_window"].astype(bool))
            res["false_moments"] = int(false.sum())
            res["false_episodes"] = episodes(false)
            res["false_minutes"] = int(false.sum()) * 5      # грубо: щільні моменти — кожні 5 хв
            if pd.notna(run["jump_at"]):
                pre = g[(g["moment"] < run["jump_at"]) & (g["moment"] >= run["jump_at"] - pd.Timedelta(hours=24))]
                nf = pre["jump_fits"].astype(str) == "False"
                res["jump_truth_fits"] = str(run["jump_fits"]) == "True"
                res["jump_said_not_fit"] = bool(nf.any())
                res["jump_lead_h"] = ((run["jump_at"] - streak_start(pre, nf, pre[nf]["moment"].max()))
                                      .total_seconds() / 3600) if nf.any() else np.nan
            out.append(res)
    return pd.DataFrame(out)


def situation_summary(oc: pd.DataFrame, mt: pd.DataFrame) -> pd.DataFrame:
    """Короткий підсумок по ситуаціях (лінійний метод) — для таблиці seed 1 / 2 / 3."""
    lin = oc[oc["method"] == "linear"]
    rows = []
    for s, g in lin.groupby("situation"):
        r = g[g["reached"]]
        m = mt[(mt["situation"] == s) & (mt["method"] == "linear")]
        row = {"situation": s, "runs": len(g), "reached": len(r), "in_time": int((~r["miss"]).sum()),
               "lead_mean_h": r["lead_h"].mean(), "lead_min_h": r["lead_h"].min(),
               "misses": int(r["miss"].sum()), "false_runs": int((g["false_episodes"] > 0).sum()),
               "false_episodes": int(g["false_episodes"].sum())}
        for h in HORIZONS:
            row[f"mae{h}"] = float(m[f"mae{h}"].iloc[0]) if not m.empty else np.nan
        if "t90_err_med_h" in g:
            row["t90_err_first_med_h"] = r["t90_err_h"].abs().median() if len(r) else np.nan
            row["t90_err_all_med_h"] = r["t90_err_med_h"].median() if len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mae_table(bt: pd.DataFrame) -> pd.DataFrame:
    """MAE (п. п.) кожного методу на кожному горизонті по ситуаціях, лише «до втручання»."""
    rows = []
    for (sit, method), g in bt.groupby(["situation", "method"]):
        row = {"situation": sit, "method": method}
        for h in HORIZONS:
            ok = g[f"a{h}"].notna() & g[f"p{h}"].notna() & ~g[f"x{h}"].astype(bool)
            row[f"mae{h}"] = (g.loc[ok, f"p{h}"] - g.loc[ok, f"a{h}"]).abs().mean()
            row[f"n{h}"] = int(ok.sum())
            row[f"excl{h}"] = int((g[f"a{h}"].notna() & g[f"x{h}"].astype(bool)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Частка фактичних значень усередині інтервалу 10–90 % (лінійний метод)."""
    g = bt[bt["method"] == "linear"]
    rows = []
    for sit, s in g.groupby("situation"):
        row = {"situation": sit}
        for h in HORIZONS:
            ok = s[f"a{h}"].notna() & s[f"lo{h}"].notna() & ~s[f"x{h}"].astype(bool)
            inside = (s.loc[ok, f"a{h}"] >= s.loc[ok, f"lo{h}"] - 0.05) & \
                     (s.loc[ok, f"a{h}"] <= s.loc[ok, f"hi{h}"] + 0.05)
            row[f"cov{h}"] = inside.mean() if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Графіки для звіту
# ---------------------------------------------------------------------------

METHOD_LABEL = {"linear": "лінійний", "naive_last": "наївний «як зараз»",
                "naive_daily": "наївний «як учора»"}
METHOD_COLOR = {"linear": C_TREND, "naive_last": "#9a9a9a", "naive_daily": "#e08a00"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clean_axes(ax):
    ax.grid(color="#eeeeee")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_lead_times(oc: pd.DataFrame, png: Path) -> None:
    """Розподіл часу попередження (до 90 %) по ситуаціях — лінійний метод."""
    plt = _plt()
    lin = oc[(oc["method"] == "linear") & oc["reached"] & oc["lead_h"].notna()]
    sits = sorted(lin["situation"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for i, s in enumerate(sits):
        v = lin.loc[lin["situation"] == s, "lead_h"].to_numpy()
        ax.boxplot([v], positions=[i], widths=0.5, showfliers=False,
                   medianprops={"color": C_TREND, "lw": 2})
        ax.plot(i + rng.uniform(-0.15, 0.15, len(v)), v, "o", ms=4, color=C_TREND, alpha=0.5)
    ax.set_xticks(range(len(sits)), sits)
    ax.set_yscale("log")
    ax.set_ylabel("за скільки до 90 % попередив, год (лог. шкала)")
    ax.set_title("Час попередження по ситуаціях (лінійний метод)", loc="left", fontsize=10)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def plot_mae(mt: pd.DataFrame, png_bars: Path, png_heat: Path) -> None:
    """MAE за горизонтами (усі ситуації разом) і теплова карта ситуація × метод."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.27
    for k, m in enumerate(METHODS):
        g = mt[mt["method"] == m]
        vals = []
        for h in HORIZONS:
            w = g[f"n{h}"]
            vals.append(float((g[f"mae{h}"] * w).sum() / w.sum()) if w.sum() else np.nan)
        x = np.arange(len(HORIZONS)) + (k - 1) * width
        ax.bar(x, vals, width, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(HORIZONS)), ["+1 год", "+24 год", "+7 днів"])
    ax.set_ylabel("MAE, п. п.")
    ax.set_title("Середня помилка прогнозу рівня за горизонтами (усі ситуації, до втручання)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(png_bars, dpi=110)
    plt.close(fig)

    sits = sorted(mt["situation"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, h, name in zip(axes, HORIZONS, ["+1 год", "+24 год", "+7 днів"]):
        grid = np.array([[mt[(mt["situation"] == s) & (mt["method"] == m)][f"mae{h}"].squeeze()
                          for m in METHODS] for s in sits], dtype=float)
        ax.imshow(np.nan_to_num(grid, nan=0), cmap="Blues", aspect="auto", vmin=0,
                  vmax=np.nanpercentile(grid, 95) if np.isfinite(grid).any() else 1)
        for i in range(len(sits)):
            row = grid[i]
            best = np.nanargmin(row) if np.isfinite(row).any() else None
            for j in range(len(METHODS)):
                txt = "—" if np.isnan(row[j]) else f"{row[j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        fontweight="bold" if j == best else "normal",
                        color="white" if np.isfinite(row[j]) and row[j] > np.nanpercentile(grid, 70) else "#222")
        ax.set_xticks(range(len(METHODS)), ["лінійний", "«як зараз»", "«як учора»"], fontsize=8)
        ax.set_title(f"MAE {name}", fontsize=10)
    axes[0].set_yticks(range(len(sits)), sits)
    fig.suptitle("Ситуація × метод: MAE, п. п. (жирне — найменша; «—» — усі горизонти після втручання)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_heat, dpi=110)
    plt.close(fig)


def plot_false_alarms(oc: pd.DataFrame, png: Path) -> None:
    plt = _plt()
    sits = sorted(oc["situation"].unique())
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.27
    for k, m in enumerate(METHODS):
        g = oc[oc["method"] == m]
        vals = [int((g[g["situation"] == s]["false_episodes"] > 0).sum()) for s in sits]
        x = np.arange(len(sits)) + (k - 1) * width
        ax.bar(x, vals, width, color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax.set_xticks(range(len(sits)), sits)
    ax.set_ylabel("прогонів з хибною тривогою")
    ax.set_title("Хибні тривоги за методами", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def plot_runtime(bt: pd.DataFrame, png: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = [bt[bt["method"] == m]["ms"].mean() for m in METHODS]
    ax.barh([METHOD_LABEL[m] for m in METHODS], vals, color=[METHOD_COLOR[m] for m in METHODS])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f} мс", va="center", fontsize=8)
    ax.set_xlabel("середній час одного прогнозу (один диск, три питання), мс")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def gallery_moment(run: dict, cfg: dict, rows: pd.DataFrame) -> pd.Timestamp:
    """Показовий момент прогону: перше попередження перед 90 % (або за 6 год до 90 %);
    для S7 — за годину до бекапу; S8 — у розпалі піку; S1–S3 — середина перевірки."""
    ev = run["events"]
    used = run["used"]
    thr = cfg["threshold_pct"]
    if run["situation"] == "S7":
        return ev["jump_at"] - pd.Timedelta(hours=1)
    if run["situation"] == "S8":
        return (ev["start"] + pd.Timedelta(minutes=5)).floor("5min")
    if (used >= thr).any():
        t90 = used[used >= thr].index.min()
        w = rows[(rows["method"] == "linear") & rows["warn"] & (rows["moment"] < t90)
                 & (rows["moment"] >= t90 - pd.Timedelta(hours=72))]
        return w["moment"].min() if not w.empty else t90 - pd.Timedelta(hours=6)
    span = used.index[-1] - run["eval_from"]
    return (run["eval_from"] + span / 2).floor("1h")


def plot_bench_run(run_id: str, seed: int, at: str | None = None, png: Path | None = None,
                   rows: pd.DataFrame | None = None) -> Path:
    """PNG прогону стенду за номером (напр. S4-03): прогін відтворюється з (seed, номер)."""
    from simulator.disk_runs import generate
    cfg = fc.load_config()
    situation, k = run_id.split("-")
    run = generate(seed, situation, int(k))
    if at:
        moment = pd.Timestamp(at, tz="UTC")
    else:
        if rows is None:
            bt, _ = load_results()
            rows = bt[(bt["source"] == "bench") & (bt["seed"] == seed) & (bt["run"] == run_id)]
        moment = gallery_moment(run, cfg, rows)
    used, free = run["used"], run["free"]
    hist = pd.Timedelta(days=cfg["history_days"])
    u = used[(used.index > moment - hist) & (used.index <= moment)]
    f = free[(free.index > moment - hist) & (free.index <= moment)]
    after = used[(used.index > moment) & (used.index <= moment + pd.Timedelta(days=7))]
    png = png or OUT_DIR / f"moment_run_{seed}_{run_id}_{moment:%m%d-%H%M}.png"
    from simulator.disk_runs import SITUATIONS
    plot_moment(u, f, after, moment, cfg,
                f"Стенд, seed {seed}, прогін {run_id} ({SITUATIONS[situation]}), диск {run['drive']}, "
                f"момент {moment:%d.%m %H:%M}", png)
    return png


def plot_evolution(server: str, drive: str, moments: list[str], title: str, png: Path) -> None:
    """«Як змінювався прогноз»: тренд у кілька моментів одного інциденту."""
    plt = _plt()
    cfg = fc.load_config()
    thr = cfg["threshold_pct"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharey=True)
    for ax, mt_text in zip(axes.flat, moments):
        moment = pd.Timestamp(mt_text, tz="UTC")
        used, free, after = load_history_disk(server, drive, moment, cfg)
        r = fc.LinearMethod().predict(used, free, moment, cfg)
        tr = r["trend"]
        left = moment - pd.Timedelta(days=5)
        b = used[used.index >= left]
        ax.plot(b.index, b.values, "-", lw=1.1, color=C_BEFORE)
        ax.plot(after.index, after.values, "-", lw=1.1, color=C_AFTER)
        if tr.get("status") == "ok":
            line = tr["line"]
            ts = np.linspace(0, 168, 100)
            sh = tr.get("shift", 0.0)
            fut = np.array([fc.clip_pct(tuple(v + sh for v in line.at(h))) for h in ts])
            tt = [moment + pd.Timedelta(hours=float(h)) for h in ts]
            ax.plot(tt, fut[:, 0], "-", lw=2, color=C_TREND)
            ax.fill_between(tt, fut[:, 1], fut[:, 2], color=C_TREND, alpha=0.15, lw=0)
            text = (f"{tr['speed_pct_day']:+.1f} %/добу; {fc.until_text(tr['hours_to_threshold'], thr)}")
        else:
            text = f"тренд: {tr.get('status')}"
        warn, reasons, _ = warn_info(r, cfg)
        ax.axhline(thr, color=C_THR, lw=1, ls="--")
        ax.axvline(moment, color="#888888", lw=1)
        ax.set_xlim(left, moment + pd.Timedelta(days=7))
        ax.set_title(f"{moment:%d.%m %H:%M}: {text}" + (f" — ПОПЕРЕДЖЕННЯ ({reasons})" if warn else ""),
                     loc="left", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        _clean_axes(ax)
    axes[0, 0].set_ylabel("зайнято, %")
    axes[1, 0].set_ylabel("зайнято, %")
    fig.suptitle(title + " — чорне: дані до моменту; помаранчеве: що було далі; синє: тренд з інтервалом",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def plot_incident_evolution(inc_id: int, title: str, png: Path) -> None:
    """Моменти — частки тривалості інциденту з ключа відповідей (5, 25, 55, 85 %)."""
    with db_session() as conn:
        incs = {i["id"]: i for i in history_incidents(conn)}
    if inc_id not in incs:
        print(f"Інциденту #{inc_id} немає — графік «як змінювався прогноз» пропущено")
        return
    i = incs[inc_id]
    dur = i["end"] - i["start"]
    moments = [(i["start"] + dur * q).floor("5min").strftime("%Y-%m-%d %H:%M") for q in (0.05, 0.25, 0.55, 0.85)]
    plot_evolution(i["server"], i["drive"], moments, title, png)


# ---------------------------------------------------------------------------
# Звіт
# ---------------------------------------------------------------------------

def fmt_h(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.1f} год" if v < 48 else f"{v / 24:.1f} дн"


def build_report(seed: int) -> None:
    from simulator.disk_runs import SITUATIONS
    cfg = fc.load_config()
    thr = cfg["threshold_pct"]
    plt_dir = OUT_DIR
    bt_all, runs_all = load_results()
    bt = bt_all[(bt_all["source"] == "bench") & (bt_all["seed"] == seed)]
    runs = runs_all[(runs_all["source"] == "bench") & (runs_all["seed"] == seed)]
    oc = run_outcomes(bt, runs, thr)
    mt = mae_table(bt)
    cov = coverage_table(bt)

    plot_lead_times(oc, plt_dir / "lead_times.png")
    plot_mae(mt, plt_dir / "mae_horizons.png", plt_dir / "mae_heatmap.png")
    plot_false_alarms(oc, plt_dir / "false_alarms.png")
    plot_runtime(bt, plt_dir / "runtime.png")

    L = []
    L += ["# Прогноз заповнення диска: перевірка (крок 2.2b)", "",
          f"Згенеровано `python -m core.forecast_bench --report --seed {seed}`. "
          "Сирі результати — `docs/experiments/forecast/backtest.csv` (у Git не потрапляє, "
          "відтворюється командою `--run`), опис прогонів — `runs.csv`.", ""]
    conclusions = OUT_DIR / "conclusions.md"
    if conclusions.exists():
        L += [conclusions.read_text(encoding="utf-8").strip(), ""]

    # --- Що змінено під час налаштування + seed 1 / seed 2 (версія 1) / seed 3 ---
    changes = OUT_DIR / "changes.md"
    if changes.exists():
        L += [changes.read_text(encoding="utf-8").strip(), ""]
    seeds = {}
    for s_ in (1, seed):
        b_ = bt_all[(bt_all["source"] == "bench") & (bt_all["seed"] == s_)]
        r_ = runs_all[(runs_all["source"] == "bench") & (runs_all["seed"] == s_)]
        if not b_.empty:
            seeds[s_] = situation_summary(run_outcomes(b_, r_, thr), mae_table(b_)).set_index("situation")
    v1_path = OUT_DIR / "summary_v1_seed2.csv"
    v1 = pd.read_csv(v1_path).set_index("situation") if v1_path.exists() else None
    if seeds and v1 is not None:
        cols = [("seed 1 (налаштування)", seeds.get(1)), ("seed 2 (версія 1)", v1),
                (f"seed {seed} (фінал)", seeds.get(seed))]
        L += ["### Seed 1 / seed 2 (версія 1) / seed 3 поруч (лінійний метод)", "",
              "Версія 1 — старий код і старі визначення (одна хвилина ≥ 90 % = «дійшов»; щотижневе "
              "прибирання виключалось як втручання), тому порівнювати її з seed 1 / 3 можна лише "
              "приблизно.", "",
              "| Ситуація | Вчасно / дійшли: " + " · ".join(n for n, _ in cols) +
              " | Прогонів з хибною тривогою: " + " · ".join(n for n, _ in cols) +
              " | MAE +24 год: " + " · ".join(n for n, _ in cols) + " |", "|---|---|---|---|"]
        for s_ in sorted(SITUATIONS):
            def cell(df, f):
                if df is None or s_ not in df.index:
                    return "—"
                r = df.loc[s_]
                return f(r)
            a = " · ".join(cell(d, lambda r: f"{int(r['in_time'])}/{int(r['reached'])}") for _, d in cols)
            b = " · ".join(cell(d, lambda r: f"{int(r['false_runs'])}") for _, d in cols)
            c = " · ".join(cell(d, lambda r: "—" if pd.isna(r["mae24"]) else f"{r['mae24']:.2f}") for _, d in cols)
            L.append(f"| {s_} | {a} | {b} | {c} |")
        L += [""]
    L += ["## Як перевіряли", "",
          f"- **Сценарний стенд:** {runs['run'].nunique()} коротких незалежних прогонів "
          f"(по {RUNS_PER_SITUATION} на ситуацію), seed {seed}. Пороги налаштовано на інших прогонах "
          "(seed 1), тут — лише перевірка. Кожен прогін: 10–13 днів норми, одна ситуація, 2–4 дні після; "
          "параметри випадкові (розмір диска, початковий рівень, швидкості, час доби і день тижня, "
          "розмір стрибка, шум). Генерують ті самі класи, що й симулятор (`simulator/disk_runs.py`).",
          "- **Моменти перевірки:** кожні 6 год від 7-го дня; у ±6 год навколо початку і кінця ситуації, "
          "моменту 90 % і критичного бекапу — кожні 5 хв. У кожен момент метод бачить лише дані до нього.",
          "- **Методи:** лінійний (робочий, три питання); наївні «як зараз» і «як учора о цій порі» — лінійки.",
          f"- **Дійшов до {thr} %** — рівень ≥ {thr} % тримається щонайменше {HOLD_MIN} хв поспіль "
          "(як у правилах; одна хвилина шуму — не рахується), і лише після початку перевірки.",
          f"- **Попередження** — прогнозований час до {thr} % < 72 год (за трендом, за незвичним ростом "
          f"або «після наступного стрибка диск буде вище {thr} %»). **Вчасно** — попередження було "
          f"до моменту {thr} %; **за скільки** — від початку безперервної серії попереджень до "
          f"моменту {thr} %. **Хибна тривога** — попередження, після якого диск не дійшов до {thr} % "
          "ні за обіцяний час, ні за 72 год (і адміністратор не втручався). **Пропуск** — дійшов без "
          "попередження. **Похибка моменту 90 %** — обіцяний час («до 90 % — N год») мінус фактичний.",
          "- **MAE «до втручання»:** якщо між моментом і горизонтом адміністратор позапланово почистив "
          "диск, цей горизонт у MAE не входить — дію людини з ряду не передбачити. Щотижневе прибирання, "
          "видалення бекапу і тимчасового файлу — передбачувана поведінка, не виключаються "
          f"(щотижневих прибирань у прогонах: {int(runs['weekly_cleanups'].sum()) if 'weekly_cleanups' in runs else '—'}; "
          f"дій адміністратора: {int(runs['actions'].sum())}). Щільні моменти (кожні 5 хв біля подій) "
          "мають у MAE більшу вагу.",
          ""]

    # --- Підсумок по ситуаціях ---
    L += ["## Попередження по ситуаціях (лінійний метод)", "",
          f"| Ситуація | Прогонів | Дійшли до {thr} % | Попередив вчасно | За скільки (середнє) | "
          "Найгірше (найменше) | Пропуски | Прогонів з хибною тривогою | Епізодів хибних тривог |",
          "|---|---|---|---|---|---|---|---|---|"]
    lin = oc[oc["method"] == "linear"]
    for s in sorted(SITUATIONS):
        g = lin[lin["situation"] == s]
        r = g[g["reached"]]
        L.append(f"| {s} {SITUATIONS[s]} | {len(g)} | {len(r)} | "
                 f"{int((~r['miss']).sum()) if len(r) else '—'} з {len(r)} | {fmt_h(r['lead_h'].mean())} | "
                 f"{fmt_h(r['lead_h'].min())} | {int(r['miss'].sum())} | "
                 f"{int((g['false_episodes'] > 0).sum())} | {int(g['false_episodes'].sum())} |")
    L += ["", "Наївні методи попереджають лише тоді, коли диск уже вище порогу, тому «вчасно» в них "
          "немає за визначенням; для них показуємо лише MAE.", "",
          "![Час попередження](forecast/lead_times.png)", ""]

    # --- Похибка моменту 90 % ---
    if "t90_err_h" in lin:
        L += [f"## Похибка прогнозованого моменту {thr} % (лінійний метод)", "",
              f"Саме це бот напише черговому: «до {thr} % — N год». Похибка = обіцяно − насправді "
              "(+ — прогноз обіцяв пізніше, ніж сталося; − — раніше).", "",
              "| Ситуація | Прогонів | Перше попередження: обіцяв (медіана) | Насправді (медіана) | "
              "Похибка першого, медіана |·| | Найбільша |·| | Похибка по всій серії попереджень, медіана |·| |",
              "|---|---|---|---|---|---|---|"]
        for s in sorted(SITUATIONS):
            r = lin[(lin["situation"] == s) & lin["reached"] & lin["t90_err_h"].notna()]
            if r.empty:
                continue
            L.append(f"| {s} | {len(r)} | {fmt_h(r['t90_promised_first_h'].median())} | "
                     f"{fmt_h(r['t90_actual_first_h'].median())} | {fmt_h(r['t90_err_h'].abs().median())} | "
                     f"{fmt_h(r['t90_err_h'].abs().max())} | {fmt_h(r['t90_err_med_h'].median())} |")
        L += ["", "Перше попередження бот дає, коли до порогу ще далеко, тож і похибка першого "
              "найбільша; ближче до 90 % прогноз уточнюється (остання колонка).", ""]

    # --- Кожен пропуск окремо ---
    misses = lin[lin["reached"] & lin["miss"]]
    L += ["## Пропуски — кожен окремо", ""]
    if misses.empty:
        L += ["Пропусків немає.", ""]
    for _, m in misses.iterrows():
        run = runs[runs["run"] == m["run"]].iloc[0]
        g = bt[(bt["run"] == m["run"]) & (bt["method"] == "linear") & (bt["moment"] < run["t90"])]
        g = g.sort_values("moment").tail(4)
        L += [f"**{m['run']}** ({SITUATIONS[m['situation']]}): дійшов до {thr} % о "
              f"{run['t90']:%d.%m %H:%M}; параметри прогону: `{run['info']}`. Останні прогнози перед цим:", "",
              "| Момент | Рівень, % | Попередження | Причина | Обіцяно, год |", "|---|---|---|---|---|"]
        for _, row in g.iterrows():
            promised = "—" if pd.isna(row["promised_h"]) else f"{row['promised_h']:.1f}"
            reasons = row["reasons"] if isinstance(row["reasons"], str) else "—"
            L.append(f"| {row['moment']:%d.%m %H:%M} | {row['now_pct']:.1f} | "
                     f"{'так' if row['warn'] else 'ні'} | {reasons} | {promised} |")
        L += [""]

    # --- S7 ---
    s7 = lin[(lin["situation"] == "S7")]
    if not s7.empty and "jump_truth_fits" in s7:
        nf = s7[~s7["jump_truth_fits"].astype(bool)]
        fit = s7[s7["jump_truth_fits"].astype(bool)]
        last_ok = []
        for _, r in s7.iterrows():
            run = runs[runs["run"] == r["run"]].iloc[0]
            g = bt[(bt["run"] == r["run"]) & (bt["method"] == "linear") & (bt["moment"] < run["jump_at"])]
            last = g.sort_values("moment").iloc[-1]
            said_fits = str(last["jump_fits"]) != "False"
            last_ok.append(said_fits == bool(r["jump_truth_fits"]))
        s7 = s7.assign(last_ok=last_ok)
        L += ["## S7: регулярний стрибок (бекап) після вивантаження", "",
              "| Насправді | Прогонів | Хоч раз за добу до бекапу казав «не поміститься» | "
              "Перше «не поміститься» — за скільки до бекапу (середнє / найменше) | "
              "Вердикт за 5 хв до бекапу правильний |", "|---|---|---|---|---|",
              f"| не помістився | {len(nf)} | {int(nf['jump_said_not_fit'].sum())} | "
              f"{fmt_h(nf['jump_lead_h'].mean())} / {fmt_h(nf['jump_lead_h'].min())} | "
              f"{int(s7.loc[nf.index, 'last_ok'].sum())} з {len(nf)} |",
              f"| помістився впритул (запас 3–25 %) | {len(fit)} | {int(fit['jump_said_not_fit'].sum())} | — | "
              f"{int(s7.loc[fit.index, 'last_ok'].sum())} з {len(fit)} |", "",
              "Поки вивантаження ще йде, прогноз обережний: ріст, що триває, рахується до самого бекапу. "
              "Коли вивантаження скінчилося, вердикт стає точним.", ""]

    # --- S8 ---
    s8 = lin[lin["situation"] == "S8"]
    if not s8.empty:
        L += ["## S8: короткий різкий пік", "",
              f"Тривога була в {int((s8['false_episodes'] > 0).sum())} з {len(s8)} прогонів; "
              f"найдовша — ~{int(s8['false_minutes'].max())} хв. Стрибок за одну хвилину — це не «ріст» "
              "(правило `max_single_minute_share`), а коли файл видалили, тренд починається заново.", ""]

    # --- MAE ---
    L += ["## Точність рівня (MAE, п. п.) — до втручання", "",
          "| Ситуація | Метод | +1 год | +24 год | +7 днів | Виключено горизонтів (втручання): +1 год / +24 год / +7 днів |",
          "|---|---|---|---|---|---|"]
    for s in sorted(mt["situation"].unique()):
        g = mt[mt["situation"] == s]
        best = {h: g[f"mae{h}"].min() for h in HORIZONS}
        for m in METHODS:
            r = g[g["method"] == m].iloc[0]
            cells = []
            for h in HORIZONS:
                v = r[f"mae{h}"]
                cells.append("—" if pd.isna(v) else (f"**{v:.2f}**" if v == best[h] else f"{v:.2f}"))
            excl = " / ".join(str(int(r[f"excl{h}"])) for h in HORIZONS) if m == "linear" else ""
            L.append(f"| {s if m == 'linear' else ''} | {METHOD_LABEL[m]} | {' | '.join(cells)} | {excl} |")
    total_excl = {h: int(mt[mt["method"] == "linear"][f"excl{h}"].sum()) for h in HORIZONS}
    total_n = {h: int(mt[mt["method"] == "linear"][f"n{h}"].sum()) for h in HORIZONS}
    L += ["", "Усього виключено через втручання (лінійний): " + ", ".join(
        f"+{h} год — {total_excl[h]} з {total_excl[h] + total_n[h]}" for h in HORIZONS) + ". "
          "«—» — у цій ситуації всі такі горизонти перетинають втручання (напр. S2: щотижневе прибирання "
          "в межах 7 днів завжди).", "",
          "![MAE за горизонтами](forecast/mae_horizons.png)", "",
          "![Ситуація × метод](forecast/mae_heatmap.png)", ""]

    # --- Інтервал ---
    L += ["## Чесність інтервалу 10–90 % (лінійний метод)", "",
          "Частка фактичних значень усередині інтервалу; для чесного інтервалу — близько 80 %.", "",
          "| Ситуація | +1 год | +24 год | +7 днів |", "|---|---|---|---|"]
    for _, r in cov.iterrows():
        L.append(f"| {r['situation']} | " + " | ".join(
            "—" if pd.isna(r[f"cov{h}"]) else f"{r[f'cov{h}']:.0%}" for h in HORIZONS) + " |")
    L += [""]

    # --- Хибні тривоги, час роботи ---
    L += ["## Хибні тривоги і час роботи", "",
          "![Хибні тривоги](forecast/false_alarms.png)", "",
          "![Час роботи](forecast/runtime.png)", "",
          "| Метод | Середній час одного прогнозу, мс |", "|---|---|"]
    for m in METHODS:
        L.append(f"| {METHOD_LABEL[m]} | {bt[bt['method'] == m]['ms'].mean():.1f} |")
    L += [""]

    # --- Кращий метод ---
    L += ["## Ситуація → кращий метод (найменший MAE)", "",
          "| Ситуація | +1 год | +24 год | +7 днів |", "|---|---|---|---|"]
    for s in sorted(mt["situation"].unique()):
        g = mt[mt["situation"] == s]
        cells = []
        for h in HORIZONS:
            gg = g.dropna(subset=[f"mae{h}"])
            if gg.empty:
                cells.append("—")
            else:
                b = gg.loc[gg[f"mae{h}"].idxmin()]
                cells.append(f"{METHOD_LABEL[b['method']]} ({b[f'mae{h}']:.2f})")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += [""]

    # --- Історія ---
    hb = bt_all[bt_all["source"] == "history"]
    hr = runs_all[runs_all["source"] == "history"]
    if not hb.empty:
        hoc = run_outcomes(hb, hr, thr)
        hl = hoc[hoc["method"] == "linear"].set_index("key")
        L += ["## 30-денна історія (oncall.db) — змішаний приклад", "",
              f"| Інцидент | Диск | Ситуація | Дійшов до {thr} % | Попередив | За скільки | "
              "Хибних тривог (епізодів) | «Не поміститься» перед бекапом |", "|---|---|---|---|---|---|---|---|"]
        for key, r in hr.iterrows():
            inc = json.loads(r["info"])["incident"]
            if key not in hl.index:
                L.append(f"| #{inc} | {r['run']} | {r['situation']} {SITUATIONS[r['situation']]} | "
                         "не оцінено — перший тиждень історії (розігрів) | | | | |")
                continue
            o = hl.loc[key]
            nf = ("—" if pd.isna(r["jump_at"]) else
                  ("так, за " + fmt_h(o.get("jump_lead_h")) if o.get("jump_said_not_fit") else "ні"))
            L.append(f"| #{inc} | {r['run']} | {r['situation']} {SITUATIONS[r['situation']]} | "
                     f"{'так' if o['reached'] else 'ні'} | "
                     f"{'—' if not o['reached'] else ('ні (пропуск)' if o['miss'] else 'так')} | "
                     f"{fmt_h(o['lead_h'])} | {int(o['false_episodes'])} | {nf} |")
        hmt = mae_table(hb)
        L += ["", "MAE на історії (усі диски, до втручання):", "",
              "| Ситуація | Метод | +1 год | +24 год | +7 днів |", "|---|---|---|---|---|"]
        for s in sorted(hmt["situation"].unique()):
            for m in METHODS:
                r = hmt[(hmt["situation"] == s) & (hmt["method"] == m)].iloc[0]
                L.append(f"| {s if m == 'linear' else ''} | {METHOD_LABEL[m]} | " + " | ".join(
                    "—" if pd.isna(r[f"mae{h}"]) else f"{r[f'mae{h}']:.2f}" for h in HORIZONS) + " |")
        L += ["", "### Як змінювався прогноз", "",
              "![#19](forecast/evolution_19.png)", "", "![#2](forecast/evolution_2.png)", ""]

    # --- Галерея ---
    L += ["## Галерея: випадкові прогони кожної ситуації", "",
          "Відтворити будь-який: `python -m core.forecast_bench --plot-run <номер> --seed "
          f"{seed} [--at \"<момент>\"]`.", ""]
    rng = np.random.default_rng(seed)
    for s in sorted(SITUATIONS):
        ids = sorted(runs[runs["situation"] == s]["run"])
        for run_id in rng.choice(ids, size=min(2, len(ids)), replace=False):
            png = plt_dir / f"gallery_{run_id}.png"
            plot_bench_run(run_id, seed, png=png, rows=bt[bt["run"] == run_id])
            L += [f"**{run_id}** — {SITUATIONS[s]}", "", f"![{run_id}](forecast/{png.name})", ""]

    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Звіт: {REPORT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Перевірка прогнозу диска")
    parser.add_argument("--plot", nargs=3, metavar=("СЕРВЕР", "ДИСК", "МОМЕНТ"),
                        help='PNG моменту історії, напр.: sim-2 E: "2026-11-10 06:00" (UTC)')
    parser.add_argument("--run", action="store_true", help="сценарний стенд S1–S8")
    parser.add_argument("--seed", type=int, default=2, help="seed стенду (1 — налаштування, 2 — звіт)")
    parser.add_argument("--runs", type=int, default=RUNS_PER_SITUATION, help="прогонів на ситуацію")
    parser.add_argument("--workers", type=int, default=WORKERS, help="процесів паралельно")
    parser.add_argument("--history", action="store_true", help="перевірка на 30-денній історії")
    parser.add_argument("--summary", action="store_true", help="коротко: підсумок у консоль")
    parser.add_argument("--report", action="store_true", help="звіт і графіки з backtest.csv")
    parser.add_argument("--plot-run", metavar="НОМЕР", help="PNG прогону стенду, напр. S4-03")
    parser.add_argument("--at", help="момент для --plot-run (UTC); за замовчуванням — показовий")
    args = parser.parse_args()
    if args.plot:
        print(f"Збережено: {plot_history_moment(*args.plot)}")
    elif args.plot_run:
        print(f"Збережено: {plot_bench_run(args.plot_run, args.seed, args.at)}")
    elif args.run:
        run_bench(args.seed, args.runs, args.workers)
    elif args.history:
        run_history()
    elif args.summary:
        print_summary(args.seed)
    elif args.report:
        for inc_id, title in ((19, "#19 sim-2 E: — зламане видалення бекапів"),
                              (2, "#2 sim-1 D: — немає щотижневого прибирання логів")):
            plot_incident_evolution(inc_id, title, OUT_DIR / f"evolution_{inc_id}.png")
        build_report(args.seed)
    else:
        parser.print_help()


def print_summary(seed: int) -> None:
    """Короткий підсумок у консоль (для налаштування порогів на seed 1)."""
    thr = fc.load_config()["threshold_pct"]
    bt, runs = load_results()
    bt = bt[(bt["source"] == "bench") & (bt["seed"] == seed)]
    runs = runs[(runs["source"] == "bench") & (runs["seed"] == seed)]
    oc = run_outcomes(bt, runs, thr)
    lin = oc[oc["method"] == "linear"]
    print(f"seed {seed}: {runs['run'].nunique()} прогонів")
    for sit, g in lin.groupby("situation"):
        r = g[g["reached"]]
        print(f"  {sit}: дійшли до {thr} %: {len(r)}/{len(g)}; вчасно {int((~r['miss']).sum())}, "
              f"пропусків {int(r['miss'].sum())}; попередження сер. {r['lead_h'].mean():.1f} год, "
              f"мін {r['lead_h'].min():.1f}; з хибною тривогою {int((g['false_episodes'] > 0).sum())} "
              f"(епізодів {int(g['false_episodes'].sum())})"
              + (f"; стрибок: не поміститься сказав {int(g['jump_said_not_fit'].sum())}, "
                 f"правда «не поміститься» {int((~g['jump_truth_fits'].astype(bool)).sum())}"
                 if "jump_truth_fits" in g and g["jump_truth_fits"].notna().any() else ""))
    mt = mae_table(bt)
    print(mt.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
