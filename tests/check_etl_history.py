"""Перевірка ETL на 30-денній історії симулятора -> docs/experiments/etl_report.md

Запуск: python -m tests.check_etl_history
Перебудовує metrics_clean з нуля (як core.etl --reset) і звіряє результат
з «ключем відповідей» sim_incidents. Читати sim_incidents тут можна: це скрипт
перевірки, а не аналізатор чи бот (див. «Принципи» в CLAUDE.md).
"""
import pandas as pd

from core import etl
from core.db import db_session

REPORT = etl.EXPERIMENTS_DIR / "etl_report.md"
SHORT_SPIKE_MIN = 15          # «короткий» cpu_spike — до 15 хв (див. крок 2.1)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def outliers_between(conn, server_id, metric_like, start, end) -> tuple[int, int]:
    """(скільки рядків, скільки з них викидів) у [start, end)."""
    r = conn.execute("SELECT COUNT(*), COALESCE(SUM(is_outlier), 0) FROM metrics_clean "
                     "WHERE server_id = ? AND metric LIKE ? AND ts >= ? AND ts < ?",
                     (server_id, metric_like, start.strftime(etl.TS_FORMAT),
                      end.strftime(etl.TS_FORMAT))).fetchone()
    return r[0], r[1]


def main() -> None:
    print("ETL з нуля на історії...")
    stats = etl.run_etl(reset=True)
    etl.print_stats(stats)

    lines = ["# ETL на історії симулятора", "",
             "Згенеровано `python -m tests.check_etl_history` (історія: "
             "`python -m simulator.simulator --history 30 --reset`, seed 42).", "",
             "## Підсумок", "",
             "| Сервер | Сирих записів | Рядків у metrics_clean | Відкинуто | Заповнено "
             "| Довгих пропусків | Викидів |", "|---|---|---|---|---|---|---|"]
    for name, st in stats.items():
        lines.append(f"| {name} | {st['records']} | {st['rows']} | {st['dropped']} | "
                     f"{st['imputed']} | {len(st['long_gaps'])} | {st['outliers']} |")
    lines += ["", "Відкинуто — неможливі значення; заповнено — рядки (показник × хвилина) "
              "з `is_imputed = 1`.", "", "**Відкинуті значення:**", ""]
    for name, st in stats.items():
        for t, metric in st["dropped_at"]:
            lines.append(f"- {name}: {t:%Y-%m-%d %H:%M} UTC, `{metric}`")
    lines += ["", "**Довгі пропуски (лишились пропусками):**", ""]
    for name, st in stats.items():
        for t, n in st["long_gaps"]:
            lines.append(f"- {name}: {t:%Y-%m-%d %H:%M} UTC, {n} хв")

    checks = []      # (опис, пройшло?)
    with db_session() as conn:
        servers = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM servers")}

        # --- Викиди за показниками ---
        lines += ["", "## Викиди за показниками", "",
                  "| Сервер | Показник | Викидів |", "|---|---|---|"]
        for r in conn.execute("SELECT server_id, metric, SUM(is_outlier) n FROM metrics_clean "
                              "GROUP BY server_id, metric HAVING n > 0 ORDER BY server_id, n DESC"):
            lines.append(f"| {servers[r['server_id']]} | `{r['metric']}` | {r['n']} |")
        # Зайнятість дисків (не плутати з disk_read_mbs / disk_write_mbs — це швидкість I/O)
        disk_out = conn.execute("SELECT COUNT(*) FROM metrics_clean WHERE (metric LIKE 'disk_used_pct:%' "
                                "OR metric LIKE 'disk_free_gb:%') AND is_outlier = 1").fetchone()[0]
        checks.append(("Диски: викидів немає (шукаємо лише неможливі значення)", disk_out == 0,
                       f"{disk_out} викидів"))

        # --- Ключ відповідей: cpu_spike ---
        lines += ["", "## Перевірка за ключем відповідей", "",
                  f"### cpu_spike (короткий — до {SHORT_SPIKE_MIN} хв)", "",
                  "| # | Сервер | Початок (UTC) | Тривалість | Хвилин позначено викидами |",
                  "|---|---|---|---|---|"]
        incidents = conn.execute("SELECT id, server_id, scenario, started_at, ended_at "
                                 "FROM sim_incidents ORDER BY id").fetchall()
        for inc in incidents:
            if inc["scenario"] != "cpu_spike":
                continue
            start, end = ts(inc["started_at"]), ts(inc["ended_at"])
            minutes = int((end - start).total_seconds() // 60)
            n, out = outliers_between(conn, inc["server_id"], "cpu_pct", start, end)
            share = out / n if n else 0
            lines.append(f"| {inc['id']} | {servers[inc['server_id']]} | {start:%Y-%m-%d %H:%M} | "
                         f"{minutes} хв | {out} з {n} ({share:.0%}) |")
            if minutes <= SHORT_SPIKE_MIN:
                checks.append((f"Короткий cpu_spike #{inc['id']} ({minutes} хв) позначено викидом",
                               share >= 0.8, f"{out} з {n}"))
        lines += ["", "Довші сплески (понад ~15 хв) — це тимчасовий новий рівень, а не викид; "
                  "їх побачить детектор аномалій (крок 4.2)."]

        # --- Решта викидів CPU: епізоди поза розміченими cpu_spike ---
        lines += ["", "### Решта викидів CPU (поза розміченими cpu_spike)", "",
                  "Симулятор навмисно додає фонові сплески CPU без мітки: 1–3 хв до 70–97 %, "
                  "у середньому 1,5 на добу на сервер (`BURST_PER_DAY`, випадково — тож "
                  "за 30 днів буває і 45, і 57). Мітки в «ключі відповідей» у них немає, тому "
                  "звіряємо з сирими даними: короткий (до 3 хв) підйом CPU до ≥ 70 % поза "
                  "`cpu_spike`. Епізод — хвилини-викиди поспіль.", "",
                  "| Сервер | Днів | Коротких підйомів ≥ 70 % у сирих даних | З них позначено | "
                  "Епізодів-сплесків вгору (1 / 2 / 3 хв / довше) | Провалів вниз |",
                  "|---|---|---|---|---|---|"]
        notes = []
        for sid, name in servers.items():
            cpu = pd.read_sql_query("SELECT ts, value, is_outlier FROM metrics_clean WHERE server_id = ? "
                                    "AND metric = 'cpu_pct' ORDER BY ts", conn, params=[sid])
            if cpu.empty:
                continue
            cpu["ts"] = pd.to_datetime(cpu["ts"], utc=True)
            cpu = cpu.set_index("ts")
            spikes = [(ts(i["started_at"]), ts(i["ended_at"])) for i in incidents
                      if i["server_id"] == sid and i["scenario"] == "cpu_spike"]
            others = [(ts(i["started_at"]), ts(i["ended_at"]), i["scenario"], i["id"]) for i in incidents
                      if i["server_id"] == sid and i["scenario"] != "cpu_spike"]

            def in_spike(t):
                return any(a <= t < b for a, b in spikes)

            def label(t):
                return next((f"{s} #{i}" for a, b, s, i in others if a <= t < b), None)

            # Короткі підйоми >= 70 % у даних (хвилини поспіль, не довше 3 хв)
            high = cpu["value"] >= 70
            run_id = (high != high.shift()).cumsum()
            bursts = [g.index for _, g in cpu[high].groupby(run_id[high])
                      if len(g) <= 3 and not in_spike(g.index[0])]
            marked = [b for b in bursts if cpu.loc[b, "is_outlier"].any()]
            for b in bursts:
                if not cpu.loc[b, "is_outlier"].any():
                    before = cpu.loc[b[0] - pd.Timedelta(minutes=30):b[0] - pd.Timedelta(minutes=1), "value"]
                    after_points = len(cpu.loc[b[-1] + pd.Timedelta(minutes=1):])
                    if after_points < etl.OUTLIER_MIN_POINTS:
                        why = (f"він у самому кінці історії, після нього лише {after_points} хв даних — "
                               f"рівень «після» ще невідомий (позначка з'явиться, коли прийдуть дані)")
                    else:
                        why = (f"рівень перед ним {before.median():.0f} %, "
                               f"різниця з рівнями «до» / «після» менша за поріг")
                    notes.append(f"- {name}: {b[0]:%Y-%m-%d %H:%M} UTC — сплеск до "
                                 f"{cpu.loc[b, 'value'].max():.0f} % не позначено: {why}")

            # Епізоди-викиди поза cpu_spike
            out = cpu[(cpu["is_outlier"] == 1) & ~cpu.index.to_series().apply(in_spike)]
            episode_id = (out.index.to_series().diff() != pd.Timedelta(minutes=1)).cumsum()
            ups = {1: 0, 2: 0, 3: 0, "довше": 0}
            downs = 0
            for _, ep in out.groupby(episode_id):
                t0 = ep.index[0]
                before = cpu.loc[t0 - pd.Timedelta(minutes=30):t0 - pd.Timedelta(minutes=1), "value"]
                if ep["value"].mean() > before.median():
                    ups[len(ep) if len(ep) <= 3 else "довше"] += 1
                else:
                    downs += 1
                    notes.append(f"- {name}: {t0:%Y-%m-%d %H:%M} UTC — провал CPU {len(ep)} хв "
                                 f"(рівень перед ним {before.median():.0f} % → {ep['value'].mean():.0f} %)"
                                 f"{', усередині ' + label(t0) if label(t0) else ''}")
            days = (cpu.index[-1] - cpu.index[0]).total_seconds() / 86400
            lines.append(f"| {name} | {days:.0f} | {len(bursts)} | {len(marked)} | "
                         f"{ups[1]} / {ups[2]} / {ups[3]} / {ups['довше']} | {downs} |")
            checks.append((f"{name}: сплески-викиди вгору поза cpu_spike — це фонові сплески "
                           f"1–3 хв, позначено ≥ 95 % з них",
                           ups["довше"] == 0 and sum(ups.values()) == len(marked)
                           and len(marked) >= 0.95 * len(bursts),
                           f"сплесків у даних {len(bursts)}, позначено {len(marked)}, "
                           f"епізодів вгору {sum(ups.values())}, довших за 3 хв {ups['довше']}"))
        if notes:
            lines += ["", "Окремо:", ""] + notes
            lines += ["", "Провал CPU під час падіння пулу — справжній інцидент (пул зупинено, його "
                      "частка CPU зникла); він на межі «короткого» (~15 хв), тому позначено "
                      "лише частину хвилин."]

        # --- Прибирання D: на sim-1: знаходимо з даних (різке падіння зайнятості) ---
        lines += ["", "### Падіння зайнятості D: на sim-1 — сходинка, не викид", "",
                  "Щотижневе прибирання логів IIS (неділя 01:00 UTC) і «прибирання» "
                  "наприкінці `disk_fill`; знайдено з даних (падіння > 1 п. п. за хвилину).", "",
                  "| Час прибирання (UTC) | Було → стало, % | Викидів ±1 год (усі показники) |",
                  "|---|---|---|"]
        sid1 = next(i for i, n in servers.items() if n == "sim-1")
        d = pd.read_sql_query("SELECT ts, value FROM metrics_clean WHERE server_id = ? "
                              "AND metric = 'disk_used_pct:D:' ORDER BY ts", conn, params=[sid1])
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
        drops = d[d["value"].diff() < -1]
        for _, row in drops.iterrows():
            prev = d.loc[row.name - 1, "value"]
            n, out = outliers_between(conn, sid1, "%", row["ts"] - pd.Timedelta(hours=1),
                                      row["ts"] + pd.Timedelta(hours=1))
            lines.append(f"| {row['ts']:%Y-%m-%d %H:%M} | {prev:.1f} → {row['value']:.1f} | {out} |")
            checks.append((f"Після прибирання D: {row['ts']:%d.%m %H:%M} викидів немає", out == 0,
                           f"{out} викидів"))

        # --- Падіння RAM наприкінці memory_leak ---
        lines += ["", "### Кінець memory_leak (процес перезапущено, RAM падає) — сходинка, не викид", "",
                  "| # | Сервер | Кінець (UTC) | RAM за 5 хв до → через 5 хв, % | Викидів RAM ±1 год |",
                  "|---|---|---|---|---|"]
        for inc in incidents:
            if inc["scenario"] != "memory_leak":
                continue
            end = ts(inc["ended_at"])
            ram = pd.read_sql_query("SELECT ts, value FROM metrics_clean WHERE server_id = ? "
                                    "AND metric = 'ram_pct' AND ts >= ? AND ts < ? ORDER BY ts", conn,
                                    params=[inc["server_id"],
                                            (end - pd.Timedelta(minutes=5)).strftime(etl.TS_FORMAT),
                                            (end + pd.Timedelta(minutes=6)).strftime(etl.TS_FORMAT)])
            n, out = outliers_between(conn, inc["server_id"], "ram_pct",
                                      end - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1))
            lines.append(f"| {inc['id']} | {servers[inc['server_id']]} | {end:%Y-%m-%d %H:%M} | "
                         f"{ram['value'].iloc[0]:.1f} → {ram['value'].iloc[-1]:.1f} | {out} |")
            checks.append((f"Після падіння RAM наприкінці memory_leak #{inc['id']} викидів немає",
                           out == 0, f"{out} викидів"))

    # --- Підсумок перевірок ---
    passed = sum(ok for _, ok, _ in checks)
    lines += ["", f"## Результат: {passed} з {len(checks)} перевірок пройдено", "",
              "| Перевірка | Результат | Деталі |", "|---|---|---|"]
    for text, ok, details in checks:
        lines.append(f"| {text} | {'✅ так' if ok else '❌ ні'} | {details} |")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nПеревірок пройдено: {passed} з {len(checks)}. Звіт: {REPORT}")
    for text, ok, details in checks:
        if not ok:
            print(f"  НЕ ПРОЙДЕНО: {text} ({details})")


if __name__ == "__main__":
    main()
