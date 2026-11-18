"""Тести ETL (core/etl.py). Запуск: python -m pytest

Усі тести працюють на тимчасовій базі (tmp_path від pytest), data/oncall.db не чіпають.
Штучний ряд і правильні відповіді — у core/etl.py (demo_raw, DEMO_EXPECT).
"""
import pandas as pd
import pytest

from core import etl
from core.db import db_session, init_db

E = etl.DEMO_EXPECT


@pytest.fixture
def demo(tmp_path):
    """Штучний ряд, прогнаний через ETL: (сирі записи, очищений ряд за хвилиною)."""
    raw, clean = etl.run_demo_db(tmp_path / "test.db")
    return raw, clean.set_index("minute")


def test_seconds_dropped_no_fake_gaps(demo):
    _, clean = demo
    # Усі хвилини, крім довгого пропуску, на місці; час — рівно на початку хвилини
    assert sorted(clean.index) == [m for m in range(etl.DEMO_MINUTES) if m not in E["long_gap"]]
    assert all(ts.endswith(":00Z") for ts in clean["ts"])


def test_impossible_value_dropped(demo):
    _, clean = demo
    m = E["impossible"][0]
    assert clean.loc[m, "value"] <= 100          # 130 % не потрапило
    assert clean.loc[m, "is_imputed"] == 1       # хвилину заповнено з сусідів


def test_short_gap_filled(demo):
    _, clean = demo
    m = E["short_gap"][0]
    assert clean.loc[m, "is_imputed"] == 1
    low, high = sorted([clean.loc[m - 1, "value"], clean.loc[m + 1, "value"]])
    assert low <= clean.loc[m, "value"] <= high  # значення між сусідами


def test_long_gap_kept(demo):
    _, clean = demo
    assert not any(m in clean.index for m in E["long_gap"])


def test_spike_is_outlier_and_kept(demo):
    _, clean = demo
    spike = clean.loc[E["spike"]]
    assert (spike["is_outlier"] == 1).all()
    assert (spike["value"] > 80).all()           # значення не змінено


def test_step_and_growth_are_not_outliers(demo):
    _, clean = demo
    assert (clean.loc[E["step"], "is_outlier"] == 0).all()
    assert (clean.loc[E["growth"], "is_outlier"] == 0).all()


def test_only_spike_is_outlier(demo):
    _, clean = demo
    assert sorted(clean.index[clean["is_outlier"] == 1]) == E["spike"]


def test_disk_step_and_jump_are_not_outliers_negative_dropped():
    # Диск: 93 % -> прибирання -> 56 %; потім різкий стрибок до 90 % на 3 хв
    # (великий тимчасовий файл) і назад; плюс одне від'ємне значення
    idx = pd.date_range("2026-01-05T10:00:00Z", periods=120, freq="min")
    values = pd.Series([93.0] * 60 + [56.0] * 60, index=idx)
    values.iloc[90:93] = 90.0
    values.iloc[30] = -5.0
    clean, stats = etl.clean_series(values, "disk_used_pct:D:")
    assert stats["dropped"] == 1
    assert clean["is_outlier"].sum() == 0
    assert clean["is_imputed"].sum() == 1
    assert (clean["value"].iloc[90:93] == 90.0).all()   # стрибок лишився як є


def test_rerun_does_not_duplicate(tmp_path):
    db = tmp_path / "test.db"
    etl.run_demo_db(db)
    etl.run_etl(db)                              # другий запуск без нових даних
    with db_session(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM metrics_clean").fetchone()[0]
    assert rows == etl.DEMO_MINUTES - len(E["long_gap"])


def test_new_data_appended_and_last_hour_recalculated(tmp_path):
    db = tmp_path / "test.db"
    etl.run_demo_db(db)
    with db_session(db) as conn:
        # До нового запуску останні точки сплеску ще не мають «після» —
        # перевіряємо, що дозапис 10 нових хвилин нічого не ламає
        sid = conn.execute("SELECT id FROM servers WHERE name = 'demo-1'").fetchone()[0]
        start = etl.DEMO_START + pd.Timedelta(minutes=etl.DEMO_MINUTES)
        conn.executemany("INSERT INTO metrics (ts, server_id, cpu_pct, source) VALUES (?, ?, ?, 'demo')",
                         [((start + pd.Timedelta(minutes=i)).strftime(etl.TS_FORMAT), sid, 77.0)
                          for i in range(10)])
    etl.run_etl(db)
    with db_session(db) as conn:
        rows = conn.execute("SELECT COUNT(*) FROM metrics_clean").fetchone()[0]
        outliers = conn.execute("SELECT COUNT(*) FROM metrics_clean WHERE is_outlier = 1").fetchone()[0]
    assert rows == etl.DEMO_MINUTES + 10 - len(E["long_gap"])
    assert outliers == len(E["spike"])


def test_spike_at_tail_marked_only_when_after_data_exists(tmp_path):
    """Живий режим: сплеск у самому кінці ще не викид; коли прийшли дані «після» — викид."""
    db = tmp_path / "test.db"
    init_db(db)
    t0 = pd.Timestamp("2026-01-05T10:00:00Z")

    def add(conn, sid, minutes, value):
        conn.executemany("INSERT INTO metrics (ts, server_id, cpu_pct, source) VALUES (?, ?, ?, 'demo')",
                         [((t0 + pd.Timedelta(minutes=m)).strftime(etl.TS_FORMAT), sid,
                           value + (m % 3) * 0.5) for m in minutes])

    with db_session(db) as conn:
        conn.execute("INSERT INTO servers (name, kind) VALUES ('live-1', 'sim')")
        sid = conn.execute("SELECT id FROM servers WHERE name = 'live-1'").fetchone()[0]
        add(conn, sid, range(0, 60), 30)
        add(conn, sid, range(60, 63), 90)          # сплеск в останні 3 хв
    etl.run_etl(db)
    with db_session(db) as conn:
        first = conn.execute("SELECT COUNT(*) FROM metrics_clean WHERE is_outlier = 1").fetchone()[0]
        add(conn, sid, range(63, 100), 30)         # прийшли дані «після»
    etl.run_etl(db)
    with db_session(db) as conn:
        second = conn.execute("SELECT COUNT(*) FROM metrics_clean WHERE is_outlier = 1").fetchone()[0]
    assert first == 0
    assert second == 3


# ---------------------------------------------------------------------------
# Допоміжне для тестів нижче: сервер і записи в тимчасовій базі
# ---------------------------------------------------------------------------

def add_server(conn, name):
    conn.execute("INSERT INTO servers (name, kind) VALUES (?, 'sim')", (name,))
    return conn.execute("SELECT id FROM servers WHERE name = ?", (name,)).fetchone()[0]


def add_records(conn, sid, records):
    """records: [(час pd.Timestamp, cpu, used_pct диска C:)]; None — значення немає."""
    conn.executemany("INSERT INTO metrics (ts, server_id, cpu_pct, source) VALUES (?, ?, ?, 'test')",
                     [(t.strftime(etl.TS_FORMAT), sid, cpu) for t, cpu, _ in records])
    conn.executemany("INSERT INTO disk_usage (ts, server_id, drive, used_pct, free_gb, total_gb) "
                     "VALUES (?, ?, 'C:', ?, ?, 100)",
                     [(t.strftime(etl.TS_FORMAT), sid, disk, 100 - disk)
                      for t, _, disk in records if disk is not None])


def read_clean(db, server=None):
    with db_session(db) as conn:
        q = ("SELECT s.name, mc.ts, mc.metric, mc.value, mc.is_imputed, mc.is_outlier "
             "FROM metrics_clean mc JOIN servers s ON s.id = mc.server_id")
        params = ()
        if server:
            q += " WHERE s.name = ?"
            params = (server,)
        return [tuple(r) for r in conn.execute(q + " ORDER BY s.name, mc.metric, mc.ts", params)]


T0 = pd.Timestamp("2026-01-05T10:00:00Z")


def test_two_records_in_one_minute_averaged(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    with db_session(db) as conn:
        sid = add_server(conn, "s-1")
        records = [(T0 + pd.Timedelta(minutes=m, seconds=2), 30.0, None) for m in range(10)]
        # хвилина 5: два записи (12:05:02 і 12:05:31) — 40 і 60
        records[5] = (T0 + pd.Timedelta(minutes=5, seconds=2), 40.0, None)
        records.append((T0 + pd.Timedelta(minutes=5, seconds=31), 60.0, None))
        add_records(conn, sid, records)
    etl.run_etl(db)
    rows = {r[1]: r[3] for r in read_clean(db) if r[2] == "cpu_pct"}
    assert len(rows) == 10                                   # 10 хвилин, не 11
    assert rows[(T0 + pd.Timedelta(minutes=5)).strftime(etl.TS_FORMAT)] == 50.0


def test_two_servers_not_mixed(tmp_path):
    db = tmp_path / "test.db"
    init_db(db)
    with db_session(db) as conn:
        a = add_server(conn, "s-a")
        b = add_server(conn, "s-b")
        # s-a: CPU 20, без пропусків; s-b: CPU 80, пропуск 1 хв на хвилині 5
        add_records(conn, a, [(T0 + pd.Timedelta(minutes=m), 20.0, 40.0) for m in range(10)])
        add_records(conn, b, [(T0 + pd.Timedelta(minutes=m), 80.0, 70.0) for m in range(10) if m != 5])
    etl.run_etl(db)
    rows_a, rows_b = read_clean(db, "s-a"), read_clean(db, "s-b")
    assert {r[3] for r in rows_a if r[2] == "cpu_pct"} == {20.0}
    assert {r[3] for r in rows_b if r[2] == "cpu_pct"} == {80.0}   # пропуск заповнено своїм 80, не чужим 20
    assert {r[3] for r in rows_a if r[2] == "disk_used_pct:C:"} == {40.0}
    assert {r[3] for r in rows_b if r[2] == "disk_used_pct:C:"} == {70.0}
    assert sum(r[4] for r in rows_a) == 0
    assert sum(r[4] for r in rows_b) == 3        # хвилина 5: cpu, disk_used_pct і disk_free_gb


def mixed_records():
    """6 год даних з усім одразу: шум, сплески, сходинка, пропуски 1/2/3/12 хв,
    два записи в одну хвилину, неможливе значення, диск з ростом і прибиранням."""
    import numpy as np
    rng = np.random.default_rng(11)
    records = []
    for m in range(360):
        if m in (50, 120, 121, 170, 171, 172) or 250 <= m < 262:
            continue                                             # пропуски
        cpu = 30 + rng.uniform(-2, 2)
        if 100 <= m < 103 or 200 <= m < 205:
            cpu = 90 + rng.uniform(-2, 2)                        # сплески
        if m >= 150:
            cpu += 20                                            # сходинка
        if m == 80:
            cpu = 150.0                                          # неможливе
        disk = 60 + m * 0.05 if m < 300 else 45 + rng.uniform(0, 0.1)   # ріст і прибирання
        t = T0 + pd.Timedelta(minutes=m, seconds=float(rng.uniform(0, 5)))
        records.append((t, cpu, disk))
        if m == 60:
            records.append((t + pd.Timedelta(seconds=30), cpu + 10, disk))   # другий запис у хвилині
    return records


def test_chunked_equals_full_rebuild(tmp_path):
    """Живий режим: дані приходять частинами, ETL запускається після кожної.
    Результат має точно збігатися з одним запуском на всіх даних."""
    records = mixed_records()

    full_db = tmp_path / "full.db"
    init_db(full_db)
    with db_session(full_db) as conn:
        add_records(conn, add_server(conn, "s-1"), records)
    etl.run_etl(full_db)

    live_db = tmp_path / "live.db"
    init_db(live_db)
    with db_session(live_db) as conn:
        sid = add_server(conn, "s-1")
    # Частини різного розміру: 1 хв (як живий потік), 7, 13, 25 хв
    sizes, i, k = [1, 7, 13, 25], 0, 0
    while i < len(records):
        part = records[i:i + sizes[k % len(sizes)]]
        with db_session(live_db) as conn:
            add_records(conn, sid, part)
        etl.run_etl(live_db)
        i += len(part)
        k += 1

    full, live = read_clean(full_db), read_clean(live_db)
    assert len(full) == len(live)
    assert full == live
    # і в даних справді є що перевіряти
    assert sum(r[4] for r in full) > 0 and sum(r[5] for r in full) > 0
