"""Тести сторінки дашборду (web/app.py) без браузера — через вбудований у Streamlit AppTest.
Тимчасова база; data/oncall.db не чіпають. Запуск: python -m pytest
"""
import streamlit as st
from streamlit.testing.v1 import AppTest

from core import tools
from core.db import PROJECT_ROOT, db_session, init_db

APP = str(PROJECT_ROOT / "web" / "app.py")


def run_app(path, monkeypatch) -> AppTest:
    monkeypatch.setattr(tools, "DB_PATH", path)
    st.cache_data.clear()                       # кеш спільний для всіх запусків — чистимо
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def texts(at: AppTest) -> str:
    return " ".join(e.value for e in list(at.warning) + list(at.error))


def test_no_clean_data_shows_hint(tmp_path, monkeypatch):
    """Сервер є, а metrics_clean порожня (після перегенерації) — підказка про ETL, а не падіння."""
    path = tmp_path / "t.db"
    init_db(path)
    with db_session(path) as conn:
        conn.execute("INSERT INTO servers (name, kind, description) VALUES ('t-1', 'sim', 'тест')")
        conn.execute("INSERT INTO metrics (ts, server_id, cpu_pct) VALUES ('2026-01-05T00:00:00Z', 1, 10)")
    at = run_app(path, monkeypatch)
    assert not at.exception
    assert "python -m core.etl" in texts(at)


def test_no_servers(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    init_db(path)
    at = run_app(path, monkeypatch)
    assert not at.exception
    assert "немає серверів" in texts(at)


def test_page_renders_with_data(tmp_path, monkeypatch):
    """Звичайний випадок: 2 доби очищених даних — сторінка малюється без помилок."""
    path = tmp_path / "t.db"
    init_db(path)
    with db_session(path) as conn:
        conn.execute("INSERT INTO servers (name, kind, description) VALUES ('t-1', 'sim', 'тест')")
        rows = []
        for i in range(2 * 1440):
            t = f"2026-01-0{5 + i // 1440}T{(i % 1440) // 60:02d}:{i % 60:02d}:00Z"
            used = 40 + 2 * i / 1440
            rows += [(t, 1, "cpu_pct", 10 + i % 7), (t, 1, "ram_pct", 50),
                     (t, 1, "disk_used_pct:C:", used), (t, 1, "disk_free_gb:C:", 100 - used)]
        conn.executemany("INSERT INTO metrics_clean (ts, server_id, metric, value) VALUES (?, ?, ?, ?)", rows)
    at = run_app(path, monkeypatch)
    assert not at.exception
    assert at.title[0].value == "t-1"
    assert [h.value for h in at.subheader][0] == "A. Стан"
