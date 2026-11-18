"""Робота з базою даних SQLite.

Запуск:
    python -m core.db         створити базу (якщо її ще немає) і показати таблиці
    python -m core.db --er    додатково згенерувати ER-діаграму в docs/er.md

Повторний запуск безпечний: таблиці створюються лише якщо їх ще немає,
наявні дані не видаляються.
"""

import argparse
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# Корінь проєкту — тека на рівень вище за core/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "oncall.db"
ER_PATH = PROJECT_ROOT / "docs" / "er.md"


# ---------------------------------------------------------------------------
# Схема бази. Відповідає розділу «Схема БД» у CLAUDE.md.
# Увесь час зберігається текстом у форматі ISO 8601 в UTC,
# наприклад "2026-11-18T10:00:00Z" (див. функцію now_utc нижче).
# ---------------------------------------------------------------------------
SCHEMA = """
-- ===== Сирі дані (пишуть агент, симулятор, імпорт CSV) =====

CREATE TABLE IF NOT EXISTS servers (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT NOT NULL UNIQUE,  -- наприклад sim-1, my-pc
    kind                 TEXT NOT NULL CHECK (kind IN ('real', 'sim')),
    description          TEXT,
    -- Інвентар сервера: сайт -> пул IIS (appcmd list app) і служба -> exe
    -- (Win32_Service). Лише поточний стан, без історії
    inventory_json       TEXT,
    inventory_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    ts              TEXT NOT NULL,
    server_id       INTEGER NOT NULL REFERENCES servers(id),
    cpu_pct         REAL,
    ram_pct         REAL,
    disk_read_mbs   REAL,
    disk_write_mbs  REAL,
    net_in_mbs      REAL,
    net_out_mbs     REAL,
    app_response_ms REAL,
    app_error_pct   REAL,
    source          TEXT                        -- agent / simulator / csv
);

CREATE TABLE IF NOT EXISTS disk_usage (
    ts        TEXT NOT NULL,
    server_id INTEGER NOT NULL REFERENCES servers(id),
    drive     TEXT NOT NULL,                    -- наприклад C:
    used_pct  REAL,
    free_gb   REAL,
    total_gb  REAL
);

CREATE TABLE IF NOT EXISTS process_snapshots (
    ts        TEXT NOT NULL,
    server_id INTEGER NOT NULL REFERENCES servers(id),
    name      TEXT NOT NULL,
    pid       INTEGER,
    cpu_pct   REAL,
    ram_mb    REAL,
    app_pool  TEXT                              -- для w3wp.exe: назва пулу IIS
);

CREATE TABLE IF NOT EXISTS folder_sizes (
    ts        TEXT NOT NULL,
    server_id INTEGER NOT NULL REFERENCES servers(id),
    path      TEXT NOT NULL,
    size_gb   REAL
);

CREATE TABLE IF NOT EXISTS service_status (
    ts        TEXT NOT NULL,
    server_id INTEGER NOT NULL REFERENCES servers(id),
    name      TEXT NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('service', 'apppool')),
    status    TEXT NOT NULL                     -- running / stopped / ...
);

CREATE TABLE IF NOT EXISTS events (
    ts        TEXT NOT NULL,
    server_id INTEGER NOT NULL REFERENCES servers(id),
    log       TEXT,                             -- System / Application
    source    TEXT,                             -- наприклад WAS
    event_id  INTEGER,
    level     TEXT,
    message   TEXT
);

CREATE TABLE IF NOT EXISTS checks (
    ts         TEXT NOT NULL,
    server_id  INTEGER NOT NULL REFERENCES servers(id),
    check_type TEXT NOT NULL CHECK (check_type IN ('http', 'cert', 'backup', 'sql_port')),
    ok         INTEGER NOT NULL,                -- 1 = добре, 0 = проблема
    value      REAL,                            -- мс, дні до кінця сертифіката, год від бекапу
    details    TEXT
);

-- Пише симулятор: «правильні відповіді» для оцінки моделей
CREATE TABLE IF NOT EXISTS sim_incidents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id  INTEGER NOT NULL REFERENCES servers(id),
    scenario   TEXT NOT NULL,                   -- disk_fill / cpu_spike / ...
    started_at  TEXT NOT NULL,
    ended_at    TEXT,                           -- NULL, поки сценарій триває
    params_json TEXT                            -- параметри сценарію (служба, пул, тека...)
);

-- ===== Оброблені дані (пишуть ETL і аналізатор) =====

-- Одна метрика — один рядок. Ключ (server_id, metric, ts) не дає
-- записати одну й ту саму точку двічі, а ETL може її перезаписати.
CREATE TABLE IF NOT EXISTS metrics_clean (
    ts         TEXT NOT NULL,                   -- 1-хв сітка, UTC
    server_id  INTEGER NOT NULL REFERENCES servers(id),
    metric     TEXT NOT NULL,                   -- cpu_pct, disk_used_pct:C:, ...
    value      REAL,
    is_imputed INTEGER NOT NULL DEFAULT 0,      -- 1 = заповнено інтерполяцією
    is_outlier INTEGER NOT NULL DEFAULT 0,      -- 1 = статистичний викид
    PRIMARY KEY (server_id, metric, ts)
);

CREATE TABLE IF NOT EXISTS anomalies (
    ts         TEXT NOT NULL,
    server_id  INTEGER NOT NULL REFERENCES servers(id),
    metric     TEXT NOT NULL,
    score      REAL,
    is_anomaly INTEGER NOT NULL,
    model      TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    ts                 TEXT NOT NULL,
    server_id          INTEGER NOT NULL REFERENCES servers(id),
    metric             TEXT NOT NULL,
    model              TEXT,
    horizon_h          REAL,
    predicted_value    REAL,
    hours_to_threshold REAL                     -- NULL, якщо поріг не буде досягнуто
);

CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id     INTEGER NOT NULL REFERENCES servers(id),
    rule_id       TEXT NOT NULL,
    severity      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'acked', 'resolved')),
    opened_at     TEXT NOT NULL,
    updated_at    TEXT,
    notified_at   TEXT,
    acked_at      TEXT,
    resolved_at   TEXT,
    facts_json    TEXT,                         -- факти, на яких спрацювало правило
    justification TEXT,                         -- наприклад "hours_to_95 = 49 < 72"
    message       TEXT
);

-- ===== Користувачі та діалоги (пишуть бот і веб-чат) =====

CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_id INTEGER UNIQUE,                 -- NULL для користувачів лише веб-чату
    name        TEXT,
    notify      INTEGER NOT NULL DEFAULT 1      -- 1 = надсилати алерти
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    channel      TEXT NOT NULL CHECK (channel IN ('telegram', 'web')),
    ts           TEXT NOT NULL,
    role         TEXT NOT NULL,                 -- user / assistant
    content      TEXT,
    intent       TEXT,
    emotion      TEXT,
    input_type   TEXT CHECK (input_type IN ('text', 'voice', 'image', 'csv', 'xlsx', 'video')),
    latency_ms   INTEGER,
    llm_provider TEXT,
    tokens_in    INTEGER,
    tokens_out   INTEGER
);

CREATE TABLE IF NOT EXISTS uploads (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id),
    ts             TEXT NOT NULL,
    type           TEXT,
    filename       TEXT,
    status         TEXT,
    result_summary TEXT
);

-- ===== Якість моделей (пишуть скрипти оцінки) =====

CREATE TABLE IF NOT EXISTS model_metrics (
    ts      TEXT NOT NULL,
    model   TEXT NOT NULL,
    dataset TEXT,
    metric  TEXT NOT NULL,                      -- f1, mae, r2, faithfulness, ...
    value   REAL
);

-- ===== Індекси: пришвидшують вибірки «сервер + період» =====

CREATE INDEX IF NOT EXISTS ix_metrics_server_ts        ON metrics (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_disk_usage_server_ts     ON disk_usage (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_process_server_ts        ON process_snapshots (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_folder_sizes_server_ts   ON folder_sizes (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_service_status_server_ts ON service_status (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_events_server_ts         ON events (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_checks_server_ts         ON checks (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_anomalies_server_ts      ON anomalies (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_predictions_server_ts    ON predictions (server_id, ts);
CREATE INDEX IF NOT EXISTS ix_alerts_status            ON alerts (status);
CREATE INDEX IF NOT EXISTS ix_messages_user_ts         ON messages (user_id, ts);
"""


def now_utc() -> str:
    """Поточний час UTC у форматі, в якому він зберігається в базі."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Відкриває з'єднання з базою з потрібними налаштуваннями."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # timeout: якщо інший процес саме пише, чекаємо до 30 с, а не падаємо
    conn = sqlite3.connect(db_path, timeout=30)
    # Рядки можна читати за назвою колонки: row["cpu_pct"]
    conn.row_factory = sqlite3.Row
    # Перевірка зовнішніх ключів (у SQLite за замовчуванням вимкнена)
    conn.execute("PRAGMA foreign_keys = ON")
    # Режим WAL: симулятор, аналізатор і бот можуть працювати з базою одночасно
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def db_session(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Відкриває з'єднання на час одного блоку роботи і завжди його закриває.

    Використання:
        with db_session() as conn:
            conn.execute("INSERT ...")

    Якщо блок завершився без помилок — зміни зберігаються (commit).
    Якщо сталася помилка — зміни скасовуються (rollback), а помилка
    передається далі. В обох випадках з'єднання закривається, тому
    бот і аналізатор, що працюють годинами, не накопичують відкритих з'єднань.

    Увага: стандартне `with sqlite3.connect(...)` лише комітить,
    але НЕ закриває з'єднання, тому в проєкті використовуємо db_session().
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Створює всі таблиці та індекси, яких ще немає. Дані не чіпає."""
    with db_session(db_path) as conn:
        conn.executescript(SCHEMA)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    """Назви всіх наших таблиць (без службових таблиць SQLite)."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    ).fetchall()
    return [row["name"] for row in rows]


def build_er_diagram(conn: sqlite3.Connection) -> str:
    """Будує ER-діаграму Mermaid, читаючи схему з самої бази.

    Так діаграма завжди збігається з реальною схемою.
    """
    lines = ["erDiagram"]
    relations = []

    for table in list_tables(conn):
        # Зовнішні ключі таблиці: яка колонка на яку таблицю посилається
        fks = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        fk_columns = {fk["from"] for fk in fks}
        for fk in fks:
            # Одна батьківська таблиця — багато дочірніх рядків
            relations.append(f'    {fk["table"]} ||--o{{ {table} : "{fk["from"]}"')

        lines.append(f"    {table} {{")
        for col in conn.execute(f"PRAGMA table_info({table})").fetchall():
            col_type = col["type"] or "TEXT"
            keys = []
            if col["pk"]:
                keys.append("PK")
            if col["name"] in fk_columns:
                keys.append("FK")
            key_text = (" " + ", ".join(keys)) if keys else ""
            lines.append(f"        {col_type} {col['name']}{key_text}")
        lines.append("    }")

    return "\n".join(lines + relations)


def write_er_file(conn: sqlite3.Connection, path: Path = ER_PATH) -> None:
    """Записує ER-діаграму в docs/er.md."""
    text = (
        "# ER-діаграма бази даних\n\n"
        "Згенеровано автоматично з `data/oncall.db` командою "
        "`python -m core.db --er`. Не редагуйте вручну.\n\n"
        "```mermaid\n" + build_er_diagram(conn) + "\n```\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Створення бази і ER-діаграми")
    parser.add_argument("--er", action="store_true", help="згенерувати docs/er.md")
    args = parser.parse_args()

    init_db()
    print(f"База: {DB_PATH}")

    with db_session() as conn:
        tables = list_tables(conn)
        print(f"Таблиць: {len(tables)}")
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<20} рядків: {count}")

        if args.er:
            write_er_file(conn)
            print(f"ER-діаграму записано: {ER_PATH}")


if __name__ == "__main__":
    main()
