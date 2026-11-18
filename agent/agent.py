"""Агент збору даних з Windows-ПК або сервера.

Зараз (крок 1.4) — пробний режим: один раз зібрати дані і надрукувати.
    python -m agent.agent --print

Пізніше (крок 4.4) агент працюватиме постійно і надсилатиме дані
в REST API. Функції збору нижче використовуватимуться і там.

Формат даних збігається з таблицями сирих даних у core/db.py:
metrics, disk_usage, process_snapshots.
"""

import argparse
import time

import psutil

from core.db import db_session, init_db, now_utc

# Назва сервера в базі. Справжнє ім'я ПК не використовуємо:
# репозиторій публічний. На кроці 4.4 назва переїде в agent/config.yaml.
SERVER_NAME = "my-pc"

# Пауза між двома замірами, с. psutil рахує CPU і швидкість дисків/мережі
# як різницю між двома замірами, тому перший замір лише «запускає лічильник».
SAMPLE_INTERVAL_SEC = 2.0

TOP_PROCESSES = 5
MB = 1024 * 1024
GB = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def _net_totals() -> tuple[int, int]:
    """Сумарно отримано і надіслано байтів усіма адаптерами, крім loopback."""
    received = sent = 0
    for name, counters in psutil.net_io_counters(pernic=True).items():
        if "loopback" in name.lower():
            continue
        received += counters.bytes_recv
        sent += counters.bytes_sent
    return received, sent


def _disk_io_totals() -> tuple[int, int] | None:
    """Сумарно прочитано і записано байтів усіма дисками.

    None, якщо Windows не віддає лічильники дисків (буває, коли їх вимкнено).
    """
    counters = psutil.disk_io_counters()
    if counters is None:
        return None
    return counters.read_bytes, counters.write_bytes


def _app_pool_name(proc: psutil.Process) -> str | None:
    """Назва пулу IIS для w3wp.exe: у командному рядку вона після ключа -ap."""
    if (proc.info.get("name") or "").lower() != "w3wp.exe":
        return None
    try:
        args = proc.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return None  # без прав адміністратора командний рядок може бути недоступний
    if "-ap" in args:
        position = args.index("-ap")
        if position + 1 < len(args):
            return args[position + 1].strip('"')
    return None


# ---------------------------------------------------------------------------
# Збір даних
# ---------------------------------------------------------------------------

def collect_disk_usage(ts: str) -> list[dict]:
    """Зайнятість кожного диска з буквою (рядки для таблиці disk_usage)."""
    rows = []
    for part in psutil.disk_partitions(all=False):
        # Пропускаємо CD/DVD і пристрої без файлової системи (порожній кард-рідер)
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:
            continue  # диск недоступний (наприклад, витягнули флешку)
        rows.append({
            "ts": ts,
            "server_id": None,  # призначає база під час запису
            "drive": part.mountpoint.rstrip("\\"),  # "C:\" -> "C:"
            "used_pct": round(usage.percent, 1),
            "free_gb": round(usage.free / GB, 1),
            "total_gb": round(usage.total / GB, 1),
        })
    return rows


def collect_sample() -> dict[str, list[dict]]:
    """Один повний замір: два заміри лічильників з паузою між ними.

    Повертає словник: назва таблиці -> список рядків (словників з колонками).
    """
    cpu_count = psutil.cpu_count() or 1

    # --- Замір 1: «запускаємо лічильники» ---
    # Перший виклик cpu_percent() завжди повертає 0 — він лише запам'ятовує
    # стартову точку. Те саме для кожного процесу.
    psutil.cpu_percent(interval=None)
    processes = []
    for proc in psutil.process_iter(["name"]):
        try:
            proc.cpu_percent(interval=None)
            processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    disk_before = _disk_io_totals()
    net_before = _net_totals()
    started = time.monotonic()

    time.sleep(SAMPLE_INTERVAL_SEC)

    # --- Замір 2: читаємо різницю за паузу ---
    elapsed = time.monotonic() - started
    ts = now_utc()
    cpu_pct = psutil.cpu_percent(interval=None)
    disk_after = _disk_io_totals()
    net_after = _net_totals()

    if disk_before and disk_after:
        disk_read_mbs = (disk_after[0] - disk_before[0]) / elapsed / MB
        disk_write_mbs = (disk_after[1] - disk_before[1]) / elapsed / MB
    else:
        disk_read_mbs = disk_write_mbs = None

    metrics_row = {
        "ts": ts,
        "server_id": None,
        "cpu_pct": round(cpu_pct, 1),
        "ram_pct": round(psutil.virtual_memory().percent, 1),
        "disk_read_mbs": None if disk_read_mbs is None else round(disk_read_mbs, 2),
        "disk_write_mbs": None if disk_write_mbs is None else round(disk_write_mbs, 2),
        "net_in_mbs": round((net_after[0] - net_before[0]) / elapsed / MB, 2),
        "net_out_mbs": round((net_after[1] - net_before[1]) / elapsed / MB, 2),
        "app_response_ms": None,  # HTTP-перевірка сайту — крок 4.4
        "app_error_pct": None,    # частка помилок сайту — крок 4.4
        "source": "agent",
    }

    # --- Топ процесів за CPU ---
    process_rows = []
    for proc in processes:
        if proc.pid == 0:
            continue  # «System Idle Process» — це не навантаження, а простій
        try:
            # psutil рахує відносно одного ядра (на 8 ядрах можливо 800 %),
            # ділимо на кількість ядер — шкала 0–100 %, як у диспетчері завдань
            proc_cpu = proc.cpu_percent(interval=None) / cpu_count
            ram_mb = proc.memory_info().rss / MB
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        process_rows.append({
            "ts": ts,
            "server_id": None,
            "name": proc.info.get("name"),
            "pid": proc.pid,
            "cpu_pct": round(proc_cpu, 1),
            "ram_mb": round(ram_mb, 1),
            "app_pool": _app_pool_name(proc),
        })
    # Топ-5 за CPU і топ-5 за RAM, без повторів: так у знімок потрапляє
    # і процес, що вантажить процесор, і процес, у якого росте пам'ять
    top_cpu = sorted(process_rows, key=lambda row: row["cpu_pct"], reverse=True)[:TOP_PROCESSES]
    top_ram = sorted(process_rows, key=lambda row: row["ram_mb"], reverse=True)[:TOP_PROCESSES]
    top = top_cpu + [row for row in top_ram if row not in top_cpu]

    return {
        "metrics": [metrics_row],
        "disk_usage": collect_disk_usage(ts),
        "process_snapshots": top,
    }


# ---------------------------------------------------------------------------
# Пробний режим: друк і звірка зі схемою БД
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    return "—" if value is None else str(value)


def print_sample(sample: dict[str, list[dict]]) -> None:
    """Друкує зібрані дані у зручному для читання вигляді."""
    m = sample["metrics"][0]
    print(f"Сервер: {SERVER_NAME}   час (UTC): {m['ts']}   "
          f"ядер: {psutil.cpu_count()}   інтервал заміру: {SAMPLE_INTERVAL_SEC} с")

    print("\n[metrics]")
    print(f"  CPU                {_fmt(m['cpu_pct'])} %")
    print(f"  RAM                {_fmt(m['ram_pct'])} %")
    print(f"  Диски: читання     {_fmt(m['disk_read_mbs'])} МБ/с")
    print(f"  Диски: запис       {_fmt(m['disk_write_mbs'])} МБ/с")
    print(f"  Мережа: вхідний    {_fmt(m['net_in_mbs'])} МБ/с")
    print(f"  Мережа: вихідний   {_fmt(m['net_out_mbs'])} МБ/с")
    print(f"  Відповідь сайту    {_fmt(m['app_response_ms'])} мс (крок 4.4)")
    print(f"  Помилки сайту      {_fmt(m['app_error_pct'])} % (крок 4.4)")

    print("\n[disk_usage]")
    print(f"  {'Диск':<6}{'Зайнято, %':>12}{'Вільно, ГБ':>12}{'Всього, ГБ':>12}")
    for d in sample["disk_usage"]:
        print(f"  {d['drive']:<6}{d['used_pct']:>12}{d['free_gb']:>12}{d['total_gb']:>12}")

    print(f"\n[process_snapshots] топ-{TOP_PROCESSES} за CPU + топ-{TOP_PROCESSES} за RAM")
    print(f"  {'Процес':<28}{'PID':>8}{'CPU, %':>9}{'RAM, МБ':>10}  Пул IIS")
    for p in sample["process_snapshots"]:
        print(f"  {p['name'][:27]:<28}{p['pid']:>8}{p['cpu_pct']:>9}{p['ram_mb']:>10}  {_fmt(p['app_pool'])}")


def compare_with_schema(sample: dict[str, list[dict]]) -> bool:
    """Порівнює зібрані поля з колонками таблиць у базі. True — усе збігається."""
    init_db()  # якщо бази ще немає — створити
    all_ok = True
    print("\n[звірка зі схемою БД]")
    with db_session() as conn:
        for table, rows in sample.items():
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not rows:
                print(f"  {table:<18} немає даних для звірки")
                all_ok = False
                continue
            collected = set(rows[0])
            missing = columns - collected  # є в схемі, агент не збирає
            extra = collected - columns    # агент збирає, у схемі немає
            if not missing and not extra:
                print(f"  {table:<18} поля збігаються зі схемою")
            else:
                all_ok = False
                print(f"  {table:<18} РОЗБІЖНІСТЬ: немає в агента {sorted(missing)}, "
                      f"немає в схемі {sorted(extra)}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Агент збору даних")
    parser.add_argument("--print", action="store_true",
                        help="зібрати один раз і надрукувати, без запису в БД")
    args = parser.parse_args()

    if not args.print:
        print("Поки працює лише пробний режим: python -m agent.agent --print")
        return

    print(f"Збираю дані ({SAMPLE_INTERVAL_SEC} с)...")
    sample = collect_sample()
    print_sample(sample)
    compare_with_schema(sample)


if __name__ == "__main__":
    main()
