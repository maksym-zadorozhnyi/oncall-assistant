"""Тести функцій-інструментів (core/tools.py). Запуск: python -m pytest

Тимчасова база з даними, написаними вручну (data/oncall.db не чіпають).
Сервер t-1, дані 2 доби з 2026-01-05 00:00 UTC (= 02:00 місцевого, UTC+2).
Позначки нижче — хвилини від початку другої доби (D):
- CPU: 10 + (хв mod 7); D+60 — викид 95 %; D+30 — заповнена точка;
- D+100…D+109 — пропуск (немає жодних даних);
- диск C: 100 ГБ, рівний ріст 2 %/добу від 40 %;
- тека C:\\Logs росте 1 ГБ/добу, диск — 2 ГБ/добу (решта — «інше»);
- процеси — знімки кожні 5 хв лише до D+60;
- SitePool: стоп о D+50, знову працює о D+70; HTTP: помилка D+52…D+71;
- події: 1014 × 3 (шум), WAS 5009 о D+50, 7036 о D+70, звичайна інформаційна 100.
"""
import json

import pandas as pd
import pytest

from core import forecast as fc
from core import tools
from core.db import db_session, init_db

T0 = pd.Timestamp("2026-01-05T00:00:00Z")
D = 1440                                   # початок другої доби, хв
MINUTES = 2 * 1440
GAP = range(D + 100, D + 110)


def ts(minute: int, sec: int = 0) -> str:
    return (T0 + pd.Timedelta(minutes=minute, seconds=sec)).strftime(tools.TS_FORMAT)


def local(minute: int) -> str:
    """Той самий момент місцевим часом, як його вводить людина."""
    return (T0 + pd.Timedelta(minutes=minute)).tz_convert(tools.LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def disk_used(i: int) -> float:
    return round(40 + 2 * i / 1440, 3)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    init_db(path)
    monkeypatch.setattr(tools, "DB_PATH", path)
    inv = {"sites": [{"name": "portal.example.com", "app_pool": "SitePool"}],
           "services": [{"name": "SvcA", "exe": "a.exe"}, {"name": "W3SVC", "exe": "svchost.exe"},
                        {"name": "WAS", "exe": "svchost.exe"}]}
    with db_session(path) as conn:
        conn.execute("INSERT INTO servers (name, kind, description, inventory_json) VALUES (?, ?, ?, ?)",
                     ("t-1", "sim", "тестовий", json.dumps(inv)))
        rows = []
        for i in range(MINUTES):
            if i in GAP:
                continue
            cpu, imp, out = 10 + i % 7, 0, 0
            if i == D + 60:
                cpu, out = 95, 1
            if i == D + 30:
                imp = 1
            used = disk_used(i)
            rows += [(ts(i), 1, "cpu_pct", cpu, imp, out), (ts(i), 1, "ram_pct", 50, 0, 0),
                     (ts(i), 1, "disk_used_pct:C:", used, 0, 0),
                     (ts(i), 1, "disk_free_gb:C:", round(100 - used, 3), 0, 0)]
        conn.executemany("INSERT INTO metrics_clean VALUES (?, ?, ?, ?, ?, ?)", rows)
        conn.executemany("INSERT INTO disk_usage VALUES (?, 1, 'C:', ?, ?, 100)",
                         [(ts(i, 1), disk_used(i), 100 - disk_used(i)) for i in range(0, MINUTES, 60)])
        for i in range(0, D + 61, 5):            # знімки процесів (секунди, як у сирих даних)
            conn.executemany("INSERT INTO process_snapshots VALUES (?, 1, ?, ?, ?, ?, ?)", [
                (ts(i, 3), "a.exe", 11, 70.0, 300, None),
                (ts(i, 3), "svchost.exe", 12, 5.0, 900, None),
                (ts(i, 3), "w3wp.exe", 13, 1.0, 500, "SitePool")])
        for i in range(0, MINUTES, 5):           # тека C:\Logs росте 1 ГБ/добу
            conn.execute("INSERT INTO folder_sizes VALUES (?, 1, ?, ?)", (ts(i, 2), "C:\\Logs", 10 + i / 1440))
        for i in range(0, MINUTES, 5):           # служби
            pool = "stopped" if D + 50 <= i < D + 70 else "running"
            conn.executemany("INSERT INTO service_status VALUES (?, 1, ?, ?, ?)", [
                (ts(i, 4), "SvcA", "service", "running"), (ts(i, 4), "SitePool", "apppool", pool)])
        for i in range(0, MINUTES):              # HTTP щохвилини
            bad = D + 52 <= i < D + 72
            conn.execute("INSERT INTO checks VALUES (?, 1, 'http', ?, ?, ?)",
                         (ts(i, 5), 0 if bad else 1, 20 if bad else 150, "HTTP 503" if bad else "HTTP 200"))
        # бекап: одна перевірка о D+60 — вік 5 год (останній успішний — о D−240)
        conn.execute("INSERT INTO checks VALUES (?, 1, 'backup', 1, 5.0, ?)", (ts(D + 60), f"останній успішний: {ts(D - 240)}"))
        conn.executemany("INSERT INTO events VALUES (?, 1, ?, ?, ?, ?, ?)", [
            (ts(D + 10), "System", "DNS", 1014, "Warning", "Name resolution timed out"),
            (ts(D + 20), "System", "DNS", 1014, "Warning", "Name resolution timed out"),
            (ts(D + 40), "System", "DNS", 1014, "Warning", "Name resolution timed out"),
            (ts(D + 45), "Application", "App", 100, "Information", "щось звичайне"),
            (ts(D + 50), "System", "WAS", 5009, "Warning", "A process serving application pool 'SitePool' terminated unexpectedly."),
            (ts(D + 70), "System", "Service Control Manager", 7036, "Information", "The service entered the running state.")])
    return path


# ---------------------------------------------------------------------------

def test_local_time_to_utc():
    """Час без поясу — місцевий UTC+2; з поясом — як є."""
    assert tools.to_utc("2026-01-05 02:00") == T0
    assert tools.to_utc("2026-01-05T00:00:00Z") == T0
    assert tools.to_utc(None) is None
    assert tools.pair(T0) == {"ts_utc": "2026-01-05 00:00", "ts_local": "2026-01-05 02:00"}


def test_status_last_values_no_future(db):
    """Стан — останні значення до now; дані після now не потрапляють."""
    r = tools.get_status("t-1", now=local(D + 60))
    assert r["data_until"]["ts_local"] == local(D + 60)
    assert r["metrics"]["cpu_pct"] == 95                   # саме хвилина D+60, а не пізніша
    assert r["disks"][0]["used_pct"] == round(disk_used(D + 60), 1)
    assert r["disks"][0]["total_gb"] == 100
    assert r["stale"] is False
    pool = next(s for s in r["services"] if s["name"] == "SitePool")
    assert pool["status"] == "stopped" and pool["since_local"] == local(D + 50)
    assert r["not_running"] == ["SitePool"]
    assert r["checks"]["http"]["ok"] is False and r["checks"]["http"]["details"] == "HTTP 503"
    svc = next(s for s in r["services"] if s["name"] == "SvcA")
    assert svc["since_note"] == "стан не змінювався з початку даних"


def test_status_stale_and_live(db):
    """Після кінця даних сервер «замовк»; «наживо» — поточний час, дані — останні відомі."""
    r = tools.get_status("t-1", now=local(MINUTES + 29))
    assert r["stale"] is True and r["data_age_h"] == 0.5             # 30 хв
    fresh = tools.get_status("t-1", now=local(MINUTES - 1))
    assert fresh["stale"] is False and fresh["data_age_h"] == 0 and "stale_note" not in fresh
    live = tools.get_status("t-1")
    assert live["live"] is True and live["data_until"]["ts_local"] == local(MINUTES - 1)
    assert live["stale"] is True                                       # тестові дані — січень 2026


# --- Момент на 3 доби пізніше за дані: жодних нулів і жодного прогнозу «наче зараз» ---

LATE = MINUTES - 1 + 3 * 1440


def assert_stale(r: dict):
    """Однакові поля свіжості в усіх функціях (і у відповіді, і в помилці)."""
    assert r["stale"] is True and r["data_age_h"] == 72.0
    assert r["data_until"]["ts_local"] == local(MINUTES - 1)
    assert "немає свіжих даних" in r["stale_note"]


def test_late_status_marks_last_known(db):
    r = tools.get_status("t-1", now=local(LATE))
    assert_stale(r)
    assert r["checks"]["backup"]["age_h"] == 5.0                       # на час перевірки — відомо
    assert r["checks"]["backup"]["age_h_now"] is None                  # «на зараз» — невідомо, не вигадуємо
    assert r["metrics"]["cpu_pct"] is not None                         # останні відомі лишаються...
    assert "останні відомі" in r["notes"][0] and "3,0 дн тому" in r["notes"][0]   # ...з позначкою


def test_late_history_no_data(db):
    r = tools.get_metric_history("t-1", "cpu_pct", now=local(LATE))
    assert "немає даних" in r["error"] and f"останні — {local(MINUTES - 1)}" in r["error"]
    assert_stale(r)


def test_late_history_partial_window(db):
    """Дані закінчуються всередині вікна: пропуск у кінці і примітка; числа — лише за наявні дані."""
    r = tools.get_metric_history("t-1", "cpu_pct", now=local(MINUTES - 1 + 600))   # +10 год
    assert r["gaps"][-1]["edge"] == "end" and r["gaps"][-1]["missing_min"] == 600
    assert "дані закінчуються" in r["notes"][0]
    assert r["last_point_local"] == local(MINUTES - 1)


def test_late_folder_growth_no_zeros(db):
    r = tools.get_folder_growth("t-1", now=local(LATE))
    assert "немає даних про теки" in r["error"] and "folders" not in r
    assert_stale(r)


def test_late_folder_growth_partial(db):
    """Вікно 24 год, дані закінчились 10 год тому: ріст — за 14 год, з приміткою."""
    r = tools.get_folder_growth("t-1", now=local(MINUTES - 1 + 600))
    f = r["folders"][0]
    assert f["measured_h"] == pytest.approx(13.9, abs=0.2) and f["growth_gb"] > 0
    assert "дані закінчуються" in r["notes"][0]


def test_late_timeline_no_data(db):
    r = tools.get_timeline("t-1", now=local(LATE))
    assert "немає даних за період" in r["error"] and "items" not in r
    assert_stale(r)


def test_late_processes_no_snapshot(db):
    r = tools.get_top_processes("t-1", now=local(LATE))
    assert "немає знімка" in r["error"]
    assert_stale(r)


def test_late_forecast_not_for_now(db):
    """Прогнозу «на зараз» немає; є прогноз на момент останніх даних з явною позначкою."""
    r = tools.get_disk_forecast("t-1", now=local(LATE))
    assert_stale(r)
    assert "disks" not in r and "немає свіжих даних" in r["message"]
    last = r["forecast_at_last_data"]
    assert last["at"]["ts_local"] == local(MINUTES - 1) and "НЕ на зараз" in last["note"]
    fresh = tools.get_disk_forecast("t-1", now=local(MINUTES - 1))
    assert fresh["stale"] is False
    assert last["disks"][0]["trend"] == fresh["disks"][0]["trend"]    # той самий прогноз, лише з позначкою
    d = fresh["disks"][0]
    assert d["to_threshold_pp"] == round(90 - d["now_pct"], 1) and d["to_threshold_gb"] == pytest.approx(
        d["to_threshold_pp"], abs=0.2)                                   # диск 100 ГБ: 1 п. п. = 1 ГБ


def test_forecast_threshold_time(db):
    """Час досягнення порогу рахує код: now + hours_to_threshold, місцевим часом."""
    now = local(D + 60)
    tr = tools.get_disk_forecast("t-1", drive="C:", now=now)["disks"][0]["trend"]
    expected = (pd.Timestamp(now) + pd.Timedelta(hours=tr["hours_to_threshold"])).floor("min")
    # hours_to_threshold округлено до 0,1 год — допуск 6 хв
    got = pd.Timestamp(tr["threshold_at_local"])
    assert abs((got - expected).total_seconds()) <= 360


def test_jump_crosses_threshold():
    """Регулярний стрибок сам переводить диск через 90 %: коли і чи раніше за пряму тренду."""
    now = pd.Timestamp("2026-01-06T00:00:00Z")
    cfg = fc.load_config()
    jump = {"found": True, "at": now + pd.Timedelta(hours=2), "hours_until": 2.0, "size_gb": 10.0,
            "days_seen": 7, "free_at_gb": 15.0, "fits": True, "short_gb": 0.0,
            "free_to_stay_below_gb": 5.0, "level_after_pct": 95.0, "text": "поміститься"}

    def answer(hours):
        r = {"trend": {"status": "ok" if hours else "мало даних", "start": now, "points": 10,
                       "hours_to_threshold": hours, "speed_pct_day": 1.0, "speed_gb_day": 1.0,
                       "pred": {24: (86, 85, 87), 168: (92, 90, 94)}},
             "fast_growth": {"status": "мало даних"}, "jump": jump, "now_pct": 85.0, "now_free_gb": 15.0}
        return tools.disk_answer("C:", r, now, cfg, False, now - pd.Timedelta(days=14))

    early = answer(10.0)                     # пряма — через 10 год, стрибок — через 2 год
    assert early["trend"]["threshold_at_local"] == tools.loc(now + pd.Timedelta(hours=10))
    assert early["jump"]["crosses_threshold"] is True and early["jump"]["crosses_before_trend"] is True
    assert "раніше, ніж за прямою тренду" in early["notes"][0]
    late = answer(1.0)                       # пряма — раніше за стрибок
    assert late["jump"]["crosses_before_trend"] is False and "пізніше" in late["notes"][0]
    none = answer(None)                      # прямої немає — стрибок перший
    assert none["trend"]["threshold_at_local"] is None and none["jump"]["crosses_before_trend"] is True
    jump["level_after_pct"] = 88.0           # після стрибка нижче 90 % — не перетинає
    assert answer(10.0)["jump"]["crosses_threshold"] is False and answer(10.0)["notes"] == []


def test_folder_growth_drops(db):
    """Різке видалення всередині періоду: коли, скільки, і ріст після нього."""
    from core.db import db_session
    with db_session(db) as conn:                 # C:\Tmp: 20 ГБ +1 ГБ/год, о D+300 видалено 15 ГБ
        rows = []
        for i in range(D, D + 601, 60):
            size = 20 + (i - D) / 60 - (15 if i >= D + 300 else 0)
            rows.append((ts(i, 2), "C:\\Tmp", size))
        conn.executemany("INSERT INTO folder_sizes VALUES (?, 1, ?, ?)", rows)
    r = tools.get_folder_growth("t-1", period_h=10, now=local(D + 600))
    f = next(x for x in r["folders"] if x["path"] == "C:\\Tmp")
    assert f["growth_gb"] == -5.0                                  # «кінець − початок» ховає ріст
    assert f["drops"] == [{"at_local": local(D + 300), "freed_gb": 14.0,
                           "size_before_gb": 24.0, "size_after_gb": 10.0}] and f["drops_total"] == 1
    assert f["since_last_drop"] == {"from_local": local(D + 300), "growth_gb": 5.0, "measured_h": 5.0,
                                    "growth_gb_per_day": 24.0}
    logs = next(x for x in r["folders"] if x["path"] == "C:\\Logs")
    assert "drops" not in logs                                     # рівний ріст — без видалень


def test_history_gap_imputed_outliers(db):
    """Пропуск знаходиться; заповнені точки і викиди рахуються; після now — нічого."""
    r = tools.get_metric_history("t-1", "cpu_pct", start=local(D), end=local(D + 180), now=local(D + 150))
    assert r["to_local"] == local(D + 150) and r["last"] == 10 + (D + 150) % 7
    assert r["points"] == 150 - len(GAP)
    assert r["imputed"] == 1 and r["outliers"] == 1
    assert r["outlier_points"] == [[local(D + 60), 95]]
    assert r["max"] == {"value": 95, "ts_local": local(D + 60)}
    assert r["gaps"] == [{"from_local": local(D + 99), "to_local": local(D + 110), "missing_min": 10}]


def test_history_thinning(db):
    """Довгий ряд проріджується до max_points; короткий — без змін."""
    r = tools.get_metric_history("t-1", "cpu_pct", now=local(MINUTES - 1), max_points=20)
    assert len(r["series"]) <= 20 and r["step_min"] > 1
    r = tools.get_metric_history("t-1", "ram_pct", start=local(D), end=local(D + 30), max_points=50)
    assert len(r["series"]) == 30 and r["step_min"] == 1


def test_top_processes_nearest_snapshot(db):
    """Береться найближчий знімок не пізніше at; служба — лише якщо exe належить одній."""
    r = tools.get_top_processes("t-1", now=local(D + 63), by="cpu")
    assert r["snapshot_local"] == local(D + 60)
    assert r["cpu_server"] == 95                            # CPU сервера в ту хвилину
    first = r["processes"][0]
    assert first["name"] == "a.exe" and first["service"] == "SvcA"
    svchost = next(p for p in r["processes"] if p["name"] == "svchost.exe")
    assert svchost["service"] is None                       # svchost.exe — дві служби
    assert tools.get_top_processes("t-1", now=local(D + 63), by="ram")["processes"][0]["name"] == "svchost.exe"


def test_top_processes_no_snapshot(db):
    """Знімок давніший за 10 хв — зрозуміла помилка, не виняток."""
    r = tools.get_top_processes("t-1", now=local(D + 100))
    assert "немає знімка" in r["error"] and r["nearest_snapshot_local"] == local(D + 60)


def test_top_processes_at_past(db):
    """at — знімок у минулому (не пізніший за at); at пізніше «зараз» — помилка."""
    r = tools.get_top_processes("t-1", now=local(D + 200), at=local(D + 33))
    assert r["snapshot_local"] == local(D + 30)             # знімок D+35 — уже після at
    assert r["at_local"] == local(D + 33)
    assert r["processes"][0]["name"] == "a.exe"
    assert "at_local" not in tools.get_top_processes("t-1", now=local(D + 63))
    late = tools.get_top_processes("t-1", now=local(D + 40), at=local(D + 50))
    assert "пізніший за «зараз»" in late["error"]
    far = tools.get_top_processes("t-1", now=local(D + 200), at=local(D + 100))
    assert "немає знімка" in far["error"] and far["nearest_snapshot_local"] == local(D + 60)


def test_folder_growth_and_other(db):
    """Ріст теки і «інше» = ріст диска − сума тек."""
    r = tools.get_folder_growth("t-1", period_h=12, now=local(D + 720))
    f = r["folders"][0]
    assert f["path"] == "C:\\Logs" and f["growth_gb"] == 0.5 and f["growth_gb_per_day"] == 1.0
    d = r["disks"][0]                                       # диск: 2 %/добу = 1 ГБ за 12 год
    assert d["drive"] == "C:" and d["growth_gb"] == 1.0 and d["folders_gb"] == 0.5 and d["other_gb"] == 0.5


def test_timeline_changes_and_noise(db):
    """Хронологія: зміни стану знайдено, шум згорнуто, порядок за часом."""
    r = tools.get_timeline("t-1", start=local(D), end=local(D + 120), now=local(D + 120))
    kinds = [(x["ts_local"], x["type"]) for x in r["items"]]
    assert kinds == [(local(D + 50), "event"), (local(D + 50), "service"), (local(D + 52), "check"),
                     (local(D + 70), "event"), (local(D + 70), "service"), (local(D + 72), "check")]
    assert r["items"][1]["text"] == "SitePool (apppool): running -> stopped"
    assert r["noise"] == {"1014": 3}
    assert r["counts"] == {"event": 2, "service": 2, "check": 2, "alert": 0}
    assert all(x["code"] != 100 for x in r["items"])         # звичайна інформаційна — не показуємо


def test_timeline_change_at_window_start(db):
    """Зміна одразу на початку вікна видна завдяки останньому запису перед вікном."""
    r = tools.get_timeline("t-1", start=local(D + 47), end=local(D + 55), now=local(D + 55))
    assert any(x["type"] == "service" and x["ts_local"] == local(D + 50) for x in r["items"])


def test_timeline_filter_types(db):
    """types: лише вибрані типи; counts — після фільтра; список і рядок через кому — однаково."""
    args = dict(start=local(D), end=local(D + 120), now=local(D + 120))
    r = tools.get_timeline("t-1", types=["service"], **args)
    assert [x["type"] for x in r["items"]] == ["service", "service"]
    assert r["counts"] == {"event": 0, "service": 2, "check": 0, "alert": 0}
    assert r["noise"] == {}                                   # події не вибрано — шуму теж немає
    assert r["filters"] == {"types": ["service"], "source": None}
    r2 = tools.get_timeline("t-1", types="event,check", **args)
    assert {x["type"] for x in r2["items"]} == {"event", "check"} and r2["noise"] == {"1014": 3}
    assert "невідомий тип" in tools.get_timeline("t-1", types=["servis"], **args)["error"]
    assert "filters" not in tools.get_timeline("t-1", **args)  # без фільтрів — як раніше


def test_timeline_filter_source(db):
    """source: частковий збіг без регістру з джерелом або текстом (WAS 5009 згадує SitePool)."""
    r = tools.get_timeline("t-1", start=local(D), end=local(D + 120), now=local(D + 120), source="sitepool")
    got = [(x["type"], x["source"]) for x in r["items"]]
    assert got == [("event", "WAS"), ("service", "SitePool"), ("service", "SitePool")]
    assert r["noise"] == {}                                   # DNS 1014 не про SitePool
    r = tools.get_timeline("t-1", start=local(D), end=local(D + 120), now=local(D + 120),
                           types=["service"], source="SitePool")
    assert [x["text"] for x in r["items"]] == ["SitePool (apppool): running -> stopped",
                                               "SitePool (apppool): stopped -> running"]


def test_disk_forecast_same_as_forecast_module(db):
    """Обгортка дає ті самі числа, що й core/forecast.py (одне джерело чисел)."""
    r = tools.get_disk_forecast("t-1", drive="C:", now=local(MINUTES - 1), with_line=True)
    d = r["disks"][0]
    cfg = fc.load_config()
    now = T0 + pd.Timedelta(minutes=MINUTES - 1)
    with db_session(db) as conn:
        used, free = fc.load_disk(conn, 1, "C:", now, cfg)
    direct = fc.LinearMethod().predict(used, free, now, cfg)
    assert d["trend"]["pct_per_day"] == round(direct["trend"]["speed_pct_day"], 2)
    assert abs(d["trend"]["pct_per_day"] - 2) < 0.05
    assert d["trend"]["hours_to_threshold"] == round(direct["trend"]["hours_to_threshold"], 1)
    assert d["fast_growth"]["unusual"] is False and d["jump"]["found"] is False
    assert len(d["trend_line"]) > 10
    # звідки прогноз: погодинні точки — ті самі, що в forecast.trend; падінь не було — від початку історії
    assert d["trend"]["points"] == direct["trend"]["points"] and d["trend"]["points"] >= 47
    assert d["trend"]["from_reason"].startswith("початок історії")
    assert r["history_days"] == cfg["history_days"] == 14
    assert d["trend"]["min_points"] == cfg["trend"]["min_hours"]


def test_disk_forecast_too_little_data(db):
    """Перші години історії: тренд «мало даних», видно скільки точок є і скільки потрібно."""
    tr = tools.get_disk_forecast("t-1", drive="C:", now=local(180))["disks"][0]["trend"]
    assert tr["status"] == "мало даних" and tr["points"] < tr["min_points"]
    assert "pct_per_day" not in tr


def test_errors_are_readable(db):
    """Невідомий сервер, метрика, диск і момент до початку даних — словник з поясненням."""
    r = tools.get_status("sim-9")
    assert "невідомий сервер" in r["error"] and r["known_servers"] == ["t-1"]
    r = tools.get_status("t-1", now="2025-12-01 10:00")
    assert "немає даних" in r["error"]
    assert "known_metrics" in tools.get_metric_history("t-1", "cpu")
    assert "known_disks" in tools.get_disk_forecast("t-1", drive="Z:")
    r = tools.get_metric_history("t-1", "cpu_pct", start=local(D + 101), end=local(D + 108))
    assert "немає даних" in r["error"] and r["nearest_data_local"] == local(D + 99)
    assert "error" in tools.get_status("t-1", now="вчора")


def test_internal_error_does_not_crash(db, monkeypatch, caplog):
    """Будь-яка помилка всередині — словник «внутрішня помилка», виняток — у лог."""
    def boom(*args, **kwargs):
        raise RuntimeError("бум")
    monkeypatch.setattr(tools, "find_server", boom)
    r = tools.get_timeline("t-1")
    assert r == {"error": "внутрішня помилка: RuntimeError: бум"}
    assert "get_timeline" in caplog.text and "RuntimeError" in caplog.text


def test_parameter_checks(db):
    """max_points ≥ 2, top ≥ 1, period_h > 0, limit ≥ 1 — інакше зрозуміла помилка; «50» і 50.0 — можна."""
    assert "max_points" in tools.get_metric_history("t-1", "cpu_pct", max_points=1)["error"]
    assert "max_points" in tools.get_metric_history("t-1", "cpu_pct", max_points="багато")["error"]
    end = local(MINUTES - 1)
    assert "error" not in tools.get_metric_history("t-1", "cpu_pct", now=end, max_points="50")
    assert "error" not in tools.get_metric_history("t-1", "cpu_pct", now=end, max_points=50.0)
    assert "top" in tools.get_top_processes("t-1", now=local(D + 60), top=0)["error"]
    assert "period_h" in tools.get_folder_growth("t-1", period_h=0)["error"]
    assert "period_h" in tools.get_folder_growth("t-1", period_h=-5)["error"]
    assert "limit" in tools.get_timeline("t-1", limit=0)["error"]


def test_step_min_is_computed_step(db):
    """step_min — обчислений крок, навіть якщо між першими точками пропуск."""
    # вікно D+98…D+130: точки 99 і 110…130; крок 4 хв; відрізок 103–106 порожній
    r = tools.get_metric_history("t-1", "cpu_pct", start=local(D + 98), end=local(D + 130),
                                 now=local(D + 130), max_points=10)
    assert r["step_min"] == 4 and len(r["series"]) <= 10


def test_gaps_at_window_edges(db):
    """Пропуск на початку і в кінці вікна теж у gaps (edge: start / end)."""
    r = tools.get_metric_history("t-1", "cpu_pct", start=local(D + 105), end=local(D + 120), now=local(D + 120))
    assert r["gaps"][0] == {"from_local": local(D + 105), "to_local": local(D + 110),
                            "missing_min": 4, "edge": "start"}          # хвилини 106–109
    r = tools.get_metric_history("t-1", "cpu_pct", start=local(MINUTES - 30), now=local(MINUTES + 20))
    assert r["gaps"][-1] == {"from_local": local(MINUTES - 1), "to_local": local(MINUTES + 20),
                             "missing_min": 21, "edge": "end"}


def test_status_metric_missing_in_last_minute(db):
    """Якщо метрики немає саме в хвилину data_until — береться її попереднє значення."""
    with db_session(db) as conn:
        conn.execute("DELETE FROM metrics_clean WHERE metric = 'disk_used_pct:C:' AND ts = ?", (ts(D + 60),))
    r = tools.get_status("t-1", now=local(D + 60))
    assert r["disks"][0]["used_pct"] == round(disk_used(D + 59), 1)
    assert any("disk_used_pct:C:" in n for n in r["notes"])


def test_backup_age_now(db):
    """age_h — вік на час перевірки, age_h_now — на момент now."""
    b = tools.get_status("t-1", now=local(D + 90))["checks"]["backup"]
    assert b["age_h"] == 5.0 and b["age_h_now"] == 5.5
    assert b["details"] == f"останній успішний: {local(D - 240)} (місц.)"


def test_all_tools_registered():
    assert set(tools.TOOLS) == {"list_servers", "get_status", "get_metric_history", "get_top_processes",
                                "get_folder_growth", "get_disk_forecast", "get_timeline"}
