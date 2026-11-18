"""Тести прогнозу диска (core/forecast.py). Запуск: python -m pytest

Штучні ряди «зайнятість диска, %» з відомою відповіддю. Диск 100 ГБ, тож
1 % = 1 ГБ. Значення округлено до 0,1 %, як у справжніх даних.
Тест запису в predictions — на тимчасовій базі, data/oncall.db не чіпає.
"""
import json

import numpy as np
import pandas as pd

from core import forecast as fc
from core.db import db_session, init_db

CFG = fc.load_config()
T0 = pd.Timestamp("2026-01-05T00:00:00Z")      # понеділок, 00:00 UTC
TOTAL_GB = 100.0


def disk(level: np.ndarray, start=T0) -> tuple[pd.Series, pd.Series]:
    """Ряд «зайнято, %» (похвилинний) і відповідний ряд «вільно, ГБ»."""
    idx = pd.date_range(start, periods=len(level), freq="min")
    used = pd.Series(np.round(level, 1), index=idx)
    free = TOTAL_GB * (1 - used / 100)
    return used, free


def minutes(days: float) -> np.ndarray:
    return np.arange(int(days * 1440))


def predict(used, free, now=None):
    now = now or used.index[-1]
    return fc.LinearMethod().predict(used[:now], free[:now], now, CFG)


def daily_backup(level: np.ndarray, size: float, at_min: int, removed_after_min: int | None,
                 last_removed_day: int | None = None) -> np.ndarray:
    """Додає щоденний стрибок розміру size о хвилині доби at_min; через removed_after_min
    його видаляють (до дня last_removed_day включно; далі — не видаляють: «сходи»)."""
    level = level.copy()
    days = len(level) // 1440 + 1
    for d in range(days):
        s = d * 1440 + at_min
        if s >= len(level):
            break
        removed = removed_after_min is not None and (last_removed_day is None or d <= last_removed_day)
        e = s + removed_after_min if removed else len(level)
        level[s:e] += size
    return level


# ---------------------------------------------------------------------------

def test_steady_growth():
    """Рівний ріст 2 %/добу: правильна швидкість і час до 90 %."""
    m = minutes(14)
    used, free = disk(40 + 2 * m / 1440)
    r = predict(used, free)
    tr = r["trend"]
    assert abs(tr["speed_pct_day"] - 2) < 0.1
    expected_h = (90 - used.iloc[-1]) / 2 * 24              # (90 − 68) / 2 %/добу = 11 діб
    assert abs(tr["hours_to_threshold"] - expected_h) < 0.05 * expected_h
    assert abs(tr["pred"][24][0] - (used.iloc[-1] + 2)) < 0.3
    assert r["fast_growth"]["unusual"] is False
    assert r["jump"]["found"] is False


def test_cleanup_in_the_middle():
    """Прибирання посередині: тренд рахується лише після нього (1 %/добу, а не 3 %/добу)."""
    m = minutes(14)
    level = np.where(m < 7 * 1440, 30 + 3 * m / 1440, 20 + 1 * (m - 7 * 1440) / 1440)
    used, free = disk(level)
    tr = predict(used, free)["trend"]
    assert abs((tr["start"] - (T0 + pd.Timedelta(days=7))).total_seconds()) <= 30 * 60
    assert abs(tr["speed_pct_day"] - 1) < 0.1


def test_cleanup_stopped_staircase_warns():
    """Щоночі бекап +9 % о 00:30, видалення через 3,5 год; з 10-го дня видалення
    не працює (сходи). Звичайної ночі тривоги немає, після пропущених видалень —
    тренд ловить ріст і до 90 % менше 72 год."""
    m = minutes(13)
    level = daily_backup(np.full(len(m), 50.0), 9, 30, 210, last_removed_day=9)
    used, free = disk(level)

    normal = predict(used, free, T0 + pd.Timedelta(days=9, hours=1))     # день 9, після бекапу
    assert normal["trend"]["excluded"] is not None                       # бекап — тимчасовий
    assert abs(normal["trend"]["speed_pct_day"]) < 0.5
    assert normal["trend"]["hours_to_threshold"] is None

    broken = predict(used, free, T0 + pd.Timedelta(days=12, hours=6))
    assert broken["trend"]["speed_pct_day"] > 5
    assert broken["trend"]["hours_to_threshold"] < 72


def test_daily_jump_at_0200_is_not_alarm():
    """Щоденний стрибок о 02:00 (+9 %, видалення о 05:00): одразу після стрибка —
    не тривога; регулярний стрибок знайдено, наступний — завтра о ~02:00, поміститься."""
    m = minutes(14)
    level = daily_backup(np.full(len(m), 50.0), 9, 120, 180)
    used, free = disk(level)
    now = T0 + pd.Timedelta(days=13, hours=2, minutes=50)
    r = predict(used, free, now)
    assert r["fast_growth"]["unusual"] is False
    assert r["trend"]["hours_to_threshold"] is None
    j = r["jump"]
    assert j["found"] and j["fits"]
    expected = T0 + pd.Timedelta(days=14, hours=2)
    assert abs((j["at"] - expected).total_seconds()) <= 30 * 60
    assert abs(j["size_gb"] - 9) < 1


def test_fast_growth_alarm():
    """13 днів рівно 40 %, потім 20 %/год за останні 2 год: незвичний ріст, тривога."""
    m = minutes(13)
    level = np.full(len(m) + 120, 40.0)
    level[-120:] = 40 + 20 * np.arange(1, 121) / 60
    used, free = disk(level)
    fg = predict(used, free)["fast_growth"]
    assert fg["unusual"] is True
    assert abs(fg["speed_pct_h"] - 20) < 1
    assert fg["hours_to_threshold"] < 1


def test_jump_does_not_fit():
    """Щоденний бекап +20 % (о 02:00, видалення о 05:00) на рівні 60 %; останнього дня
    о 10:00 скопіювали великі файли (+25 %): вільно 15 ГБ, а бекапу треба 20 ГБ."""
    m = minutes(13.5)
    level = daily_backup(np.full(len(m), 60.0), 20, 120, 180)
    level[13 * 1440 + 600:] += 25
    used, free = disk(level)
    j = predict(used, free)["jump"]
    assert j["found"]
    assert j["fits"] is False
    assert j["short_gb"] >= 4                                   # фізично: 20 − 15 = 5 ГБ і більше
    assert j["free_to_stay_below_gb"] >= j["short_gb"] + 10 - 0.5   # нижче 90 % — ще +10 ГБ
    assert "не поміститься" in j["text"]


def test_staircase_jump_not_counted_twice():
    """«Сходи»: бекап +9 % щоночі без видалення. Тренд складається з самих бекапів,
    тому вільне місце до наступного бекапу не має зменшуватися ще й на тренд:
    фізично бракує рівно розмір бекапу мінус вільне зараз."""
    m = minutes(13)
    level = daily_backup(np.full(len(m), 40.0), 9, 30, 210, last_removed_day=8)
    used, free = disk(level)
    now = T0 + pd.Timedelta(days=12, hours=1)      # після 4-ї сходинки: 40 + 4·9 = 76 %
    j = predict(used, free, now)["jump"]
    assert abs(j["free_at_gb"] - j["free_now_gb"]) < 1.5
    assert j["fits"] is True                       # вільно 24 ГБ, бекап 9 ГБ
    assert j["above_threshold_after"] is False     # 76 + 9 = 85 % < 90 %


def test_after_step_one_hour_is_flat():
    """Одразу після щоденного стрибка +1 год — на рівні після сходинки, а не «росте далі»."""
    m = minutes(14)
    level = daily_backup(np.full(len(m), 50.0), 9, 120, 180)
    used, free = disk(level)
    now = T0 + pd.Timedelta(days=13, hours=2, minutes=40)     # через 40 хв після стрибка
    fg = predict(used, free, now)["fast_growth"]
    assert fg["after_step_from"] is not None
    assert abs(fg["pred"][1][0] - 59) < 0.5
    assert abs(fg["speed_pct_h"]) < 0.5


def test_unusual_growth_not_in_trend_and_already_above():
    """Швидке заповнення 40 -> 93 %: тренд без відрізка незвичного росту (лишається
    пласким), а коли диск уже вище 90 % — години до порогу = 0 («вже вище»)."""
    m = minutes(13)
    level = np.full(len(m) + 240, 40.0)
    level[-240:] = np.minimum(40 + 15 * np.arange(1, 241) / 60, 93)   # 15 %/год до 93 %
    level[-20:] = 93 + 0.5 * np.arange(1, 21) / 20                    # ще трохи росте
    used, free = disk(level)
    r = predict(used, free)
    assert r["trend"]["above_threshold_now"] is True
    assert r["trend"]["hours_to_threshold"] == 0.0
    assert fc.until_text(r["trend"]["hours_to_threshold"], 90) == "вже вище 90 %"

    mid = used.index[-240 + 90]                                       # у розпалі росту (~62 %)
    r_mid = predict(used, free, mid)
    assert r_mid["fast_growth"]["unusual"] is True
    assert r_mid["trend"]["excluded_growth_from"] is not None
    assert abs(r_mid["trend"]["speed_pct_day"]) < 1                   # тренд не «підхопив» ріст


def test_predictions_written(tmp_path):
    """Запис у predictions: по 4 рядки на диск (trend ×2, fast_growth, jump), model = linear."""
    db = tmp_path / "test.db"
    init_db(db)
    m = minutes(3)
    used, free = disk(40 + 2 * m / 1440)
    with db_session(db) as conn:
        conn.execute("INSERT INTO servers (name, kind) VALUES ('t-1', 'sim')")
        sid = conn.execute("SELECT id FROM servers WHERE name = 't-1'").fetchone()[0]
        rows = [(t.strftime(fc.TS_FORMAT), sid, "disk_used_pct:C:", float(v)) for t, v in used.items()]
        rows += [(t.strftime(fc.TS_FORMAT), sid, "disk_free_gb:C:", float(v)) for t, v in free.items()]
        conn.executemany("INSERT INTO metrics_clean (ts, server_id, metric, value) VALUES (?, ?, ?, ?)", rows)
        now, results = fc.forecast_server(conn, sid, None, CFG)
        n = fc.save_predictions(conn, sid, now, results, "linear")
        saved = conn.execute("SELECT model, kind, horizon_h, predicted_value, details_json "
                             "FROM predictions ORDER BY kind, horizon_h").fetchall()
    assert n == 4
    assert sorted(r["kind"] for r in saved) == ["fast_growth", "jump", "trend", "trend"]
    assert {r["model"] for r in saved} == {"linear"}
    trend24 = next(r for r in saved if r["kind"] == "trend" and r["horizon_h"] == 24)
    assert abs(json.loads(trend24["details_json"])["speed_pct_day"] - 2) < 0.1
