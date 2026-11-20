"""Дашборд, сторінка «Сервер» (етап 0). Запуск: streamlit run web/app.py

Усі числа — лише з функцій-інструментів core/tools.py (власних SQL-запитів тут немає).
Блок G «Що отримає LLM» показує JSON тих самих функцій з параметрами, які на
етапі 1 передаватиме бот.

Момент можна задати в адресі (для скриншотів і посилань з бота):
    http://localhost:8501/?server=sim-2&now=2026-11-10%2008:00&window=24&drive=E:
    &expand=1 — розгорнути всі згорнуті блоки.
"""
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# streamlit запускає файл з теки web/, тому корінь проєкту додаємо в шлях імпорту
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import tools  # noqa: E402

WINDOWS = {"6 год": 6, "24 год": 24, "7 днів": 168}
DASH_POINTS = 1500            # точок на графік дашборду (для LLM — 50, за замовчуванням)
DISK_HISTORY_DAYS = 14        # запасне значення, якщо прогноз не повернув history_days (помилка)
LOCAL_FMT = "%Y-%m-%d %H:%M"

st.set_page_config(page_title="Сервер — асистент чергового", layout="wide")
params = st.query_params
EXPAND = params.get("expand") == "1"


# ---------------------------------------------------------------------------
# Виклики інструментів: кеш (на історії відповідь для моменту не змінюється)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def call(name: str, **kwargs) -> dict:
    return tools.TOOLS[name](**kwargs)


def show_error(result: dict) -> bool:
    """Показує помилку інструмента. True — помилка була."""
    if "error" in result:
        st.warning(result["error"])
        return True
    return False


def ts(series: list) -> list:
    return [pd.Timestamp(p[0]) for p in series]


def fmt(value, unit: str = "", digits: int | None = None, sign: bool = False) -> str:
    """Число для показу; None — «—» (немає даних), а не падіння."""
    if value is None:
        return "—"
    value = 0.0 if value == 0 else value                 # -0.0 == 0, тож «-0.0» не буде
    if digits is not None:
        text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    else:
        text = f"{value:+}" if sign else f"{value}"
    return text + unit


# ---------------------------------------------------------------------------
# Бічна панель: сервер, момент, вікно
# ---------------------------------------------------------------------------

listing = call("list_servers")
if show_error(listing):
    st.stop()
servers = listing["servers"]
if not servers:
    st.error("У базі немає серверів. Запустіть симулятор: python -m simulator.simulator --history 30")
    st.stop()
names = [s["name"] for s in servers]
main_top = st.container()          # місце в основній частині для повідомлень з бічної панелі

with st.sidebar:
    st.header("Сервер")
    default = names.index(params["server"]) if params.get("server") in names else 0
    server = st.selectbox("Сервер", names, index=default, label_visibility="collapsed")
    info = next(s for s in servers if s["name"] == server)
    st.caption(info["description"] or "")
    if info["data_until"] is None:
        # сирі дані є, а очищених (metrics_clean) ще немає — наприклад, після перегенерації
        main_top.warning(f"Для {server} ще немає очищених даних (metrics_clean). "
                   "Запустіть ETL: python -m core.etl, потім оновіть сторінку.")
        st.stop()
    first = datetime.strptime(info["data_from"]["ts_local"], LOCAL_FMT)
    last = datetime.strptime(info["data_until"]["ts_local"], LOCAL_FMT)

    # Момент «зараз»: наживо або дата + час (місцевий). Стан — у session_state,
    # кнопки ±1 год / ±1 доба зсувають його.
    if "moment" not in st.session_state or st.session_state.get("moment_server") != server:
        start = last
        if params.get("now"):
            try:
                start = datetime.strptime(params["now"], LOCAL_FMT)
            except ValueError:
                pass
        # момент не обмежуємо межами даних: за межами інструменти чесно скажуть «немає даних»
        st.session_state.moment = start
        st.session_state.moment_server = server
        st.session_state.live = not params.get("now")
        st.session_state.m_date = st.session_state.moment.date()
        st.session_state.m_time = st.session_state.moment.time()

    def shift(hours: float) -> None:
        m = datetime.combine(st.session_state.m_date, st.session_state.m_time) + timedelta(hours=hours)
        st.session_state.m_date, st.session_state.m_time = m.date(), m.time()
        st.session_state.live = False

    st.subheader("Момент (місцевий час, UTC+2)")
    live = st.toggle("Наживо (поточний час)", key="live")
    st.date_input("Дата", key="m_date", min_value=first.date() - timedelta(days=365),
                  max_value=last.date() + timedelta(days=365), disabled=live, format="DD.MM.YYYY")
    st.time_input("Час", key="m_time", step=timedelta(minutes=5), disabled=live)
    c1, c2, c3, c4 = st.columns(4)
    c1.button("−1 д", on_click=shift, args=(-24,), width="stretch")
    c2.button("−1 г", on_click=shift, args=(-1,), width="stretch")
    c3.button("+1 г", on_click=shift, args=(1,), width="stretch")
    c4.button("+1 д", on_click=shift, args=(24,), width="stretch")
    win_default = next((k for k, v in WINDOWS.items() if str(v) == params.get("window")), "24 год")
    win_label = st.radio("Вікно графіків", list(WINDOWS), index=list(WINDOWS).index(win_default),
                         horizontal=True)
    window_h = WINDOWS[win_label]
    st.caption(f"Дані: {info['data_from']['ts_local']} – {info['data_until']['ts_local']}")

now = None if live else datetime.combine(st.session_state.m_date, st.session_state.m_time).strftime(LOCAL_FMT)

st.title(f"{server}")

# ---------------------------------------------------------------------------
# A. Стан
# ---------------------------------------------------------------------------

status = call("get_status", server=server, now=now)
if show_error(status):
    st.stop()

# Момент «зараз» (у живому режимі — поточний час) і початок вікна графіків, місцевий час
now_dt = datetime.strptime(status["now"]["ts_local"], LOCAL_FMT)
win_start = (now_dt - timedelta(hours=window_h)).strftime(LOCAL_FMT)

st.caption(f"Момент: {status['now']['ts_local']} (UTC {status['now']['ts_utc']})"
           f"{' — наживо' if status['live'] else ''}. Дані до {status['data_until']['ts_local']}.")
if status["stale"]:
    st.error(f"Сервер замовк: немає нових даних {tools.age_text(status['data_age_h'])} "
             f"(останні — {status['data_until']['ts_local']}). Усі блоки нижче — останні відомі дані.")


def stale_mark(result: dict) -> None:
    """Підпис блоку, коли свіжих даних немає (stale): значення — останні відомі, а не поточні."""
    if result.get("stale"):
        st.caption(f"⚠ останні відомі дані на {result['data_until']['ts_local']} "
                   f"({tools.age_text(result['data_age_h'])} тому)")


def show_notes(result: dict) -> None:
    """Лише ситуаційні примітки (notes). Постійні (about) — лише в JSON для LLM."""
    for note in result.get("notes", []):
        st.caption(note)


st.subheader("A. Стан")
stale_mark(status)
m = status["metrics"]
cards = [("CPU", fmt(m.get("cpu_pct"), " %")), ("RAM", fmt(m.get("ram_pct"), " %"))]
cards += [(f"Диск {d['drive']}", fmt(d["used_pct"], " %"),
           f"вільно {fmt(d['free_gb'])} з {fmt(d['total_gb'])} ГБ") for d in status["disks"]]
if "app_response_ms" in m:
    cards += [("Відповідь сайту", fmt(m["app_response_ms"], " мс", 0)),
              ("Помилки сайту", fmt(m.get("app_error_pct"), " %"))]
# Більше 4 карток — два рядки, інакше у вузькому вікні числа обрізаються («2…»)
per_row = len(cards) if len(cards) <= 4 else math.ceil(len(cards) / 2)
for i in range(0, len(cards), per_row):
    row = cards[i:i + per_row]
    for col, card in zip(st.columns(per_row), row):
        col.metric(card[0], card[1], border=True)
        if len(card) > 2:
            col.caption(card[2])


def red_rows(df: pd.DataFrame, bad: pd.Series):
    """Червоне тло для рядків-проблем (зупинена служба, перевірка не пройшла)."""
    df = df.astype(object).where(df.notna(), "")       # порожня клітинка замість None
    return df.style.apply(lambda row: ["background-color: #f8d7da; color: #721c24" if bad[row.name] else ""
                                       for _ in row], axis=1)


left, right = st.columns(2)
with left:
    st.markdown("**Служби і пули IIS**")
    svc = pd.DataFrame(status["services"])
    if not svc.empty:
        svc = svc.reindex(columns=["name", "kind", "status", "since_local", "checked_local", "since_note"])
        svc.columns = ["назва", "тип", "стан", "відколи", "перевірено", "примітка"]
        st.dataframe(red_rows(svc, svc["стан"] != "running"), hide_index=True)
with right:
    st.markdown("**Перевірки**")
    rows = []
    for name, c in status["checks"].items():
        value = {k: v for k, v in c.items() if k not in ("ok", "details", "checked_local")}
        rows.append({"перевірка": name, "ok": "так" if c["ok"] else "НІ",
                     "значення": ", ".join(f"{k} = {fmt(v)}" for k, v in value.items()),
                     "деталі": c["details"], "перевірено": c["checked_local"]})
    chk = pd.DataFrame(rows)
    if not chk.empty:
        st.dataframe(red_rows(chk, chk["ok"] != "так"), hide_index=True)
        st.caption("http — щохвилинний запит агента до сторінки перевірки сайту: код і час відповіді "
                   "(не пройшла — код ≠ 200 або довше 2 с); адреса в даних не зберігається, лише код. "
                   "cert — днів до кінця сертифіката сайту; backup — вік останнього успішного бекапу, год; "
                   "sql_port — чи відкривається порт SQL Server.")
show_notes(status)

# ---------------------------------------------------------------------------
# B. Графіки
# ---------------------------------------------------------------------------

st.subheader(f"B. Графіки за {win_label}")
stale_mark(status)


def mark_now(fig) -> None:
    """Вертикаль «зараз» — щоб період без даних між останньою точкою і «зараз» було видно."""
    fig.add_shape(type="line", x0=now_dt, x1=now_dt, y0=0, y1=1, yref="paper",
                  line=dict(color="#888888", width=1, dash="dot"))
    fig.add_annotation(x=now_dt, y=1, yref="paper", text="зараз", showarrow=False, yanchor="bottom",
                       xanchor="right")


def metric_chart(metric: str, title: str, height: int = 260):
    """Лінія метрики; заповнені точки — сірі кружечки, викиди — червоні.
    Вісь X — від початку вікна до моменту «зараз», навіть якщо дані закінчились раніше."""
    r = call("get_metric_history", server=server, metric=metric, start=win_start, now=now,
             max_points=DASH_POINTS)
    if show_error(r):
        return
    fig = go.Figure()
    fig.update_xaxes(range=[datetime.strptime(win_start, LOCAL_FMT), now_dt])
    mark_now(fig)
    fig.add_scatter(x=ts(r["series"]), y=[p[1] for p in r["series"]], mode="lines", name=title,
                    line=dict(width=2, color="#1f77b4"))
    if r["imputed_points"]:
        fig.add_scatter(x=ts(r["imputed_points"]), y=[p[1] for p in r["imputed_points"]], mode="markers",
                        name="заповнений пропуск (ETL)", marker=dict(size=8, color="#999999", symbol="circle-open"))
    if r["outlier_points"]:
        fig.add_scatter(x=ts(r["outlier_points"]), y=[p[1] for p in r["outlier_points"]], mode="markers",
                        name="короткий сплеск (позначка ETL)", marker=dict(size=9, color="#d62728"))
    step = f", середнє по {r['step_min']} хв" if r["step_min"] > 1 else ""
    fig.update_layout(title=f"{title}, {r['unit']}{step}",
                      height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), title_font_size=13)
    st.plotly_chart(fig)
    # статистика — під графіком (у заголовку довгий рядок обрізався)
    st.caption(f"макс {fmt(r['max']['value'])} о {r['max']['ts_local']}, сер. {fmt(r['mean'])}; "
               f"заповнених пропусків {r['imputed']}, коротких сплесків {r['outliers']}, "
               f"незаповнених пропусків {r['gaps_total']}")
    show_notes(r)


c1, c2 = st.columns(2)
with c1:
    metric_chart("cpu_pct", "CPU")
with c2:
    metric_chart("ram_pct", "RAM")
if "app_response_ms" in m:
    c1, c2 = st.columns(2)
    with c1:
        metric_chart("app_response_ms", "Час відповіді сайту")
    with c2:
        metric_chart("app_error_pct", "Помилки сайту")
    sites = ", ".join(f"{x['site']} (пул {x['app_pool']})" for x in status["sites"])
    st.caption(f"Сайт — запити користувачів до сайтів сервера ({sites}): середній час відповіді "
               "за хвилину (мс) і частка запитів, що завершились помилкою (%). Це не те саме, що "
               "HTTP-перевірка в блоці A: вона — один запит агента щохвилини.")
with st.expander("Диски і мережа", expanded=EXPAND):
    c1, c2 = st.columns(2)
    for i, d in enumerate(status["disks"]):
        with (c1 if i % 2 == 0 else c2):
            metric_chart(f"disk_used_pct:{d['drive']}", f"Диск {d['drive']} зайнято")
    c1, c2 = st.columns(2)
    with c1:
        metric_chart("disk_read_mbs", "Читання з дисків")
        metric_chart("net_in_mbs", "Мережа, вхід")
    with c2:
        metric_chart("disk_write_mbs", "Запис на диски")
        metric_chart("net_out_mbs", "Мережа, вихід")

# ---------------------------------------------------------------------------
# C. Диск: історія за 14 днів + прогноз
# ---------------------------------------------------------------------------

st.subheader("C. Диск і прогноз")
drives = [d["drive"] for d in status["disks"]]
drive_default = drives.index(params["drive"]) if params.get("drive") in drives else 0
drive = st.radio("Диск", drives, index=drive_default, horizontal=True)
fc = call("get_disk_forecast", server=server, drive=drive, now=now, with_line=True)
# історія на графіку — рівно стільки, скільки бачить прогноз (history_days з forecast.yaml)
history_days = fc.get("history_days", DISK_HISTORY_DAYS)
disk_start = (now_dt - timedelta(days=history_days)).strftime(LOCAL_FMT)
st.caption(f"Історія на графіку — {history_days} днів: стільки бачить прогноз.")
hist = call("get_metric_history", server=server, metric=f"disk_used_pct:{drive}", start=disk_start,
            now=now, max_points=DASH_POINTS)
stale_mark(fc)
full_scale = st.toggle("Вісь 0–100 %", key="disk_full_scale")
d, label = None, ""
if not show_error(fc):
    if "disks" in fc:
        d = fc["disks"][0]
    else:
        # немає свіжих даних: прогнозу на «зараз» немає — лише на момент останніх даних, з позначкою
        st.warning(fc["message"])
        d = fc["forecast_at_last_data"]["disks"][0]
        label = f" (на момент останніх даних, {fc['forecast_at_last_data']['at']['ts_local']})"
        st.caption(f"⚠ {fc['forecast_at_last_data']['note']}")
    if "error" in d:
        # прогноз для цього диска не порахувався (наприклад, замало даних) — плашка, а не KeyError
        st.warning(f"Прогноз для {drive}: {d['error']}")
        d = None
thr = fc.get("threshold_pct", 90)

# Графік: історія за 14 днів до «зараз» + пряма тренду; вісь X — щонайменше до «зараз»
fig = go.Figure()
ys = []
if not show_error(hist):
    fig.add_scatter(x=ts(hist["series"]), y=[p[1] for p in hist["series"]], mode="lines", name="зайнято, %",
                    line=dict(width=2, color="#1f77b4"))
    ys += [p[1] for p in hist["series"]]
x_ends = [now_dt]
# Прогноз малюємо лише на свіжих даних; коли stale — на графіку тільки справжні дані,
# «зараз» і плашка (прогноз на момент останніх даних — лише текстом нижче, з позначкою)
if d and d.get("trend_line") and not label:
    tl = d["trend_line"]
    # обрізаємо там, де пряма дійшла до 100 %: далі диск повний, пласка лінія на 100 % нічого не каже
    full = next((i for i, p in enumerate(tl) if p[1] >= 100), None)
    if full is not None:
        tl = tl[:full + 1]
    x = ts(tl)
    fig.add_scatter(x=x + x[::-1], y=[p[3] for p in tl] + [p[2] for p in tl][::-1], fill="toself",
                    fillcolor="rgba(255,127,14,0.15)", line=dict(width=0), name="інтервал 10–90 %",
                    hoverinfo="skip")
    fig.add_scatter(x=x, y=[p[1] for p in tl], mode="lines", name="тренд",
                    line=dict(width=2, color="#ff7f0e", dash="dash"))
    ys += [p[1] for p in tl]
    x_ends.append(x[-1])

# Що сталося насправді після «зараз» — лише для моменту в історії і лише на дашборді:
# той самий інструмент з пізнішим now. Функції для LLM, як і раніше, бачать лише дані до «зараз»
if now is not None and not fc.get("stale"):
    horizon = (now_dt + timedelta(hours=168)).strftime(LOCAL_FMT)
    after = call("get_metric_history", server=server, metric=f"disk_used_pct:{drive}", start=now,
                 end=horizon, now=horizon, max_points=DASH_POINTS)
    if "error" not in after and after["series"]:
        fig.add_scatter(x=ts(after["series"]), y=[p[1] for p in after["series"]], mode="lines",
                        name="насправді (прогноз цього не бачив)", line=dict(width=2, color="#9e9e9e"))
        ys += [p[1] for p in after["series"]]
        x_ends.append(ts(after["series"])[-1])

fig.add_hline(y=thr, line=dict(color="#d62728", width=1, dash="dot"),
              annotation_text=f"{thr} %", annotation_position="top left")
mark_now(fig)
x_end = max(x_ends)
fig.update_xaxes(range=[datetime.strptime(disk_start, LOCAL_FMT), x_end])
if full_scale or not ys:
    y_range = [0, 101]
else:
    # вісь Y — за даними з запасом: на «тихому» диску видно й невеликі зміни
    lo, hi = min(ys), max(ys)
    pad = max(2.0, (hi - lo) * 0.15)
    y_range = [max(0, lo - pad), min(101, hi + pad)]
fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  yaxis=dict(range=y_range), legend=dict(orientation="h", y=-0.12))
st.plotly_chart(fig)
if "error" not in hist:
    show_notes(hist)

if d:
    tr, fg, jp = d["trend"], d["fast_growth"], d["jump"]
    if tr["status"] == "ok" and not label:
        # звідки прогноз — один рядок під графіком
        st.caption(f"Прогноз: пряма по погодинних даних від {tr['from_local']} ({tr['from_reason']}), "
                   f"{tr['points']} точок; історія для пошуку стрибків — {fc['history_days']} днів.")
    elif tr["status"] == "мало даних" and not label:
        # чому пунктиру немає: від початку прямої ще замало погодинних точок
        st.caption(f"Тренд: мало даних — від {tr['from_local']} ({tr['from_reason']}) погодинних точок: "
                   f"{tr['points']} з потрібних {tr['min_points']}; пунктиру тренду немає. "
                   "Питання 2 і 3 нижче рахуються.")

    def until(h):
        if h is None:
            return "не досягне"
        if h == 0:
            return f"вже вище {thr} %"
        return f"до {thr} % — {h:.0f} год" if h < 72 else f"до {thr} % — {h / 24:.1f} дн"

    if d["to_threshold_pp"] is not None and d["to_threshold_pp"] <= 0:
        distance = f"вже вище {thr} %"
    else:
        distance = (f"до {thr} % лишилось {fmt(d['to_threshold_pp'])} п. п. "
                    f"({fmt(d['to_threshold_gb'], ' ГБ')})")
    st.markdown(f"**{'Останнє відоме' if label else 'Зараз'}{label}:** {fmt(d['now_pct'], ' %')}, "
                f"вільно {fmt(d['now_free_gb'], ' ГБ')}; {distance}")
    if tr["status"] == "ok":
        st.markdown(f"**1. Тренд** (з {tr['from_local']}): {fmt(tr['pct_per_day'], sign=True)} %/добу "
                    f"({fmt(tr['gb_per_day'])} ГБ/добу); +24 год — {fmt(tr['in_24h']['pct'], ' %')} "
                    f"({fmt(tr['in_24h']['low'])}–{fmt(tr['in_24h']['high'])}); "
                    f"+7 днів — {fmt(tr['in_7d']['pct'], ' %')}; **{until(tr['hours_to_threshold'])}**")
    else:
        st.markdown(f"**1. Тренд:** {tr['status']} (з {tr['from_local']}; погодинних точок: "
                    f"{tr['points']} з потрібних {tr['min_points']})")
    if fg["status"] == "ok":
        st.markdown(f"**2. Ріст за {fg['window_min']} хв:** {fmt(fg['pct_per_h'], sign=True)} %/год "
                    f"(зазвичай до {fmt(fg['usual_max_pct_per_h'])}, межа {fmt(fg['limit_pct_per_h'])}) — "
                    f"{'**НЕЗВИЧНИЙ**' if fg['unusual'] else 'звичайний'}; {until(fg['hours_to_threshold'])}")
    else:
        st.markdown(f"**2. Ріст:** {fg['status']}")
    if jp["found"]:
        st.markdown(f"**3. Регулярний стрибок** ~{fmt(jp['size_gb'], ' ГБ', 0)} ({jp['days_seen']} з "
                    f"{jp['lookback_days']} днів), наступний {jp['next_local']}: вільно буде "
                    f"{fmt(jp['free_then_gb'], ' ГБ', 0)} — **{jp['text']}**")
    else:
        st.markdown(f"**3.** {jp['text']}")
    show_notes(d)

# ---------------------------------------------------------------------------
# D. Процеси
# ---------------------------------------------------------------------------

st.subheader("D. Процеси")
stale_mark(status)
c1, c2 = st.columns(2)
for col, by, title in ((c1, "cpu", "Топ за CPU"), (c2, "ram", "Топ за RAM")):
    with col:
        p = call("get_top_processes", server=server, now=now, by=by)
        if show_error(p):
            continue
        st.markdown(f"**{title}** — знімок {p['snapshot_local']}; CPU сервера {p['cpu_server']} %, "
                    f"RAM {p['ram_server_pct']} %")
        df = pd.DataFrame(p["processes"]).rename(columns={"name": "процес", "cpu_pct": "CPU, %",
                                                          "ram_mb": "RAM, МБ", "app_pool": "пул", "service": "служба"})
        df[["пул", "служба"]] = df[["пул", "служба"]].astype(object).where(df[["пул", "служба"]].notna(), "")
        st.dataframe(df, hide_index=True)

# ---------------------------------------------------------------------------
# E. Теки
# ---------------------------------------------------------------------------

st.subheader(f"E. Що росте за {win_label}")
g = call("get_folder_growth", server=server, period_h=window_h, now=now)
stale_mark(g)
if not show_error(g):
    labels = [f["path"] for f in g["folders"]] + [f"інше ({dk['drive']})" for dk in g["disks"]]
    values = [f["growth_gb"] for f in g["folders"]] + [dk["other_gb"] for dk in g["disks"]]
    # None (немає даних на початку періоду) — стовпчика немає, підпис «—»
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=[fmt(v, " ГБ", 2, sign=True) for v in values],
                           marker_color=["#1f77b4"] * len(g["folders"]) + ["#aaaaaa"] * len(g["disks"])))
    fig.update_layout(height=60 + 35 * len(labels), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="ріст, ГБ")
    st.plotly_chart(fig)
    st.caption("; ".join(f"{dk['drive']}: диск {fmt(dk['growth_gb'], ' ГБ', sign=True)}, "
                         f"теки {fmt(dk['folders_gb'], ' ГБ', sign=True)}, "
                         f"інше {fmt(dk['other_gb'], ' ГБ', sign=True)}" for dk in g["disks"]))
    show_notes(g)

# ---------------------------------------------------------------------------
# F. Хронологія
# ---------------------------------------------------------------------------

st.subheader(f"F. Хронологія за {win_label}")
tl = call("get_timeline", server=server, start=win_start, now=now, limit=500)
stale_mark(tl)
if not show_error(tl):
    types = st.multiselect("Типи", ["event", "service", "check", "alert"],
                           default=["event", "service", "check", "alert"])
    df = pd.DataFrame(tl["items"])
    if not df.empty:
        df = df[df["type"].isin(types)].astype(object)
        df = df.where(df.notna(), "")
        st.dataframe(df.rename(columns={"ts_local": "час", "type": "тип", "source": "джерело", "code": "код",
                                        "level": "рівень", "text": "текст"}), hide_index=True)
    elif not tl["notes"]:
        st.caption("за цей період нічого не сталося")
    type_names = {"event": "події журналу", "service": "зміни служб", "check": "зміни перевірок",
                  "alert": "алерти"}
    by_type = ", ".join(f"{type_names[k]} — {v}" for k, v in tl["counts"].items())
    noise = ", ".join(f"{code} × {n}" for code, n in tl["noise"].items()) or "немає"
    st.caption(f"За типами: {by_type}. Шумові події (лише кількість): {noise}.")
    show_notes(tl)

# ---------------------------------------------------------------------------
# G. Що отримає LLM
# ---------------------------------------------------------------------------

st.subheader("G. Що отримає LLM")
st.caption("Ті самі функції з параметрами за замовчуванням, як їх викличе бот на етапі 1 "
           "(ряд — 50 точок, без прямої прогнозу). Числа збігаються з блоками вище.")
llm_calls = [
    ("get_status", dict(server=server, now=now)),
    ("get_metric_history", dict(server=server, metric="cpu_pct", now=now)),
    ("get_metric_history", dict(server=server, metric="ram_pct", now=now)),
    ("get_disk_forecast", dict(server=server, drive=drive, now=now)),
    ("get_top_processes", dict(server=server, now=now, by="cpu")),
    ("get_top_processes", dict(server=server, now=now, by="ram")),
    ("get_folder_growth", dict(server=server, now=now)),
    ("get_timeline", dict(server=server, now=now)),
]
with st.expander("JSON викликів", expanded=EXPAND):
    for name, kwargs in llm_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        result = call(name, **kwargs)
        size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        st.markdown(f"`{name}({args})` — ~{size / 1024:.1f} КБ")
        st.json(result, expanded=EXPAND)
