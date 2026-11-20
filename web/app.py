"""Дашборд, сторінка «Сервер» (етап 0). Запуск: streamlit run web/app.py

Усі числа — лише з функцій-інструментів core/tools.py (власних SQL-запитів тут немає).
Блок G «Що отримає LLM» показує JSON тих самих функцій з параметрами, які на
етапі 1 передаватиме бот.

Момент можна задати в адресі (для скриншотів і посилань з бота):
    http://localhost:8501/?server=sim-2&now=2026-11-10%2008:00&window=24&drive=E:
    &expand=1 — розгорнути всі згорнуті блоки.
"""
import json
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
DISK_HISTORY_DAYS = 14        # скільки історії диска показувати біля прогнозу
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


# ---------------------------------------------------------------------------
# Бічна панель: сервер, момент, вікно
# ---------------------------------------------------------------------------

servers = call("list_servers")["servers"]
if not servers:
    st.error("У базі немає серверів. Запустіть симулятор: python -m simulator.simulator --history 30")
    st.stop()
names = [s["name"] for s in servers]

with st.sidebar:
    st.header("Сервер")
    default = names.index(params["server"]) if params.get("server") in names else 0
    server = st.selectbox("Сервер", names, index=default, label_visibility="collapsed")
    info = next(s for s in servers if s["name"] == server)
    st.caption(info["description"] or "")
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
    live = st.toggle("Наживо (останні дані)", key="live")
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

# Початок вікна графіків (місцевий час) — від моменту «зараз»
now_dt = last if now is None else datetime.strptime(now, LOCAL_FMT)
win_start = (now_dt - timedelta(hours=window_h)).strftime(LOCAL_FMT)

st.title(f"{server}")

# ---------------------------------------------------------------------------
# A. Стан
# ---------------------------------------------------------------------------

status = call("get_status", server=server, now=now)
if show_error(status):
    st.stop()
st.caption(f"Момент: {status['now']['ts_local']} (UTC {status['now']['ts_utc']})"
           f"{' — наживо' if status['live'] else ''}. Дані до {status['data_until']['ts_local']}.")
if status["stale"]:
    st.error(f"Сервер замовк: немає нових даних {status['silent_min']:.0f} хв "
             f"(останні — {status['data_until']['ts_local']}).")

st.subheader("A. Стан")
m = status["metrics"]
cards = [("CPU", f"{m.get('cpu_pct')} %"), ("RAM", f"{m.get('ram_pct')} %")]
cards += [(f"Диск {d['drive']}", f"{d['used_pct']} %", f"вільно {d['free_gb']} з {d['total_gb']} ГБ")
          for d in status["disks"]]
if "app_response_ms" in m:
    cards += [("Відповідь сайту", f"{m['app_response_ms']:.0f} мс"), ("Помилки сайту", f"{m.get('app_error_pct')} %")]
for col, card in zip(st.columns(len(cards)), cards):
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
                     "значення": ", ".join(f"{k} = {v}" for k, v in value.items()),
                     "деталі": c["details"], "перевірено": c["checked_local"]})
    chk = pd.DataFrame(rows)
    if not chk.empty:
        st.dataframe(red_rows(chk, chk["ok"] != "так"), hide_index=True)
for note in status["notes"]:
    st.caption(note)

# ---------------------------------------------------------------------------
# B. Графіки
# ---------------------------------------------------------------------------

st.subheader(f"B. Графіки за {win_label}")


def metric_chart(metric: str, title: str, height: int = 260):
    """Лінія метрики; заповнені точки — сірі кружечки, викиди — червоні."""
    r = call("get_metric_history", server=server, metric=metric, start=win_start, now=now,
             max_points=DASH_POINTS)
    if show_error(r):
        return
    fig = go.Figure()
    fig.add_scatter(x=ts(r["series"]), y=[p[1] for p in r["series"]], mode="lines", name=title,
                    line=dict(width=2, color="#1f77b4"))
    if r["imputed_points"]:
        fig.add_scatter(x=ts(r["imputed_points"]), y=[p[1] for p in r["imputed_points"]], mode="markers",
                        name="заповнено (ETL)", marker=dict(size=8, color="#999999", symbol="circle-open"))
    if r["outlier_points"]:
        fig.add_scatter(x=ts(r["outlier_points"]), y=[p[1] for p in r["outlier_points"]], mode="markers",
                        name="викид (ETL)", marker=dict(size=9, color="#d62728"))
    step = f", середнє по {r['step_min']} хв" if r["step_min"] > 1 else ""
    fig.update_layout(title=f"{title}, {r['unit']}{step} — макс {r['max']['value']} о {r['max']['ts_local'][11:]}, "
                            f"сер. {r['mean']}; заповнено {r['imputed']}, викидів {r['outliers']}, "
                            f"пропусків {r['gaps_total']}",
                      height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15), title_font_size=13)
    st.plotly_chart(fig)


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
disk_start = (now_dt - timedelta(days=DISK_HISTORY_DAYS)).strftime(LOCAL_FMT)
hist = call("get_metric_history", server=server, metric=f"disk_used_pct:{drive}", start=disk_start,
            now=now, max_points=DASH_POINTS)
if not show_error(fc) and not show_error(hist):
    d = fc["disks"][0]
    thr = fc["threshold_pct"]
    fig = go.Figure()
    fig.add_scatter(x=ts(hist["series"]), y=[p[1] for p in hist["series"]], mode="lines", name="зайнято, %",
                    line=dict(width=2, color="#1f77b4"))
    if d.get("trend_line"):
        tl = d["trend_line"]
        x = ts(tl)
        fig.add_scatter(x=x + x[::-1], y=[p[3] for p in tl] + [p[2] for p in tl][::-1], fill="toself",
                        fillcolor="rgba(255,127,14,0.15)", line=dict(width=0), name="інтервал 10–90 %",
                        hoverinfo="skip")
        fig.add_scatter(x=x, y=[p[1] for p in tl], mode="lines", name="тренд",
                        line=dict(width=2, color="#ff7f0e", dash="dash"))
    fig.add_hline(y=thr, line=dict(color="#d62728", width=1, dash="dot"),
                  annotation_text=f"{thr} %", annotation_position="top left")
    moment = pd.Timestamp(fc["now"]["ts_local"])
    fig.add_shape(type="line", x0=moment, x1=moment, y0=0, y1=1, yref="paper",
                  line=dict(color="#555555", width=1))
    fig.add_annotation(x=moment, y=1, yref="paper", text="зараз", showarrow=False, yanchor="bottom")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis=dict(range=[max(0, min(p[1] for p in hist["series"]) - 10), 101]),
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig)

    tr, fg, jp = d["trend"], d["fast_growth"], d["jump"]

    def until(h):
        if h is None:
            return "не досягне"
        if h == 0:
            return f"вже вище {thr} %"
        return f"до {thr} % — {h:.0f} год" if h < 72 else f"до {thr} % — {h / 24:.1f} дн"

    st.markdown(f"**Зараз:** {d['now_pct']} %, вільно {d['now_free_gb']} ГБ")
    if tr["status"] == "ok":
        st.markdown(f"**1. Тренд** (з {tr['from_local']}): {tr['pct_per_day']:+} %/добу "
                    f"({tr['gb_per_day']} ГБ/добу); +24 год — {tr['in_24h']['pct']} % "
                    f"({tr['in_24h']['low']}–{tr['in_24h']['high']}); +7 днів — {tr['in_7d']['pct']} %; "
                    f"**{until(tr['hours_to_threshold'])}**")
    else:
        st.markdown(f"**1. Тренд:** {tr['status']} (з {tr['from_local']})")
    if fg["status"] == "ok":
        st.markdown(f"**2. Ріст за {fg['window_min']} хв:** {fg['pct_per_h']:+} %/год (зазвичай до "
                    f"{fg['usual_max_pct_per_h']}, межа {fg['limit_pct_per_h']}) — "
                    f"{'**НЕЗВИЧНИЙ**' if fg['unusual'] else 'звичайний'}; {until(fg['hours_to_threshold'])}")
    if jp["found"]:
        st.markdown(f"**3. Регулярний стрибок** ~{jp['size_gb']:.0f} ГБ ({jp['days_seen']} з "
                    f"{jp['lookback_days']} днів), наступний {jp['next_local']}: вільно буде "
                    f"{jp['free_then_gb']:.0f} ГБ — **{jp['text']}**")
    else:
        st.markdown(f"**3.** {jp['text']}")
    for note in d["notes"]:
        st.caption(note)

# ---------------------------------------------------------------------------
# D. Процеси
# ---------------------------------------------------------------------------

st.subheader("D. Процеси")
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

st.subheader("E. Що росте за 24 год")
g = call("get_folder_growth", server=server, now=now)
if not show_error(g):
    labels = [f["path"] for f in g["folders"]] + [f"інше ({d['drive']})" for d in g["disks"]]
    values = [f["growth_gb"] for f in g["folders"]] + [d["other_gb"] for d in g["disks"]]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=[f"{v:+.2f} ГБ" for v in values],
                           marker_color=["#1f77b4"] * len(g["folders"]) + ["#aaaaaa"] * len(g["disks"])))
    fig.update_layout(height=60 + 35 * len(labels), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis_title="ріст, ГБ")
    st.plotly_chart(fig)
    st.caption("; ".join(f"{d['drive']}: диск {d['growth_gb']:+} ГБ, теки {d['folders_gb']:+} ГБ, "
                         f"інше {d['other_gb']:+} ГБ" for d in g["disks"]))
    for note in g["notes"]:
        st.caption(note)

# ---------------------------------------------------------------------------
# F. Хронологія
# ---------------------------------------------------------------------------

st.subheader(f"F. Хронологія за {win_label}")
tl = call("get_timeline", server=server, start=win_start, now=now, limit=500)
if not show_error(tl):
    types = st.multiselect("Типи", ["event", "service", "check", "alert"],
                           default=["event", "service", "check", "alert"])
    df = pd.DataFrame(tl["items"])
    if not df.empty:
        df = df[df["type"].isin(types)].astype(object)
        df = df.where(df.notna(), "")
        st.dataframe(df.rename(columns={"ts_local": "час", "type": "тип", "source": "джерело", "code": "код",
                                        "level": "рівень", "text": "текст"}), hide_index=True)
    else:
        st.caption("за цей період нічого не сталося")
    st.caption(f"За типами: {tl['counts']}; шумові події (лише кількість): {tl['noise'] or 'немає'}")
    for note in tl.get("notes", []):
        st.caption(note)

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
