"""Опис інструментів для LLM: назва, опис і параметри кожної функції з core/tools.py.

- description — docstring функції до рядка «Лише для коду:» (одне джерело тексту:
  те, що читає розробник, читає і LLM);
- input_schema — параметри, які бачить LLM, написані тут вручну: тип, опис,
  допустимі значення (сервери і диски — зі списку list_servers() на старті);
- now, max_points, with_line LLM не бачить: їх ставить код (core/assistant.py).

list_servers як інструмент не даємо: його короткий підсумок іде в системний промпт.

Перевірка (без LLM): python -m core.assistant --show-tools
"""

import inspect

from core import tools

CODE_ONLY_MARK = "Лише для коду:"            # з цього рядка docstring LLM не бачить
HIDDEN_PARAMS = ("now", "max_points", "with_line")
TOOL_NAMES = ("get_status", "get_metric_history", "get_top_processes",
              "get_folder_growth", "get_disk_forecast", "get_timeline")

# Як назвати інструмент людині (рядок «Дивився: …» під відповіддю)
HUMAN_NAMES = {"get_status": "стан", "get_metric_history": "історія метрики",
               "get_top_processes": "процеси", "get_folder_growth": "ріст тек",
               "get_disk_forecast": "прогноз диска", "get_timeline": "хронологія"}

BASE_METRICS = ["cpu_pct", "ram_pct", "disk_read_mbs", "disk_write_mbs", "net_in_mbs",
                "net_out_mbs", "app_response_ms", "app_error_pct"]

TIME_NOTE = "місцевий час у форматі YYYY-MM-DD HH:MM, наприклад 2026-11-10 08:00"


def description(name: str) -> str:
    """Docstring функції без частини «Лише для коду:»."""
    doc = inspect.getdoc(tools.TOOLS[name]) or ""
    return doc.split(CODE_ONLY_MARK)[0].strip()


def enum_or_free(values: list, base: dict) -> dict:
    """Параметр зі списком допустимих значень (якщо список не порожній)."""
    out = dict(base)
    if values:
        out["enum"] = list(values)
    return out


def build_specs(servers: list | None = None) -> list[dict]:
    """Специфікації інструментів для LLM.

    servers — список з tools.list_servers()["servers"]; None — прочитати зараз.
    Береться лише статичне: назви серверів і їхні диски."""
    if servers is None:
        servers = tools.list_servers().get("servers", [])
    names = [s["name"] for s in servers]
    disks = sorted({d for s in servers for d in s.get("disks", [])})
    metrics = BASE_METRICS + [f"disk_used_pct:{d}" for d in disks] + [f"disk_free_gb:{d}" for d in disks]

    server = enum_or_free(names, {"type": "string", "description": "назва сервера"})
    drive = enum_or_free(disks, {"type": "string",
                                 "description": "лише цей диск, наприклад E:; не вказуйте — усі диски"})
    start = {"type": "string", "description": f"початок періоду, {TIME_NOTE}; "
                                              f"не вказуйте — за 24 год до «зараз»"}
    end = {"type": "string", "description": f"кінець періоду, {TIME_NOTE}; не пізніше «зараз»; "
                                            f"не вказуйте — «зараз»"}

    params = {
        "get_status": ({"server": {**server, "description": "назва сервера; не вказуйте — стан усіх серверів"}},
                       []),
        "get_metric_history": ({
            "server": server,
            "metric": enum_or_free(metrics, {
                "type": "string",
                "description": "метрика: cpu_pct — CPU, %; ram_pct — пам'ять, %; disk_used_pct:<диск> — "
                               "зайнято на диску, %; disk_free_gb:<диск> — вільно, ГБ; app_response_ms — "
                               "час відповіді сайту, мс; app_error_pct — частка помилок сайту, %; "
                               "disk_read_mbs, disk_write_mbs, net_in_mbs, net_out_mbs — МБ/с"}),
            "start": start, "end": end}, ["server", "metric"]),
        "get_top_processes": ({
            "server": server,
            "by": {"type": "string", "enum": ["cpu", "ram"],
                   "description": "сортувати за CPU або за пам'яттю; не вказуйте — cpu"},
            "top": {"type": "integer", "minimum": 1,
                    "description": "скільки процесів; не вказуйте — 5 (у знімку їх до 10)"}}, ["server"]),
        "get_folder_growth": ({
            "server": server,
            "period_h": {"type": "number", "minimum": 1,
                         "description": "за скільки годин до «зараз»; не вказуйте — 24 (тиждень — 168)"},
            "drive": drive}, ["server"]),
        "get_disk_forecast": ({"server": server, "drive": drive}, ["server"]),
        "get_timeline": ({
            "server": server, "start": start, "end": end,
            "types": {"type": "array", "items": {"type": "string", "enum": list(tools.TIMELINE_TYPES)},
                      "description": "лише ці типи записів: event — події журналу Windows; service — зміни "
                                     "стану служб і пулів IIS; check — зміни результату перевірок (http, "
                                     "cert, backup, sql_port); alert — алерти; не вказуйте — усі"},
            "source": {"type": "string",
                       "description": "лише записи, де джерело або текст містить цей рядок (без урахування "
                                      "регістру), наприклад SitePool, MSSQLSERVER, backup"},
            "limit": {"type": "integer", "minimum": 1,
                      "description": "скільки записів максимум (якщо більше — останні); не вказуйте — 50"}},
            ["server"]),
    }
    return [{"name": n, "description": description(n),
             "input_schema": {"type": "object", "properties": params[n][0], "required": params[n][1]}}
            for n in TOOL_NAMES]


def allowed_params(specs: list[dict], name: str) -> set[str]:
    """Які параметри LLM може передати цьому інструменту (усі інші код ігнорує)."""
    for s in specs:
        if s["name"] == name:
            return set(s["input_schema"]["properties"])
    return set()
