"""Тести логіки бота (bot/bot.py) без Telegram і без справжнього API. Запуск: python -m pytest

BotCore перевіряємо напряму: дозволені ID, команди, питання до асистента з
підставною LLM, запис у users / messages, черга в межах одного чату.
База — тимчасова з tests/test_tools.py (сервер t-1; SitePool стоїть D+50…D+70).
"""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

from bot import bot as B
from core import assistant as A
from test_assistant import MOMENT, FakeLLM, no_real_api, reply  # noqa: F401  (no_real_api — autouse)
from test_tools import D, db, local  # noqa: F401  (db — фікстура тимчасової бази)


def rows(db_path, sql):
    from core.db import db_session
    with db_session(db_path) as conn:
        return [dict(r) for r in conn.execute(sql).fetchall()]


def fake_answer(text="ок"):
    return SimpleNamespace(text=text, ms_total=5, model="claude-haiku-5-5", tokens_in=100, tokens_out=10)


def test_split_message():
    """Довга відповідь — частини ≤ ліміту, по рядках; разом — увесь текст."""
    text = "\n".join(f"рядок {i} " + "х" * 50 for i in range(200))
    parts = B.split_message(text, limit=1000)
    assert len(parts) > 1 and all(len(p) <= 1000 for p in parts)
    assert "\n".join(parts) == text
    long_line = "я" * 2500                                    # один рядок, довший за ліміт
    parts = B.split_message(long_line, limit=1000)
    assert [len(p) for p in parts] == [1000, 1000, 500] and "".join(parts) == long_line
    assert B.split_message("") == [""] and B.split_message("коротко") == ["коротко"]


def test_denied_user_no_llm(db):
    """Чужий ID — відмова без виклику LLM; у журналі — denied, у users — нічого."""
    def boom(*a, **k):
        raise AssertionError("LLM не можна викликати для чужого")
    core = B.BotCore({1}, ask_fn=boom, db_path=db)
    assert core.is_allowed(1) and not core.is_allowed(2) and not core.is_allowed(None)
    assert core.deny(2, "дай пароль") == B.DENY_TEXT
    rec = json.loads(A.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["denied"] is True and rec["user"] == "2" and rec["source"] == "telegram"
    assert rows(db, "SELECT * FROM users") == []


def test_question_answer_and_saved(db):
    """Питання -> ask() з моментом чату -> відповідь; users і messages записано."""
    fake = FakeLLM(reply(calls=[("get_status", {"server": "t-1"})]), reply("SitePool стоїть."))
    core = B.BotCore({7}, ask_fn=lambda q, s: A.ask(q, s, chat_fn=fake), db_path=db)
    core.cmd_moment(100, 7, MOMENT)
    parts = asyncio.run(core.on_text(100, 7, "Черговий", "Що з t-1?"))
    assert parts[0].startswith("SitePool стоїть.") and "Дивився: стан (t-1)" in parts[0]
    session = core.chats[100].session
    assert session.source == "telegram" and session.user == "7" and session.now == MOMENT
    assert rows(db, "SELECT telegram_id, name, notify FROM users") == [
        {"telegram_id": 7, "name": "Черговий", "notify": 1}]
    msgs = rows(db, "SELECT role, channel, content, input_type, llm_provider, tokens_in FROM messages ORDER BY id")
    assert [m["role"] for m in msgs] == ["user", "assistant"] and all(m["channel"] == "telegram" for m in msgs)
    assert msgs[0]["content"] == "Що з t-1?" and msgs[0]["input_type"] == "text"
    assert msgs[1]["content"] == parts[0] and msgs[1]["tokens_in"] == 200    # 2 виклики підставної LLM


def test_moment_and_reset_commands(db):
    """/moment: показати, дата, зсув, помилка; /reset очищає історію, момент лишає."""
    core = B.BotCore({1}, ask_fn=lambda q, s: A.ask(q, s, chat_fn=FakeLLM(reply("ок"))), db_path=db)
    assert core.cmd_moment(5, 1, None).startswith("Зараз: живий режим")
    assert "вівторок, 06.01.2026 03:00" in core.cmd_moment(5, 1, MOMENT)
    assert "06.01.2026 04:00" in core.cmd_moment(5, 1, "+1h")
    assert core.cmd_moment(5, 1, "-30d").startswith("Не вдалося змінити момент")
    asyncio.run(core.on_text(5, 1, None, "привіт"))
    core.cmd_moment(5, 1, "-1h")                              # розмова була — буде примітка
    assert "Момент змінено" in core.chats[5].session.moment_note
    assert "Історію розмови очищено" in core.cmd_reset(5, 1)
    s = core.chats[5].session
    assert s.history == [] and s.moment_note is None and s.now == MOMENT


def test_status_without_llm(db):
    """/status — шаблон з get_status, без LLM; записується в messages."""
    def boom(*a, **k):
        raise AssertionError("/status не викликає LLM")
    core = B.BotCore({1}, ask_fn=boom, db_path=db)
    core.cmd_moment(9, 1, local(D + 60))
    text = core.cmd_status(9, 1, "Черговий")
    assert text.startswith("Момент: 06.01 03:00 (історія)")
    assert "t-1 — дані на 06.01 03:00" in text and "CPU 95,0 %" in text
    assert "Не працює: SitePool (stopped з 06.01 02:50)" in text
    assert "сайт НЕ ok (HTTP 503" in text and "бекап" in text
    assert "застарілі" not in text
    msgs = rows(db, "SELECT role, content, llm_provider FROM messages ORDER BY id")
    assert msgs[0]["content"] == "/status" and msgs[1]["llm_provider"] is None
    late = B.BotCore({1}, ask_fn=boom, db_path=db)
    late.cmd_moment(9, 1, local(D + 1440 + 600))               # після кінця даних
    assert "⚠ застарілі" in late.cmd_status(9, 1)


def test_guards_without_llm(db):
    """Невідома команда, задовге питання, порожній текст — відповідь без LLM; збій — повідомлення."""
    calls = []
    core = B.BotCore({1}, ask_fn=lambda q, s: calls.append(q) or fake_answer(), db_path=db)
    assert asyncio.run(core.on_text(1, 1, None, "/foo")) == [B.UNKNOWN_COMMAND]
    assert asyncio.run(core.on_text(1, 1, None, "я" * (B.MAX_QUESTION + 1))) == [B.TOO_LONG]
    assert asyncio.run(core.on_text(1, 1, None, "   ")) == [B.ONLY_TEXT]
    assert calls == []

    def broken(q, s):
        raise RuntimeError("щось зламалось")
    bad = B.BotCore({1}, ask_fn=broken, db_path=db)
    assert asyncio.run(bad.on_text(2, 1, None, "Що з t-1?")) == [B.INTERNAL_ERROR]


def test_one_chat_waits_other_chats_parallel(db):
    """Два питання одного чату — по черзі; двох різних чатів — одночасно."""
    guard, state = threading.Lock(), {"now": 0, "max": 0}

    def slow(q, s):
        with guard:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.3)
        with guard:
            state["now"] -= 1
        return fake_answer(q)

    async def both(chat_a, chat_b):
        core = B.BotCore({1}, ask_fn=slow, db_path=db)
        return await asyncio.gather(core.on_text(chat_a, 1, None, "перше"), core.on_text(chat_b, 1, None, "друге"))

    assert asyncio.run(both(1, 1)) == [["перше"], ["друге"]] and state["max"] == 1
    state["max"] = 0
    asyncio.run(both(1, 2))
    assert state["max"] == 2


def test_moment_waits_for_running_answer(db):
    """/moment під час відповіді чекає її кінця: відповідь — на старий момент, в історії
    з ним же; новий момент — після, з приміткою «Момент змінено»."""
    seen = {}

    def slow(q, s):
        time.sleep(0.3)
        seen["now"] = s.now                       # момент, на який реально відповідали
        s.history += [{"role": "user", "content": q}, {"role": "assistant", "content": "ок"}]
        return fake_answer()

    async def scenario():
        core = B.BotCore({1}, ask_fn=slow, db_path=db)
        core.cmd_moment(3, 1, MOMENT)
        answer = asyncio.create_task(core.on_text(3, 1, None, "Що з t-1?"))
        await asyncio.sleep(0.05)                 # відповідь уже йде
        moment_reply = await core.moment(3, 1, "+1h")
        await answer
        return core, moment_reply

    core, moment_reply = asyncio.run(scenario())
    assert seen["now"] == MOMENT                  # зміна моменту не вклинилась у відповідь
    s = core.chats[3].session
    assert s.now == local(D + 120) and "06.01.2026 04:00" in moment_reply
    assert "Момент змінено" in s.moment_note


def test_save_failure_still_answers(db, tmp_path):
    """База для users / messages недоступна — відповідь усе одно є, помилка — в журналі."""
    broken_db = tmp_path / "empty.db"             # файл без таблиць: INSERT упаде
    core = B.BotCore({1}, ask_fn=lambda q, s: fake_answer("відповідь"), db_path=broken_db)
    assert asyncio.run(core.on_text(1, 1, "Черговий", "Що з t-1?")) == ["відповідь"]
    rec = json.loads(A.LOG_PATH.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["source"] == "telegram" and rec["error"].startswith("запис у users / messages")
    text = core.cmd_status(2, 1)                  # /status теж відповідає, хоч запис не вдався
    assert "t-1 — дані на" in text


def test_status_error_is_reported(db, monkeypatch):
    """Збій get_status — INTERNAL_ERROR, а не мовчання."""
    def broken(**kw):
        raise RuntimeError("база зламалась")
    monkeypatch.setattr(B.tools, "get_status", broken)
    assert B.BotCore({1}, db_path=db).cmd_status(1, 1) == B.INTERNAL_ERROR


def test_help_data_bounds(db):
    """/help — межі даних з list_servers; «свіжих даних немає» — лише коли дані старі."""
    import pandas as pd
    from core import tools
    servers = tools.list_servers()["servers"]
    last = pd.Timestamp(local(2 * 1440 - 1), tz=tools.LOCAL_TZ)
    text = B.help_text(servers, clock=last)
    assert "Сервери: t-1. Дані в базі: з 05.01.2026 02:00 до 07.01.2026 01:59." in text
    assert "Свіжих даних зараз немає" not in text
    assert "Свіжих даних зараз немає" in B.help_text(servers, clock=last + pd.Timedelta(hours=1))
    assert "18.11.2026" not in B.HELP_TEXT                       # дата не вписана в текст
    assert "Даних у базі ще немає" in B.help_text([])


def test_bot_db_busy_timeout(db):
    """З'єднання бота чекають на зайняту базу DB_TIMEOUT_S секунд (PRAGMA busy_timeout)."""
    from core.db import db_session
    with db_session(db, B.DB_TIMEOUT_S) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == B.DB_TIMEOUT_S * 1000


def test_router_builds():
    """Обгортка aiogram збирається (без мережі)."""
    router = B.build_router(B.BotCore({1}))
    assert router.message.handlers
