"""Telegram-бот асистента чергового інженера (aiogram 3, long polling).

Запуск:
    python -m bot.bot             # бот працює, доки не натиснути Ctrl+C
    python -m bot.bot --check     # лише перевірити токен (getMe) і вийти

Як влаштовано:
- BotCore — уся логіка без Telegram (дозволені ID, сесії чатів, команди,
  питання до асистента, запис у users / messages). Її перевіряють тести.
- build_router() — тонка обгортка aiogram: бере з повідомлення chat_id,
  user_id і текст, кличе BotCore і надсилає відповідь.
- Бот нічого не виконує на серверах: лише читає дані через core/assistant.py.
- Сесії — у пам'яті; після перезапуску — заново, у живому режимі (now = None).
"""
import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field

from core import assistant as A
from core import tools
from core.db import DB_PATH, db_session, now_utc

MAX_MESSAGE = 4096       # ліміт Telegram на одне повідомлення
MAX_QUESTION = 2000      # довше питання не надсилаємо в LLM (кожен виклик коштує)

DB_TIMEOUT_S = 5        # бот чекає на зайняту базу (пише симулятор / ETL) не довше 5 с

HELP_TEXT = """Я — асистент чергового інженера. Відповідаю за даними моніторингу; на серверах нічого не виконую, лише раджу.

Питайте звичайними словами, наприклад:
– Що з серверами?
– Коли заповниться E: на sim-2?
– А що там росте?
– Чому гальмує sim-1?
– Що сталося з сайтом за останню годину?

Команди:
/status — короткий стан усіх серверів (без LLM, швидко)
/moment — показати момент «зараз»
/moment 10.11.2026 08:00 (або 10.11 08:00, 2026-11-10 08:00) — відповідати так, наче зараз цей момент в історії (місцевий час)
/moment +1h, /moment -1d, /moment +30m — зсунути момент
/moment live — живий режим (справжній поточний час)
/reset — почати розмову заново

За замовчуванням — живий режим."""

DENY_TEXT = "Вибачте, цей бот лише для чергових інженерів зі списку доступу."
ONLY_TEXT = "Поки що я розумію лише текстові повідомлення."
UNKNOWN_COMMAND = "Не знаю такої команди. Список — /help."
TOO_LONG = f"Питання задовге (більше {MAX_QUESTION} символів). Спитайте коротше, будь ласка."
INTERNAL_ERROR = "Внутрішня помилка бота; деталі — в журналі. Спробуйте ще раз."

log = logging.getLogger("bot")


# ---------------------------------------------------------------------------
# Допоміжне: розбиття довгої відповіді, формат чисел
# ---------------------------------------------------------------------------

def split_message(text: str, limit: int = MAX_MESSAGE) -> list[str]:
    """Розбити текст на частини не довші за limit — по рядках, а задовгий рядок — по символах."""
    parts, current = [], ""
    for line in text.split("\n"):
        while len(line) > limit:                       # рядок сам довший за ліміт
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            parts.append(current)
            current = line
        else:
            current = candidate
    if current or not parts:
        parts.append(current)
    return parts


def n(value, digits: int = 1) -> str:
    """Число з десятковою комою: 77.04 -> «77,0»; None -> «—»."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# /status — шаблон без LLM (числа лише з get_status)
# ---------------------------------------------------------------------------

def status_text(result: dict, moment: str | None) -> str:
    """Короткий стан усіх серверів з результату tools.get_status() — лише факти, без висновків."""
    head = f"Момент: {A.short(moment)} (історія)" if moment else "Живий режим"
    lines = [head]
    for s in result.get("servers", []):
        lines.append("")
        if "error" in s:
            lines.append(f"{s['server']}: {s['error']}")
            continue
        until = A.short(s["data_until"]["ts_local"]) if s.get("data_until") else "невідомо"
        lines.append(f"{s['server']} — дані на {until}" +
                     (f" (⚠ застарілі, {tools.age_text(s['data_age_h'])} тому)" if s.get("stale") else ""))
        m = s.get("metrics", {})
        lines.append(f"– CPU {n(m.get('cpu_pct'))} %, RAM {n(m.get('ram_pct'))} %")
        if s.get("disks"):
            lines.append("– Диски: " + ", ".join(
                f"{d['drive']} {n(d['used_pct'])} % (вільно {n(d['free_gb'])} ГБ)" for d in s["disks"]))
        stopped = [x for x in s.get("services", []) if x["status"] != "running"]
        if stopped:
            lines.append("– Не працює: " + ", ".join(
                f"{x['name']} ({x['status']} з {A.short(x.get('since_local'))})" for x in stopped))
        else:
            lines.append("– Служби і пули: усі працюють")
        c = s.get("checks", {})
        checks = []
        if "http" in c:
            checks.append(f"сайт {'ok' if c['http']['ok'] else 'НЕ ok'} ({c['http']['details']}, "
                          f"{n(c['http']['response_ms'], 0)} мс)")
        if "sql_port" in c:
            checks.append(f"SQL-порт {'ok' if c['sql_port']['ok'] else 'НЕ ok'}")
        if "backup" in c:
            age = c["backup"].get("age_h_now")
            checks.append(f"бекап {n(age)} год тому" if age is not None else
                          f"бекап {n(c['backup']['age_h'])} год тому на час перевірки")
        if "cert" in c:
            checks.append(f"сертифікат ще {n(c['cert']['days_left'], 0)} дн")
        if checks:
            lines.append("– Перевірки: " + "; ".join(checks))
    return "\n".join(lines)


def help_text(servers: list, clock=None) -> str:
    """/help: постійний текст + межі даних з list_servers (а не вписана дата).
    clock — «справжній» поточний час (для тестів)."""
    first, last = A.data_bounds(servers)
    if last is None:
        return HELP_TEXT + "\nДаних у базі ще немає."
    names = ", ".join(s["name"] for s in servers)
    text = (f"{HELP_TEXT}\n\nСервери: {names}. Дані в базі: з {first.strftime('%d.%m.%Y %H:%M')} "
            f"до {last.strftime('%d.%m.%Y %H:%M')}.")
    now = clock if clock is not None else A.local_now()
    if (now - last).total_seconds() / 60 > tools.STALE_MIN:
        text += " Свіжих даних зараз немає, тож у живому режимі відповіді будуть з позначкою «дані застарілі»; для історії задайте /moment."
    return text


# ---------------------------------------------------------------------------
# Логіка бота без Telegram
# ---------------------------------------------------------------------------

@dataclass
class Chat:
    """Сесія одного чату: момент і історія (A.Session) + блокування, щоб два питання
    того самого чату не виконувались одночасно і не перемішали історію."""
    session: A.Session
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BotCore:
    def __init__(self, allowed_ids: set[int], ask_fn=None, db_path=None):
        self.allowed = set(allowed_ids)
        self.ask_fn = ask_fn or A.ask          # у тестах — ask з підставною LLM
        self.db_path = db_path or DB_PATH
        self.chats: dict[int, Chat] = {}

    # --- доступ ---

    def is_allowed(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.allowed

    def deny(self, user_id: int | None, text: str | None) -> str:
        """Чужий користувач: коротка відмова без LLM і запис у журнал (лише ID і початок тексту)."""
        A.write_log({"ts": now_utc(), "source": "telegram", "user": str(user_id), "denied": True,
                     "question": (text or "")[:100]})
        return DENY_TEXT

    def chat(self, chat_id: int, user_id: int) -> Chat:
        if chat_id not in self.chats:
            self.chats[chat_id] = Chat(A.Session(now=None, source="telegram", user=str(user_id)))
        return self.chats[chat_id]

    # --- users / messages ---

    def save_user(self, telegram_id: int, name: str | None) -> int:
        """Користувач у таблиці users (notify не чіпаємо); повертає users.id."""
        with db_session(self.db_path, DB_TIMEOUT_S) as conn:
            conn.execute("INSERT INTO users (telegram_id, name) VALUES (?, ?) "
                         "ON CONFLICT(telegram_id) DO UPDATE SET name = excluded.name", (telegram_id, name))
            return conn.execute("SELECT id FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()[0]

    def save_dialog(self, user_db_id: int, question: str, asked_at: str, answer: str,
                    latency_ms: int, model: str | None = None, tokens_in=None, tokens_out=None) -> None:
        """Питання і відповідь — два рядки messages (канал telegram)."""
        provider = None
        if model:
            provider = "claude" if model.startswith("claude") else model
        with db_session(self.db_path, DB_TIMEOUT_S) as conn:
            conn.execute("INSERT INTO messages (user_id, channel, ts, role, content, input_type) "
                         "VALUES (?, 'telegram', ?, 'user', ?, 'text')", (user_db_id, asked_at, question))
            conn.execute("INSERT INTO messages (user_id, channel, ts, role, content, latency_ms, llm_provider, "
                         "tokens_in, tokens_out) VALUES (?, 'telegram', ?, 'assistant', ?, ?, ?, ?, ?)",
                         (user_db_id, now_utc(), answer, latency_ms, provider, tokens_in, tokens_out))

    def save_safely(self, user_id: int, name: str | None, question: str, asked_at: str, answer: str,
                    latency_ms: int, model=None, tokens_in=None, tokens_out=None) -> bool:
        """Запис у users / messages, який не зупиняє відповідь: якщо база зайнята чи
        зламана — помилка в журнал (logging + assistant.jsonl), а відповідь усе одно піде."""
        try:
            self.save_dialog(self.save_user(user_id, name), question, asked_at, answer, latency_ms,
                             model, tokens_in, tokens_out)
            return True
        except Exception as e:
            log.exception("не вдалося записати діалог у базу")
            A.write_log({"ts": now_utc(), "source": "telegram", "user": str(user_id),
                         "error": f"запис у users / messages: {type(e).__name__}: {e}"})
            return False

    # --- команди (без LLM) ---

    def cmd_moment(self, chat_id: int, user_id: int, arg: str | None) -> str:
        chat = self.chat(chat_id, user_id)
        if not arg or not arg.strip():
            return (f"Зараз: {A.moment_text(chat.session.now)}.\n"
                    "Змінити: /moment 2026-11-10 08:00, /moment +1h, /moment -1d, /moment live")
        try:
            moment, warning = A.parse_moment(arg, chat.session.now)
        except A.MomentError as e:
            return f"Не вдалося змінити момент: {e}"
        A.set_moment(chat.session, moment)          # + примітка в історії, якщо розмова вже була
        text = f"Момент: {A.moment_text(moment)}."
        return text + (f"\n⚠ {warning}" if warning else "")

    def cmd_reset(self, chat_id: int, user_id: int) -> str:
        chat = self.chat(chat_id, user_id)
        chat.session.history = []
        chat.session.moment_note = None
        return f"Історію розмови очищено. Момент лишився: {A.moment_text(chat.session.now)}."

    def cmd_status(self, chat_id: int, user_id: int, name: str | None = None) -> str:
        """Стан усіх серверів за шаблоном — без LLM. Блокує (SQLite): викликати через to_thread.
        Будь-яка помилка — INTERNAL_ERROR, а не мовчання бота."""
        chat = self.chat(chat_id, user_id)
        asked_at = now_utc()
        start = time.perf_counter()
        try:
            text = status_text(tools.get_status(now=chat.session.now), chat.session.now)
        except Exception:
            log.exception("помилка /status")
            return INTERNAL_ERROR
        ms = round((time.perf_counter() - start) * 1000)
        self.save_safely(user_id, name, "/status", asked_at, text, ms)
        return text

    def cmd_help(self, user_id: int, name: str | None) -> str:
        """/help з межами даних; користувача — в users (помилка запису не заважає)."""
        try:
            self.save_user(user_id, name)
        except Exception:
            log.exception("не вдалося записати користувача")
        try:
            return help_text(tools.list_servers().get("servers", []))
        except Exception:
            log.exception("помилка /help")
            return HELP_TEXT

    # Команди, що змінюють сесію, — під блокуванням чату: якщо зараз іде відповідь,
    # момент або історія зміняться лише після неї (інакше відповідь записалася б
    # в історію вже з іншим моментом)

    async def moment(self, chat_id: int, user_id: int, arg: str | None) -> str:
        chat = self.chat(chat_id, user_id)
        async with chat.lock:
            return self.cmd_moment(chat_id, user_id, arg)

    async def reset(self, chat_id: int, user_id: int) -> str:
        chat = self.chat(chat_id, user_id)
        async with chat.lock:
            return self.cmd_reset(chat_id, user_id)

    # --- звичайне питання ---

    def answer_sync(self, chat: Chat, user_id: int, name: str | None, text: str) -> str:
        """Питання до асистента + запис у users / messages. Блокує (API, SQLite) —
        викликати через asyncio.to_thread."""
        asked_at = now_utc()
        ans = self.ask_fn(text, chat.session)
        self.save_safely(user_id, name, text, asked_at, ans.text, ans.ms_total,
                         ans.model, ans.tokens_in, ans.tokens_out)
        return ans.text

    async def on_text(self, chat_id: int, user_id: int, name: str | None, text: str) -> list[str]:
        """Текст від дозволеного користувача -> частини відповіді (кожна ≤ 4096 символів)."""
        text = (text or "").strip()
        if not text:
            return [ONLY_TEXT]
        if text.startswith("/"):
            return [UNKNOWN_COMMAND]
        if len(text) > MAX_QUESTION:
            return [TOO_LONG]
        chat = self.chat(chat_id, user_id)
        async with chat.lock:              # друге питання того самого чату чекає першого
            try:
                reply = await asyncio.to_thread(self.answer_sync, chat, user_id, name, text)
            except Exception:
                log.exception("помилка під час відповіді")
                return [INTERNAL_ERROR]
        return split_message(reply)


# ---------------------------------------------------------------------------
# aiogram: тонка обгортка над BotCore
# ---------------------------------------------------------------------------

def build_router(core: BotCore):
    from aiogram import F, Router
    from aiogram.enums import ChatType
    from aiogram.filters import Command, CommandObject, CommandStart
    from aiogram.types import Message
    from aiogram.utils.chat_action import ChatActionSender

    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)     # лише особисті чати

    def who(m: Message) -> tuple[int, int | None, str | None]:
        u = m.from_user
        return m.chat.id, (u.id if u else None), (u.full_name if u else None)

    async def send(m: Message, text: str) -> None:
        for part in split_message(text):
            await m.answer(part)                                  # звичайний текст, без розмітки

    @router.message(CommandStart())
    @router.message(Command("help"))
    async def on_help(m: Message) -> None:
        chat_id, uid, name = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, m.text))
            return
        await send(m, await asyncio.to_thread(core.cmd_help, uid, name))

    @router.message(Command("moment"))
    async def on_moment(m: Message, command: CommandObject) -> None:
        chat_id, uid, _ = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, m.text))
            return
        await send(m, await core.moment(chat_id, uid, command.args))   # чекає, якщо йде відповідь

    @router.message(Command("reset"))
    async def on_reset(m: Message) -> None:
        chat_id, uid, _ = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, m.text))
            return
        await send(m, await core.reset(chat_id, uid))

    @router.message(Command("status"))
    async def on_status(m: Message) -> None:
        chat_id, uid, name = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, m.text))
            return
        core.chat(chat_id, uid)                    # сесію створюємо в головному потоці
        try:
            text = await asyncio.to_thread(core.cmd_status, chat_id, uid, name)
        except Exception:
            log.exception("помилка /status")
            text = INTERNAL_ERROR
        await send(m, text)

    @router.message(F.text)
    async def on_text(m: Message) -> None:
        chat_id, uid, name = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, m.text))
            return
        async with ChatActionSender.typing(chat_id=chat_id, bot=m.bot):   # «друкує…», поки чекаємо
            parts = await core.on_text(chat_id, uid, name, m.text)
        for part in parts:
            await m.answer(part)

    @router.message()
    async def on_other(m: Message) -> None:                     # голос, фото, файли — пізніше
        _, uid, _ = who(m)
        if not core.is_allowed(uid):
            await m.answer(core.deny(uid, None))
            return
        await m.answer(ONLY_TEXT)

    return router


async def run(check_only: bool) -> None:
    from aiogram import Bot, Dispatcher

    from core.config import get_allowed_ids, get_setting
    token = get_setting("TELEGRAM_BOT_TOKEN")
    bot = Bot(token)
    try:
        me = await bot.get_me()
        print(f"Токен працює: бот @{me.username}")
        if check_only:
            return
        allowed = get_allowed_ids()
        dp = Dispatcher()
        dp.include_router(build_router(BotCore(allowed)))
        print(f"Дозволених користувачів: {len(allowed)}. Зупинити — Ctrl+C.")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Telegram-бот асистента чергового інженера")
    p.add_argument("--check", action="store_true", help="лише перевірити токен і вийти")
    args = p.parse_args()
    # INFO — без адрес запитів; детальні логи (DEBUG) не вмикаємо: в адресах Telegram є токен
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args.check))
    except KeyboardInterrupt:
        print("Бот зупинено.")


if __name__ == "__main__":
    main()
