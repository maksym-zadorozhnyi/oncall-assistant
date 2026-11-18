"""Налаштування проєкту з файлу .env.

Усі модулі беруть ключі і параметри лише звідси:
    from core.config import get_setting
    api_key = get_setting("ANTHROPIC_API_KEY")

Перевірка .env:
    python -m core.config
Показує лише перші 4 символи кожного значення, тож вивід безпечно
показувати іншим.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Читаємо .env один раз під час імпорту модуля.
# Якщо змінна вже задана в системі (наприклад, у Docker), вона має перевагу.
load_dotenv(ENV_PATH)

# Усі змінні, які очікує проєкт, і для чого вони
SETTINGS = {
    "ANTHROPIC_API_KEY": "ключ Claude API",
    "GROQ_API_KEY": "ключ Groq (Whisper і порівняння LLM)",
    "TELEGRAM_BOT_TOKEN": "токен Telegram-бота",
    "TELEGRAM_ALLOWED_IDS": "Telegram ID користувачів через кому",
    "LLM_PROVIDER": "провайдер LLM: claude або groq",
    "INGEST_API_KEY": "ключ агента для REST API",
    "DASHBOARD_PASSWORD": "пароль входу в дашборд",
}

LLM_PROVIDERS = ("claude", "groq")


class SettingError(Exception):
    """Змінну не задано або задано неправильно."""


def get_setting(name: str, default: str | None = None) -> str:
    """Повертає значення змінної з .env.

    Якщо змінна порожня і default не задано — зрозуміла помилка
    замість загадкового збою десь далі в коді.
    """
    value = os.getenv(name, "").strip()
    if value:
        return value
    if default is not None:
        return default
    hint = SETTINGS.get(name, "")
    raise SettingError(
        f"Змінну {name} ({hint}) не задано. Впишіть її у файл {ENV_PATH} "
        f"(шаблон — .env.example)."
    )


def get_allowed_ids() -> set[int]:
    """Telegram ID з TELEGRAM_ALLOWED_IDS як множина чисел."""
    raw = get_setting("TELEGRAM_ALLOWED_IDS")
    try:
        return {int(part) for part in raw.split(",") if part.strip()}
    except ValueError:
        raise SettingError(
            "TELEGRAM_ALLOWED_IDS має містити лише числа через кому, "
            "наприклад: 123456789,987654321"
        ) from None


def get_llm_provider() -> str:
    """Назва провайдера LLM: claude або groq."""
    provider = get_setting("LLM_PROVIDER", default="claude").lower()
    if provider not in LLM_PROVIDERS:
        raise SettingError(f"LLM_PROVIDER має бути claude або groq, а не «{provider}».")
    return provider


def mask(value: str) -> str:
    """Перші 4 символи і довжина — щоб упізнати ключ, не розкриваючи його."""
    return f"{value[:4]}… ({len(value)} симв.)"


def main() -> int:
    if not ENV_PATH.exists():
        print(f"Файлу {ENV_PATH} немає. Скопіюйте .env.example у .env і впишіть значення.")
        return 1

    print(f"Файл: {ENV_PATH}")
    problems = 0
    for name, hint in SETTINGS.items():
        value = os.getenv(name, "").strip()
        if value:
            print(f"  OK         {name:<22} {mask(value)}")
        else:
            print(f"  ПОРОЖНЯ    {name:<22} ({hint})")
            problems += 1

    # Додаткові перевірки формату
    for check in (get_allowed_ids, get_llm_provider):
        try:
            check()
        except SettingError as e:
            # Порожні змінні вже показано вище — не дублюємо
            if "не задано" not in str(e):
                print(f"  ПОМИЛКА    {e}")
                problems += 1

    print("Усе гаразд." if problems == 0 else f"Проблем: {problems}.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
