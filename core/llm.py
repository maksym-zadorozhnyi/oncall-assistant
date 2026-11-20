"""Виклик LLM — єдине місце в проєкті, де звертаємося до API моделі.

Зараз провайдер один — Claude (бібліотека anthropic, Messages API з tool use).
На етапі 3 тут з'явиться Groq; ядро (core/assistant.py) від цього не зміниться.

Формат розмови — як у Messages API: список повідомлень
{"role": "user" | "assistant", "content": текст або список блоків}.

Чого тут немає свідомо:
- temperature: claude-haiku-5-5 відповідає помилкою 400 на нестандартну температуру,
  а бібліотека anthropic 2.x цього параметра не приймає;
- thinking: у Haiku 5.5 «обдумування» адаптивне й увімкнене за замовчуванням;
  глибину задає effort (low / medium / high);
- кешування промпту: економія — частки цента (Haiku дешевий).

У тестах цей модуль не викликається: ядро отримує підставну функцію замість chat().
"""

import time
from dataclasses import dataclass

import anthropic

from core.config import get_llm_provider, get_setting


class LLMError(Exception):
    """Помилка виклику LLM. Текст — уже готове повідомлення для користувача."""


@dataclass
class ToolCall:
    """Запит LLM на виклик функції."""
    id: str                # id блоку tool_use — з ним повертаємо результат
    name: str
    input: dict


@dataclass
class LLMReply:
    """Одна відповідь LLM."""
    text: str              # текст (може бути порожнім, якщо модель лише просить функції)
    tool_calls: list       # список ToolCall; порожній — відповідь остаточна
    content: list          # повна відповідь моделі: повертаємо в розмову без змін
    stop_reason: str
    model: str
    tokens_in: int
    tokens_out: int
    ms: int


_client = None             # один клієнт на процес (з'єднання перевикористовуються)


def get_client(timeout_s: float, max_retries: int):
    """Клієнт Claude. Ключ — лише з .env через core/config.py."""
    global _client
    if _client is None:
        provider = get_llm_provider()
        if provider != "claude":
            raise LLMError(f"Провайдер «{provider}» ще не підключено (Groq — етап 3). "
                           f"У .env поставте LLM_PROVIDER=claude.")
        _client = anthropic.Anthropic(api_key=get_setting("ANTHROPIC_API_KEY"),
                                      timeout=timeout_s, max_retries=max_retries)
    return _client


def chat(system: str, messages: list, tools: list, model: str, effort: str = "low",
         max_tokens: int = 4000, timeout_s: float = 60, max_retries: int = 2) -> LLMReply:
    """Один запит до LLM. Повертає текст або запити на функції; помилка — LLMError.

    Бібліотека сама повторює запит при 429, 5xx і обриві зв'язку (max_retries разів).
    """
    client = get_client(timeout_s, max_retries)
    t0 = time.perf_counter()
    try:
        r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                   messages=messages, tools=tools,
                                   output_config={"effort": effort})
    # від окремого випадку до загального: тайм-аут — різновид помилки зв'язку
    except anthropic.APITimeoutError:
        raise LLMError(f"LLM зараз недоступна: не відповіла за {timeout_s:g} с. "
                       f"Спробуйте пізніше.") from None
    except anthropic.APIConnectionError:
        raise LLMError("LLM зараз недоступна: немає зв'язку з сервером API. "
                       "Спробуйте пізніше.") from None
    except anthropic.AuthenticationError:
        raise LLMError("LLM зараз недоступна: ключ ANTHROPIC_API_KEY не прийнято "
                       "(перевірте .env: python -m core.config).") from None
    except anthropic.RateLimitError:
        raise LLMError("LLM зараз недоступна: забагато запитів. Спробуйте за хвилину.") from None
    except anthropic.BadRequestError as e:
        raise LLMError(f"LLM відхилила запит (400): {e.message}") from None
    except anthropic.APIStatusError as e:
        raise LLMError(f"LLM зараз недоступна: помилка API {e.status_code}. "
                       f"Спробуйте пізніше.") from None
    ms = round((time.perf_counter() - t0) * 1000)

    if r.stop_reason == "refusal":
        raise LLMError("LLM відмовилась відповідати на це питання; спробуйте переформулювати.")
    if r.stop_reason == "max_tokens":
        raise LLMError(f"Відповідь LLM не вмістилась у ліміт {max_tokens} токенів; "
                       f"спробуйте вужче питання.")
    # Блоки відповіді читаємо за типом, а не за позицією: на початку можуть бути
    # блоки thinking («обдумування»; текст порожній, але блок треба повернути як є)
    text = "\n".join(b.text for b in r.content if b.type == "text").strip()
    calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
    return LLMReply(text=text, tool_calls=calls, content=list(r.content), stop_reason=r.stop_reason,
                    model=r.model, tokens_in=r.usage.input_tokens, tokens_out=r.usage.output_tokens, ms=ms)
