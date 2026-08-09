"""Агент StudyMate: модель, системний промпт і діалог з контекстом.

Агент створюється ліниво, тому імпорт пакета не вимагає ані ключа, ані мережі:
інструменти це чисті функції, і їх треба вміти тестувати офлайн.
"""
from dataclasses import dataclass, field
from typing import Optional

from .tools import TOOLS

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 900

ERROR_PREFIX = "⚠️ Помилка виклику агента:"

SYSTEM_PROMPT = """Ти StudyMate, освітній асистент для студентів з математики, фізики та хімії.

## Роль
Ти не видаєш відповіді, ти пояснюєш. Студент має зрозуміти, звідки береться результат.
Пояснюй простою мовою, як однокласник, який добре розібрався в темі.

## Правила виклику інструментів
1. formula_lookup — ЗАВЖДИ, коли питають формулу. Формулу з власної пам'яті наводити заборонено.
2. check_formula_for_task — ОБОВ'ЯЗКОВО, коли студент описує свою задачу і збирається
   застосувати формулу. Формула може бути правильною і при цьому непридатною саме тут.
3. convert_units — для будь-якого переведення одиниць, навіть очевидного.
4. plan_exam_prep — коли планують підготовку.

## Жорсткі межі
- Якщо інструмент не знайшов формулу, ТАК І СКАЖИ. Краще чесне «немає в базі»,
  ніж правдоподібна вигадка: студент не зможе відрізнити одне від одного.
- Якщо в картці є умова застосовності, ОБОВ'ЯЗКОВО передай її студенту.
- Якщо інструмент повернув кілька варіантів, не обирай сам: перепитай студента.
- Якщо перевірка показала «не перевірено», так і скажи: це не те саме, що «підходить».
- Не розв'язуй контрольну за студента без пояснення кроків.
- Якщо питання не про точні науки, ввічливо поясни свій профіль.

## Формат
Стисло: спочатку суть, потім пояснення, за потреби приклад.
"""


@dataclass
class Turn:
    """Один хід діалогу з поясненням, як саме отримано відповідь."""

    answer: str
    history: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.answer.startswith(ERROR_PREFIX)


class StudyMateAgent:
    """Обгортка над агентом LangChain: створення, виклик і облік інструментів."""

    def __init__(self, model_name: str = LLM_MODEL, temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS, tools: list = None,
                 system_prompt: str = SYSTEM_PROMPT):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or TOOLS
        self.system_prompt = system_prompt
        self._agent = None

    @property
    def agent(self):
        """Агент створюється при першому зверненні, а не при імпорті."""
        if self._agent is None:
            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model=self.model_name, temperature=self.temperature,
                               max_tokens=self.max_tokens)
            self._agent = create_agent(model=model, tools=self.tools,
                                       system_prompt=self.system_prompt)
        return self._agent

    @staticmethod
    def extract_tool_calls(messages: list) -> list:
        """Назви викликаних інструментів.

        Без цього неможливо відрізнити відповідь з бази від відповіді
        з пам'яті моделі, а саме походження відповіді тут і є предметом контролю.
        """
        called = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = (call.get("name") if isinstance(call, dict)
                        else getattr(call, "name", None))
                if name:
                    called.append(name)
        return called

    def ask(self, user_input: str, history: Optional[list] = None,
            verbose: bool = False) -> Turn:
        """Один хід діалогу з урахуванням контексту.

        Історія передається в агента цілком на кожному кроці: саме це робить
        можливими уточнення на кшталт «а якщо з даху?», у яких немає ані чисел,
        ані назви формули.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_input})
        try:
            result = self.agent.invoke({"messages": messages})
        except Exception as error:
            return Turn(f"{ERROR_PREFIX} {type(error).__name__}: {error}", messages, [])

        updated = result["messages"]
        turn = Turn(updated[-1].content, updated,
                    self.extract_tool_calls(updated[len(messages):]))
        if verbose:
            print(f"👤 {user_input}")
            if turn.tools_used:
                print(f"🔧 {', '.join(turn.tools_used)}")
            print(f"🤖 {turn.answer}\n" + "─" * 78)
        return turn
