"""Переведення фізичних одиниць.

Уся арифметика тут детермінована й перевірювана: таблиця зберігає коефіцієнти
лінійного перетворення, а не готові значення й не lambda-функції. Так дані
лишаються даними, їх видно очима, і додати пару одиниць означає дописати рядок.
"""
import re
from typing import Optional

from .data import PHYSICS_CONVERSIONS, UNIT_ALIASES
from .text import format_number

# Літери, що можуть бути частиною слова. Потрібні, щоб шукати одиниці по межах
# слова: без цього «кал» знаходиться всередині «шкала», а «па» всередині «пару».
LETTER = "а-яіїєґёa-z"

NUMBER_PATTERN = re.compile(r"-?\d+(?:[  ]\d{3})*(?:[.,]\d+)?")


class UnitConverter:
    """Розпізнає одиниці в тексті запиту й виконує перетворення."""

    def __init__(self, table: dict = PHYSICS_CONVERSIONS, aliases: dict = UNIT_ALIASES):
        self.table = table
        self.aliases = aliases

    def convert(self, value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Лінійне перетворення value * k + b. None, якщо пари немає в таблиці."""
        pair = self.table.get((from_unit.strip(), to_unit.strip()))
        if pair is None:
            return None
        factor, offset = pair
        return value * factor + offset

    def find_units(self, text: str) -> list:
        """Знаходить одиниці по МЕЖАХ СЛОВА разом з позицією входження."""
        found = []
        for canonical, aliases in self.aliases.items():
            positions = []
            for alias in aliases:
                pattern = rf"(?<![{LETTER}]){re.escape(alias)}(?![{LETTER}])"
                positions += [m.start() for m in re.finditer(pattern, text)]
            if positions:
                found.append((min(positions), canonical))
        return sorted(found)

    @staticmethod
    def parse_number(text: str) -> Optional[tuple]:
        """Число разом з його позицією. Розуміє «1 000» і «0,5».

        Розділювач розрядів тут не дрібниця: сам інструмент друкує великі числа
        через пробіл, і без цієї підтримки власний вивід, поданий йому на вхід,
        читався б у тисячу разів меншим.
        """
        match = NUMBER_PATTERN.search(text)
        if not match:
            return None
        raw = match.group().replace(" ", "").replace(" ", "").replace(",", ".")
        return float(raw), match.start()

    def explain(self, query: str) -> str:
        """Розбирає запит і повертає готову відповідь для студента."""
        padded = f" {query.lower()} "
        parsed = self.parse_number(padded)
        if parsed is None:
            return "Не бачу числа в запиті. Приклад: '100 км/год у м/с'."
        value, number_pos = parsed

        units = self.find_units(padded)
        if len(units) < 2:
            pairs = ", ".join(f"{a}→{b}" for a, b in list(self.table)[:6])
            return (f"Не розпізнав обидві одиниці в '{query}'.\n"
                    f"Підтримуються, наприклад: {pairs}.")

        from_unit, to_unit = self._resolve_direction(units, number_pos, padded)
        result = self.convert(value, from_unit, to_unit)
        if result is None:
            return f"Конвертацію з '{from_unit}' у '{to_unit}' не підтримано."
        return (f"{format_number(value)} {from_unit} = "
                f"{format_number(result)} {to_unit}")

    @staticmethod
    def _resolve_direction(units: list, number_pos: int, text: str) -> tuple:
        """Визначає, яка одиниця вихідна, а яка цільова.

        Основне правило: величину називають одразу за числом («40 літрів»).
        Але число буває і в кінці («з кілограмів у грами: 2»), і тоді правило
        перевертало напрямок, даючи помилку в мільйон разів. Тому спершу
        дивимось на прийменник мети: те, що стоїть після «у/в/на», це ціль.
        """
        target_markers = (" у ", " в ", " на ", " до ")
        target_pos = max((text.rfind(m) for m in target_markers), default=-1)

        # Прийменник вказує на ціль лише тоді, коли між ним і одиницею немає числа.
        # У фразі «скільки кубічних метрів У 40 ЛІТРАХ» після прийменника йде
        # саме вихідна величина, і без цієї перевірки напрямок перевертався.
        if target_pos >= 0:
            after_target = [u for pos, u in units if pos > target_pos]
            number_between = target_pos < number_pos < (after_target and
                                                        min(pos for pos, u in units
                                                            if pos > target_pos) or 0)
            if after_target and not number_between:
                to_unit = after_target[0]
                rest = [u for _, u in units if u != to_unit]
                if rest:
                    return rest[0], to_unit

        after_number = [u for pos, u in units if pos > number_pos]

        from_unit = (after_number[0] if after_number
                     else min(units, key=lambda p: abs(p[0] - number_pos))[1])
        rest = [u for _, u in units if u != from_unit]
        return from_unit, (rest[0] if rest else from_unit)
