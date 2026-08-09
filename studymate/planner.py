"""Планування підготовки до іспиту.

Чистий алгоритм без моделі: розбір запиту, розрахунок навантаження і чесна
відмова, коли план нереалістичний. Модель тут не потрібна, а її участь
лише додала б варіативності там, де потрібна арифметика.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional

NUM = r"(-?\d+(?:[.,]\d+)?)"
MAX_REALISTIC_HOURS = 12
SHOWN_DAYS = 7


@dataclass
class StudyPlanInput:
    """Розібрані параметри запиту."""

    topics: Optional[float] = None
    days: Optional[float] = None
    hours: Optional[float] = None

    @property
    def missing(self) -> list:
        return [name for name, value in
                (("тем", self.topics), ("днів", self.days), ("годин на день", self.hours))
                if value is None]


class StudyPlanner:
    """Будує план підготовки за кількістю тем, днів і годин."""

    DIFFICULTY = {"важка": 2.5, "середня": 1.5, "легка": 1.0}

    @staticmethod
    def _first_match(patterns: list, text: str) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1).replace(",", "."))
        return None

    def parse(self, query: str) -> StudyPlanInput:
        """Дістає числа за ключовими словами, а не за порядком у рядку.

        У живих запитах порядок довільний: «чи вистачить 7 днів щоб вивчити
        20 тем» не має читатися як 7 тем за 20 днів.
        """
        text = query.lower()
        weeks = self._first_match([rf"{NUM}\s*тижн"], text)
        if weeks is None and re.search(r"\bтиждень\b", text):
            weeks = 1.0
        days = weeks * 7 if weeks is not None else self._first_match(
            [rf"{NUM}\s*(?:дн|день|доб|діб)", rf"дн\w*\s*[-:]?\s*{NUM}"], text)
        return StudyPlanInput(
            # Прикметники між числом і словом: «20 важких тем».
            topics=self._first_match([rf"{NUM}\s*(?:\w+\s+){{0,2}}тем",
                                      rf"тем\w*\s*[-:]?\s*{NUM}"], text),
            days=days,
            # Явне «на день» має пріоритет, інакше «24 години до іспиту»
            # читалося б як 24 години щодня.
            hours=self._first_match([
                rf"{NUM}\s*год\w*\s*(?:на день|щодня|за день|в день)",
                rf"{NUM}\s*(?:год|час)", rf"год\w*\s*[-:]?\s*{NUM}",
            ], text),
        )

    @staticmethod
    def detect_difficulty(text: str) -> str:
        """Складність матеріалу за формулюваннями студента.

        Порядок перевірок принциповий: «нескладн» містить «складн», тому
        заперечну форму шукаємо першою. Голого «склад» тут немає: воно ловило
        звичайне «складати іспит» і мовчки піднімало оцінку на третину.
        """
        lowered = text.lower()
        if "нескладн" in lowered or "легк" in lowered:
            return "легка"
        if "важк" in lowered or "складн" in lowered:
            return "важка"
        return "середня"

    def build(self, query: str) -> str:
        """Готовий план або пояснення, яких даних бракує."""
        parsed = self.parse(query)
        if parsed.missing:
            return (f"Бракує даних: {', '.join(parsed.missing)}. "
                    "Приклад: '20 тем, 10 днів, 3 години на день'.")

        topics, days, hours = int(parsed.topics), int(parsed.days), parsed.hours
        if topics <= 0 or days <= 0 or hours <= 0:
            return "Усі значення мають бути більші за нуль."
        if hours > MAX_REALISTIC_HOURS:
            return (f"{hours:g} годин на день нереалістично: на сон і відпочинок "
                    f"не лишається часу.\nБільше за {MAX_REALISTIC_HOURS} годин "
                    "планувати немає сенсу.")

        difficulty = self.detect_difficulty(query)
        hours_per_topic = self.DIFFICULTY[difficulty]

        # Округлюємо ОДИН раз і далі рахуємо різницю за округленими значеннями,
        # інакше таблиця суперечить сама собі: 7.5 і 25 друкувалися як 8 і 25,
        # а резерв рахувався від неокруглених і виходив 18 замість 17.
        needed = round(topics * hours_per_topic)
        available = round(days * hours)

        lines = [
            "ПЛАН ПІДГОТОВКИ", "",
            f"Тем: {topics}, днів: {days}, годин на день: {hours:g}",
            f"Складність матеріалу: {difficulty} ({hours_per_topic:g} год на тему)",
            f"Потрібно годин: {needed}, доступно: {available}",
        ]
        lines.append(f"Резерв {available - needed} год на повторення"
                     if available >= needed else
                     f"Дефіцит {needed - available} год: додай годин або скороти обсяг")
        lines += ["", *self._schedule(topics, days, hours, hours_per_topic)]
        return "\n".join(lines)

    @staticmethod
    def _schedule(topics: int, days: int, hours: float, hours_per_topic: float) -> list:
        """Розподіл тем по днях.

        Темп рахується не тільки з тем і днів, а й з наявних годин: інакше
        система визнавала дефіцит і тут же видавала розклад, за яким на тему
        лишалося пʼятнадцять хвилин замість півтори години.
        """
        by_topics = math.ceil(topics / days)
        by_hours = max(1, int(hours // hours_per_topic))
        per_day = min(by_topics, by_hours) or 1

        lines = [f"Темп: {per_day} теми на день "
                 f"({'обмежено часом' if by_hours < by_topics else 'вистачає часу'})",
                 "", "Розподіл:"]
        covered = 0
        for day in range(1, min(days, SHOWN_DAYS) + 1):
            if covered >= topics:
                break
            end = min(covered + per_day, topics)
            lines.append(f"  День {day}: теми {covered + 1}-{end}")
            covered = end
        if covered < topics:
            lines.append(f"  Далі тим самим темпом, решта: {covered + 1}-{topics}")
        return lines
