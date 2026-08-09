"""Типи даних StudyMate.

Модуль навмисно не має залежностей від решти пакета: тут лише структури,
жодної логіки. Завдяки цьому їх можна імпортувати звідусіль без ризику
циклічних імпортів.
"""
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Formula:
    """Картка формули з довідника.

    Незмінна свідомо: дані не мають правитися під час роботи системи.

    Три поля відрізняють цю картку від рядка в пошуковій видачі:
    `predicates` дають машиночитані умови застосовності, за якими рішення
    ухвалює код; `synonyms` закривають розрив між мовою студента і мовою
    довідника; `note` несе попередження, яке обовʼязково доходить до студента.
    """

    uid: str
    subject: str
    name: str
    expression: str
    variables: dict
    example: str
    synonyms: tuple = ()
    predicates: tuple = ()
    note: str = ""
    source: str = ""


@dataclass
class RetrievalResult:
    """Знайдена картка разом з поясненням, звідки вона і наскільки їй вірити."""

    formula: Formula
    score: float
    found_by: str            # lexical, semantic або both
    lexical_score: float = 0.0
    confident: bool = False


@dataclass
class ApplicabilityVerdict:
    """Вердикт про придатність формули до конкретної задачі.

    `applicable` має ТРИ стани, і третій тут принциповий:
    True    придатна, умови перевірені;
    False   не придатна, знайдено пряме порушення;
    None    перевірити не вдалося, у тексті немає ознак.

    Раніше третій випадок зливався з першим, і система за замовчуванням казала
    «придатна». Для продукту, теза якого «відмова безпечніша за впевнену
    помилку», умовчання стояло рівно в протилежний бік.
    """

    applicable: Optional[bool]
    reason: str
    detected: tuple = ()

    @property
    def label(self) -> str:
        return {True: "✅ ПРИДАТНА", False: "❌ НЕ ПРИДАТНА"}.get(
            self.applicable, "⚠️ НЕ ПЕРЕВІРЕНО")


@dataclass
class Resolution:
    """Результат спроби знайти формулу.

    Три взаємовиключні стани, кожен веде до своєї поведінки продукту:
    formula   знайдено впевнено, можна працювати;
    options   кілька однаково близьких, треба перепитати;
    message   нічого придатного, треба чесно відмовити.
    """

    formula: Optional[Formula] = None
    options: Sequence[RetrievalResult] = ()
    message: str = ""
    found_by: str = ""

    @property
    def ok(self) -> bool:
        return self.formula is not None
