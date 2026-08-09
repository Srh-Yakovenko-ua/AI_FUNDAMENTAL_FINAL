"""Фільтр застосовності: рішення, яке ухвалює код, а не модель.

Це головний компонент системи, і він принципово не використовує LLM.

Retrieval відповідає на питання «про що це». На питання «чи можна це застосувати
саме тут» він відповісти не здатний: у векторному просторі «умова виконується»
і «умова порушена» лежать поруч, бо слова майже ті самі. Я перевіряв це чисельно.
Тому придатність визначається зіставленням машиночитаних предикатів картки
з умовами задачі, витягнутими детермінованими правилами.
"""
import re
from typing import Optional, Sequence

from .data import FORMULAS, SITUATION_MARKERS
from .models import ApplicabilityVerdict, Formula
from .text import stems

# Слова, після яких маркер втрачає силу: «кидаю НЕ з даху, а з землі».
# «без» сюди свідомо не входить: у фізичних умовах повно зворотів «без тертя»
# і «без опору повітря», які не заперечують місце кидання.
NEGATIONS = ("не ", "ні ", "нема")

# Межі, на яких заперечення закінчується: далі вже інша частина речення.
NEGATION_STOPPERS = ",;.:"

# Пари «вимога картки → умова задачі, яка її порушує» разом з поясненням.
CONTRADICTIONS = {
    ("h0 == 0", "h0 > 0"):
        "У задачі тіло кидають з висоти (h₀ > 0), а формула виведена для кидання "
        "з нульової висоти. Потрібен розрахунок через час польоту.",
    ("h0 > 0", "h0 == 0"):
        "У задачі кидають з рівної поверхні (h₀ = 0), а ця формула призначена "
        "для кидання з висоти. Візьми простішу формулу дальності.",
    ("triangle_type == 'прямокутний'", "triangle_type != 'прямокутний'"):
        "Трикутник у задачі не прямокутний, теорема Піфагора не застосовується.",
    ("resistance_constant == True", "resistance_constant == False"):
        "У задачі опір не постійний, закон Ома в такій формі не працює.",
    ("v_units == 'м/с'", "v_units != 'м/с'"):
        "Швидкість у задачі не в м/с. Спершу переведи одиниці, інакше результат "
        "буде неправильним у 3.6 раза.",
    ("T_units == 'К'", "T_units != 'К'"):
        "Температура в задачі не в кельвінах. Переведи її, інакше рівняння дасть "
        "безглузде число.",
    ("V_units == 'м³'", "V_units != 'м³'"):
        "Об'єм у задачі не в м³. Переведи в СІ: 40 л це 0.04 м³, а не 0.4.",
}


def is_negated(text: str, position: int) -> bool:
    """Чи стоїть маркер під запереченням.

    Заперечення діє від слова «не» до найближчого розділового знака, а не
    на фіксованому вікні символів. У фразі «не з балкона третього поверху,
    а з землі» воно має накривати і «балкон», і «поверх»: вони в одній частині
    речення. Вікно фіксованої довжини накривало лише перше слово, і система
    бачила одночасно «кидання з висоти» і «кидання з землі».
    """
    prefix = text[:position]
    last_stopper = max((prefix.rfind(ch) for ch in NEGATION_STOPPERS), default=-1)
    clause = prefix[last_stopper + 1:]
    return any(neg in clause for neg in NEGATIONS)


def detect_conditions(situation: str) -> set:
    """Витягує умови задачі з тексту за явними маркерами."""
    text = " " + situation.lower() + " "
    found = set()
    for condition, markers in SITUATION_MARKERS.items():
        for marker in markers:
            # Перебираємо ВСІ входження, а не лише перше: якщо перше стоїть
            # під запереченням, наступні мають лишатися видимими.
            for match in re.finditer(re.escape(marker), text):
                if is_negated(text, match.start()):
                    continue
                found.add(condition)
                break
            if condition in found:
                break
    return found


class ApplicabilityChecker:
    """Перевіряє придатність формули до конкретної умови задачі."""

    def __init__(self, formulas: Sequence[Formula] = FORMULAS,
                 contradictions: dict = CONTRADICTIONS):
        self.formulas = list(formulas)
        self.contradictions = contradictions

    def check(self, formula: Formula, situation: str) -> ApplicabilityVerdict:
        """Вердикт про придатність. Модель у цьому кроці не бере участі взагалі."""
        if not formula.predicates:
            # Свідомо «не перевірено», а не «придатна»: відсутність обмежень
            # у базі не означає, що формула підходить саме до цієї задачі.
            # Інакше площа кола отримувала зелену галочку на задачі про кидання.
            return ApplicabilityVerdict(
                None,
                "У картці немає записаних обмежень, тому автоматично перевірити "
                "застосовність неможливо. Звір умову задачі самостійно.",
            )

        detected = detect_conditions(situation)
        if not detected:
            return ApplicabilityVerdict(
                None,
                "В умові немає ознак, за якими можна перевірити застосовність. "
                "Перевір самостійно: чи виконуються обмеження цієї формули.",
            )

        for predicate in formula.predicates:
            for condition in detected:
                reason = self.contradictions.get((predicate, condition))
                if reason:
                    return ApplicabilityVerdict(False, reason, tuple(sorted(detected)))

        return ApplicabilityVerdict(True, "Умови застосовності виконані.",
                                    tuple(sorted(detected)))

    def find_alternative(self, rejected: Sequence[Formula],
                         situation: str) -> Optional[Formula]:
        """Формула тієї ж теми, придатна для цієї задачі.

        Потрібна саме тоді, коли перевірка дала «не придатна»: сказати студенту,
        що формула не підходить, і не назвати правильну, означає лишити його
        рівно там, звідки він прийшов.
        """
        rejected_uids = {f.uid for f in rejected}
        base_stems = stems(rejected[0].name)
        for other in self.formulas:
            if other.uid in rejected_uids or other.subject != rejected[0].subject:
                continue
            if not (stems(other.name) & base_stems):
                continue
            # Формула без предикатів дає «не перевірено», а не «придатна»,
            # тому пропонувати як заміну можна лише те, що перевірку пройшло.
            if self.check(other, situation).applicable is True:
                return other
        return None
