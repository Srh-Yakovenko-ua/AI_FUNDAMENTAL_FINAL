"""Ядро StudyMate: дані, гібридний пошук, фільтр застосовності, інструменти, агент.

Зібрано з тих самих джерел, що й ноутбук, тому логіка гарантовано одна.
Придатний для імпорту в тестах: семантичний шар і агент будуються ліниво,
тому імпорт не потребує ані ключа, ані мережі.
"""

import os

# Ключ беремо з Colab Secrets, далі зі змінної середовища, і лише потім питаємо вручну.


import json
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class Formula:
    """Картка формули. Незмінна: дані не мають правитися під час роботи системи."""

    uid: str
    subject: str
    name: str
    expression: str
    variables: dict
    example: str
    synonyms: tuple = ()
    predicates: tuple = ()      # машиночитані умови застосовності
    note: str = ""              # попередження для студента
    source: str = ""


FORMULAS = [
    # --- фізика: кінематика -------------------------------------------------
    Formula(
        uid="phys.projectile.range_flat",
        subject="фізика",
        name="дальність польоту тіла, кинутого під кутом",
        expression="L = v₀² · sin(2α) / g",
        variables={
            "L": "дальність польоту (м)",
            "v₀": "початкова швидкість (м/с)",
            "α": "кут кидання до горизонту",
            "g": "прискорення вільного падіння (9.8 м/с²)",
        },
        example="v₀ = 20 м/с, α = 45°: L = 400 · 1 / 9.8 ≈ 40.8 м",
        synonyms=("дальність кидка", "куди впаде тіло", "на яку відстань полетить"),
        predicates=("h0 == 0", "air_resistance == False"),
        note="Формула виведена для випадку, коли точка кидання і точка падіння на одній "
             "висоті. Для кидання з даху, вежі чи столу вона занижує відповідь.",
        source="Загальна фізика, розділ 2.4",
    ),
    Formula(
        uid="phys.projectile.range_elevated",
        subject="фізика",
        name="дальність польоту при киданні з висоти",
        expression="час t з рівняння h₀ + v₀·sin(α)·t − g·t²/2 = 0, далі L = v₀·cos(α)·t",
        variables={
            "h₀": "початкова висота (м)",
            "v₀": "початкова швидкість (м/с)",
            "α": "кут кидання",
            "t": "час польоту (с)",
        },
        example="h₀ = 12 м, v₀ = 15 м/с, α = 30°: t ≈ 2.51 с, L ≈ 32.5 м",
        synonyms=("кидання з даху", "кидання з висоти", "кинули з балкона"),
        predicates=("h0 > 0", "air_resistance == False"),
        note="Саме цей випадок плутають з формулою для рівної поверхні найчастіше.",
        source="Загальна фізика, розділ 2.5",
    ),
    Formula(
        uid="phys.energy.kinetic",
        subject="фізика",
        name="кінетична енергія",
        expression="Eₖ = m·v² / 2",
        variables={"Eₖ": "енергія (Дж)", "m": "маса (кг)", "v": "швидкість (м/с)"},
        example="m = 2 кг, v = 3 м/с: Eₖ = 9 Дж",
        synonyms=("енергія руху",),
        predicates=("v_units == 'м/с'",),
        note="Швидкість обов'язково в м/с. Якщо дано км/год, спершу переведи одиниці.",
        source="Загальна фізика, розділ 3.1",
    ),
    Formula(
        uid="phys.energy.potential",
        subject="фізика",
        name="потенціальна енергія",
        expression="Eₚ = m·g·h",
        variables={"Eₚ": "енергія (Дж)", "m": "маса (кг)", "g": "9.8 м/с²", "h": "висота (м)"},
        example="m = 5 кг, h = 10 м: Eₚ = 490 Дж",
        synonyms=("енергія висоти",),
        note="Висота відлічується від рівня, який ти сам обрав за нульовий.",
        source="Загальна фізика, розділ 3.2",
    ),
    Formula(
        uid="phys.electricity.ohm",
        subject="фізика",
        name="закон ома",
        expression="I = U / R",
        variables={"I": "сила струму (А)", "U": "напруга (В)", "R": "опір (Ом)"},
        example="U = 12 В, R = 4 Ом: I = 3 А",
        synonyms=("сила струму через напругу",),
        predicates=("resistance_constant == True",),
        note="Виконується для ділянки кола з постійним опором.",
        source="Загальна фізика, розділ 5.2",
    ),
    Formula(
        uid="phys.kinematics.speed",
        subject="фізика",
        name="середня швидкість",
        expression="v = s / t",
        variables={"v": "швидкість (м/с)", "s": "шлях (м)", "t": "час (с)"},
        example="s = 100 м, t = 10 с: v = 10 м/с",
        note="Це середня швидкість. Для змінного руху миттєва швидкість інша.",
        source="Загальна фізика, розділ 1.2",
    ),
    # --- хімія --------------------------------------------------------------
    Formula(
        uid="chem.gas.ideal",
        subject="хімія",
        name="рівняння стану ідеального газу",
        expression="p·V = n·R·T",
        variables={
            "p": "тиск (Па)", "V": "об'єм (м³)", "n": "кількість речовини (моль)",
            "R": "8.314 Дж/(моль·К)", "T": "температура (К)",
        },
        example="p = 1.2 МПа, V = 0.04 м³, T = 300 К: n = pV/(RT) ≈ 19.5 моль",
        synonyms=("рівняння менделєєва клапейрона", "закон ідеального газу"),
        predicates=("T_units == 'К'", "V_units == 'м³'"),
        note="Температура обов'язково в кельвінах, об'єм у м³. Найчастіша помилка: "
             "40 л це 0.04 м³, а не 0.4 м³.",
        source="Загальна хімія, розділ 4.1",
    ),
    Formula(
        uid="chem.solution.molar_concentration",
        subject="хімія",
        name="молярна концентрація",
        expression="C = n / V",
        variables={"C": "концентрація (моль/л)", "n": "кількість (моль)", "V": "об'єм розчину (л)"},
        example="0.5 моль у 2 л: C = 0.25 моль/л",
        synonyms=("концентрація розчину", "концентрація речовини"),
        note="V це об'єм усього розчину, а не лише розчинника.",
        source="Загальна хімія, розділ 6.3",
    ),
    Formula(
        uid="chem.molar_mass",
        subject="хімія",
        name="молярна маса",
        expression="M = m / n",
        variables={"M": "молярна маса (г/моль)", "m": "маса (г)", "n": "кількість (моль)"},
        example="36 г води, 2 моль: M = 18 г/моль",
        source="Загальна хімія, розділ 2.1",
    ),
    # --- математика ---------------------------------------------------------
    Formula(
        uid="math.geometry.circle_area",
        subject="математика",
        name="площа кола",
        expression="S = π·r²",
        variables={"S": "площа", "r": "радіус"},
        example="r = 5 см: S ≈ 78.54 см²",
        source="Геометрія, розділ 8",
    ),
    Formula(
        uid="math.geometry.pythagoras",
        subject="математика",
        name="теорема піфагора",
        expression="a² + b² = c²",
        variables={"a, b": "катети", "c": "гіпотенуза"},
        example="a = 3, b = 4: c = 5",
        synonyms=("гіпотенуза", "сторони прямокутного трикутника"),
        predicates=("triangle_type == 'прямокутний'",),
        note="Працює лише для прямокутних трикутників.",
        source="Геометрія, розділ 5",
    ),
    Formula(
        uid="math.algebra.quadratic",
        subject="математика",
        name="квадратне рівняння",
        expression="x = (−b ± √(b² − 4ac)) / 2a",
        variables={"a, b, c": "коефіцієнти рівняння ax² + bx + c = 0"},
        example="x² − 5x + 6 = 0: x₁ = 3, x₂ = 2",
        synonyms=("дискримінант", "корені рівняння"),
        note="Дискримінант D = b² − 4ac визначає кількість коренів.",
        source="Алгебра, розділ 7",
    ),
]

FORMULAS_BY_UID = {f.uid: f for f in FORMULAS}

SUBJECTS = {}
for _formula in FORMULAS:
    SUBJECTS.setdefault(_formula.subject, []).append(_formula)

STATS = {
    subject: (len(items), sum(1 for f in items if f.predicates))
    for subject, items in SUBJECTS.items()
}


STOP_WORDS = {
    "формула", "формулу", "формули", "яка", "який", "яке", "що", "таке", "як",
    "мені", "потрібна", "потрібно", "розкажи", "поясни", "для", "про", "будь",
    "ласка", "підкажи", "напиши", "покажи", "треба", "хочу", "знайти",
}
MIN_WORD_LENGTH, STEM_LENGTH = 3, 5
LEXICAL_THRESHOLD = 0.6      # нижче цього збіг вважається слабким
MIN_QUERY_LENGTH = 4         # коротший запит не несе змісту: «я» не має нічого знаходити
RRF_K = 60                   # стандартна константа згладжування для RRF


def stem_word(word: str) -> str:
    """Корінь слова, стійкий до українських відмінків.

    Просте обрізання до N символів тут не працює: «площа» і «площі» мають однакову
    довжину 5, тому обидва лишалися б собою і не збігалися. Тому коротким словам
    відрізаємо принаймні одну літеру закінчення, а довгі ріжемо до STEM_LENGTH.
    «маса»/«маси» дають «мас», «площа»/«площі» дають «площ», «енергія»/«енергії»
    дають «енерг».
    """
    if len(word) > STEM_LENGTH:
        return word[:STEM_LENGTH]
    if len(word) >= 4:
        return word[:len(word) - 1]
    return word


def stems(text: str, drop_stop: bool = False) -> set:
    """Множина коренів значущих слів."""
    words = [w.strip(".,;:!?()«»\"'") for w in text.lower().split()]
    return {
        stem_word(w) for w in words
        if len(w) >= MIN_WORD_LENGTH and not (drop_stop and w in STOP_WORDS)
    }


def lexical_score(query: str, target: str) -> float:
    """Двостороння міра схожості F2: покриття запиту важить більше за покриття назви.

    Чому саме так, а не проста частка збігів. Назва «швидкість» складається з одного
    слова, тому запит «швидкість світла» покриває її на 100%, і симетрична міра
    вважала б це відмінним збігом, хоча про світло в базі немає нічого. Слово запиту,
    якого немає в назві, це сигнал «питають не про це», і він має важити сильніше.
    """
    q, t = stems(query, drop_stop=True), stems(target)
    if not q or not t:
        return 0.0
    hits = len(q & t)
    if not hits:
        return 0.0
    precision, recall = hits / len(t), hits / len(q)
    beta_sq = 4
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def lexical_search(query: str, top_k: int = 5) -> list:
    """Лексичний пошук по назвах і синонімах. Повертає [(uid, score), ...]."""
    scored = []
    for formula in FORMULAS:
        best = max(
            [lexical_score(query, formula.name)]
            + [lexical_score(query, syn) for syn in formula.synonyms]
        )
        if best > 0:
            scored.append((formula.uid, best))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


class SemanticSearch:
    """Семантичний шар на ембеддінгах OpenAI.

    Вмикається лише за наявності ключа. Якщо ключа немає або API недоступне,
    система не падає, а працює на самому лексичному шарі: демо має відтворюватися
    в будь-якому середовищі, а деградація має бути явною, а не тихою.
    """

    def __init__(self, formulas: Sequence[Formula], model: str = EMBED_MODEL):
        self.formulas = list(formulas)
        self.model = model
        self.available = False
        self.vectors = None
        self.tokens_used = 0

    def build(self) -> bool:
        """Будує індекс. Повертає True, якщо семантичний шар доступний."""
        try:
            from openai import OpenAI

            self._client = OpenAI()
            # Для ембеддінга беремо назву, синоніми і примітку: саме вони несуть зміст,
            # а не сам вираз формули, який складається з символів.
            texts = [
                f"{f.name}. {' '.join(f.synonyms)}. {f.note}".strip()
                for f in self.formulas
            ]
            response = self._client.embeddings.create(model=self.model, input=texts)
            self.tokens_used += response.usage.total_tokens
            self.vectors = np.array([item.embedding for item in response.data])
            self.available = True
            print(f"✅ Семантичний шар побудовано: {self.vectors.shape}, "
                  f"токенів {response.usage.total_tokens}")
        except Exception as error:
            print(f"⚠️  Семантичний шар вимкнено ({type(error).__name__}). "
                  "Система працює на лексичному пошуку.")
            self.available = False
        return self.available

    def search(self, query: str, top_k: int = 5) -> list:
        """Пошук за змістом. Порожній список, якщо шар недоступний."""
        if not self.available:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=[query])
            self.tokens_used += response.usage.total_tokens
            q = np.array(response.data[0].embedding)
            # Вектори OpenAI нормалізовані, тому косинус це звичайний скалярний добуток.
            sims = self.vectors @ q
            order = np.argsort(sims)[::-1][:top_k]
            return [(self.formulas[i].uid, float(sims[i])) for i in order]
        except Exception:
            return []


semantic = SemanticSearch(FORMULAS)
# У ноутбуці будуємо одразу, щоб демо працювало «з коробки».
# Модуль studymate_core робить це ліниво, з боку застосунку.

def reciprocal_rank_fusion(*rankings: list, k: int = RRF_K) -> list:
    """Зливає кілька ранжувань у одне за формулою RRF: score = Σ 1/(k + rank).

    Перевага перед складанням оцінок у тому, що шкали шарів не треба зводити
    докупи: лексична міра лежить у [0,1], косинусна близькість теж, але розподілені
    вони по-різному. RRF працює з порядком, а не зі значеннями.
    """
    scores = {}
    for ranking in rankings:
        for rank, (uid, _) in enumerate(ranking, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


@dataclass
class RetrievalResult:
    """Результат пошуку з поясненням, звідки він узявся і наскільки йому можна вірити."""

    formula: Formula
    score: float
    found_by: str            # lexical, semantic або both
    lexical_score: float = 0.0
    confident: bool = False  # чи достатньо збігу, щоб віддати єдину відповідь


def hybrid_search(query: str, top_k: int = 3) -> list:
    """Гібридний пошук: лексика плюс семантика, злиття через RRF.

    Кожен результат несе прапорець `confident`. Він потрібен тому, що сам факт
    «щось знайшлося» нічого не означає: слабкий збіг за одним загальним словом
    теж потрапляє у видачу. Рішення «віддати відповідь чи перепитати» ухвалюється
    за цим прапорцем, а не за фактом наявності результату.
    """
    lex = lexical_search(query, top_k=5)
    sem = semantic.search(query, top_k=5)

    lex_uids = {uid for uid, _ in lex}
    sem_uids = {uid for uid, _ in sem}
    lex_scores = dict(lex)

    fused = reciprocal_rank_fusion(lex, sem) if sem else lex
    results = []
    for uid, score in fused[:top_k]:
        source = "both" if uid in lex_uids and uid in sem_uids else (
            "lexical" if uid in lex_uids else "semantic"
        )
        lex_score = lex_scores.get(uid, 0.0)
        # Впевненість дає або сильний лексичний збіг, або підтвердження обома шарами:
        # якщо і лексика, і семантика вказали на ту саму картку, це вагомий сигнал.
        results.append(RetrievalResult(
            formula=FORMULAS_BY_UID[uid],
            score=score,
            found_by=source,
            lexical_score=lex_score,
            confident=lex_score >= LEXICAL_THRESHOLD or source == "both",
        ))
    return results


AMBIGUITY_TOLERANCE = 0.1


def is_ambiguous(results: Sequence[RetrievalResult],
                 tolerance: float = AMBIGUITY_TOLERANCE) -> bool:
    """Чи занадто близькі два найкращі результати, щоб обирати між ними самостійно.

    Нічия це не збій, а корисний сигнал. Запит «формула дальності польоту» однаково
    описує і кидання з рівної поверхні, і кидання з висоти, а різниця між ними
    вирішальна: одна формула для іншої задачі дає відповідь, меншу в півтора рази.
    Мовчки взяти першу картку означало б приховати від студента, що варіант був не один.

    Поріг саме на близькість, а не на точну рівність. На цій парі міра дала 0.769
    і 0.714: формально різні числа, але різниця в 0.055 не є підставою впевнено
    обирати. Там, де система не має підстав для вибору, вона має перепитати,
    а не вгадувати.
    """
    if len(results) < 2:
        return False
    return abs(results[0].lexical_score - results[1].lexical_score) < tolerance


# Маркери ситуацій у тексті задачі. Це навмисно простий і прозорий механізм:
# його можна прочитати, перевірити й доповнити, не чіпаючи решту системи.
SITUATION_MARKERS = {
    "h0 > 0": ["з даху", "з балкона", "з балкону", "з вежі", "зі столу", "з обриву",
               "з висоти", "з мосту", "з дерева", "згори", "з вікна", "зі скелі",
               "з драбини", "з поверху", "поверху", "початкова висота", "h0 =", "h₀ ="],
    "h0 == 0": ["з землі", "з поверхні землі", "на рівній", "з підлоги", "рівна поверхня"],
    "triangle_type != 'прямокутний'": ["гострокутн", "тупокутн", "рівносторонн",
                                       "не прямокутн"],
    "resistance_constant == False": ["змінний опір", "нелінійн", "напівпровідник"],
    # Одиниці: картка вимагає СІ, а в задачі дано інше. Раніше ці предикати були
    # в базі, але жодна гілка їх не перевіряла, тобто вони існували лише на папері.
    "v_units != 'м/с'": ["км/год", "км на годину", "кілометрів на годину", "миль/год"],
    "T_units != 'К'": ["°c", "цельсі", "градусів цельсія", "за цельсієм"],
    "V_units != 'м³'": ["літр", "мілілітр", " мл ", " л "],
}

# Слова, після яких маркер втрачає силу: «кидаю НЕ з даху, а з землі».
NEGATIONS = ("не ", "ні ", "нема", "без ")


@dataclass
class ApplicabilityVerdict:
    """Вердикт про придатність формули.

    `applicable` має ТРИ стани, і третій тут принциповий:
    True    придатна, умови перевірені;
    False   не придатна, знайдено пряме порушення;
    None    перевірити не вдалося, у тексті немає ознак.

    Раніше третій випадок зливався з першим, і система за замовчуванням казала
    «придатна». Для продукту, теза якого «відмова безпечніша за впевнену помилку»,
    умовчання стояло рівно в протилежний бік.
    """

    applicable: Optional[bool]
    reason: str
    detected: tuple = ()

    @property
    def verdict_label(self) -> str:
        return {True: "✅ ПРИДАТНА", False: "❌ НЕ ПРИДАТНА"}.get(
            self.applicable, "⚠️ НЕ ПЕРЕВІРЕНО")


def detect_conditions(situation: str) -> set:
    """Витягує умови задачі з тексту за явними маркерами.

    Маркер під запереченням не зараховується: «кидаю не з даху, а з землі»
    не має давати умову «кидання з висоти».
    """
    text = " " + situation.lower() + " "
    found = set()
    for condition, markers in SITUATION_MARKERS.items():
        for marker in markers:
            position = text.find(marker)
            if position == -1:
                continue
            prefix = text[max(0, position - 12):position]
            if any(neg in prefix for neg in NEGATIONS):
                continue
            found.add(condition)
            break
    return found


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


def check_applicability(formula: Formula, situation: str) -> ApplicabilityVerdict:
    """Перевіряє, чи придатна формула для описаної ситуації.

    Рішення ухвалює код: предикати картки зіставляються з умовами, витягнутими
    з тексту. Модель у цьому кроці не бере участі взагалі, бо саме тут вона
    найчастіше помиляється, підтверджуючи знайому формулу.
    """
    if not formula.predicates:
        return ApplicabilityVerdict(True, "Формула не має обмежень у базі.")

    detected = detect_conditions(situation)
    if not detected:
        return ApplicabilityVerdict(
            None,
            "В умові немає ознак, за якими можна перевірити застосовність. "
            "Перевір самостійно: чи виконуються обмеження цієї формули.",
        )

    for predicate in formula.predicates:
        for condition in detected:
            reason = CONTRADICTIONS.get((predicate, condition))
            if reason:
                return ApplicabilityVerdict(False, reason, tuple(sorted(detected)))

    return ApplicabilityVerdict(True, "Умови застосовності виконані.",
                                tuple(sorted(detected)))


from functools import wraps

from langchain_core.tools import tool


def requires(**field_prompts):
    """Декоратор: перевіряє, що названі аргументи не порожні.

    Раніше ці ж три рядки перевірки повторювалися в кожному з чотирьох інструментів.
    Декоратор прибирає дублювання і, головне, робить вимогу видимою в сигнатурі:
    видно, який аргумент обовʼязковий і що саме побачить студент, якщо його забути.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = dict(zip(func.__code__.co_varnames, args))
            bound.update(kwargs)
            for field, prompt in field_prompts.items():
                value = bound.get(field)
                if not value or not str(value).strip():
                    return prompt
            return func(*args, **kwargs)
        return wrapper
    return decorator


# --- чисті функції рендерингу --------------------------------------------
def render_card(formula: Formula, found_by: str = "") -> str:
    """Картка формули у вигляді тексту. Нічого не шукає й не вирішує."""
    lines = [
        f"{formula.name.upper()} ({formula.subject})",
        "",
        f"Формула: {formula.expression}",
        "",
        "Змінні:",
    ]
    lines += [f"  {sym}: {meaning}" for sym, meaning in formula.variables.items()]
    lines += ["", f"Приклад: {formula.example}"]
    if formula.note:
        lines += ["", f"⚠️ Умова застосовності: {formula.note}"]
    if formula.source:
        lines += ["", f"Джерело: {formula.source}"]
    if found_by:
        lines += ["", f"[знайдено: {found_by}, id={formula.uid}]"]
    return "\n".join(lines)


def render_options(results: Sequence[RetrievalResult]) -> str:
    """Перелік варіантів для уточнення."""
    return "\n".join(f"  • {r.formula.name} ({r.formula.subject})" for r in results)


def render_verdict(formula: Formula, verdict: ApplicabilityVerdict) -> list:
    """Рядки вердикту про придатність однієї формули."""
    return [formula.name, f"  {verdict.verdict_label}. {verdict.reason}"]


# --- резолвер: пошук формули разом з рішенням про довіру ------------------
@dataclass
class Resolution:
    """Результат спроби знайти формулу.

    Три взаємовиключні стани, і кожен веде до своєї поведінки продукту:
    formula   знайдено впевнено, можна працювати;
    options   знайдено кілька однаково близьких, треба перепитати;
    message   нічого придатного, треба чесно відмовити.
    """

    formula: Optional[Formula] = None
    options: Sequence[RetrievalResult] = ()
    message: str = ""
    found_by: str = ""

    @property
    def ok(self) -> bool:
        return self.formula is not None


def resolve_formula(query: str, top_k: int = 3) -> Resolution:
    """Шукає формулу і вирішує, чи можна віддавати її як відповідь.

    Ця логіка потрібна двом інструментам: пошуку і перевірці застосовності.
    Коли вона була продубльована, гейт впевненості стояв лише в одному з них,
    і другий на запит «закон збереження імпульсу» впевнено віддавав ЗАКОН ОМА.
    Спільний резолвер робить таке розходження неможливим.
    """
    results = hybrid_search(query, top_k=top_k)
    if not results:
        subjects = ", ".join(sorted({f.subject for f in FORMULAS}))
        return Resolution(message=(
            f"Формули '{query}' немає в базі StudyMate.\n"
            f"Доступні предмети: {subjects}.\n"
            "Я не можу навести формулу, якої немає в базі."
        ))

    # Порядок перевірок принциповий: спершу впевненість, потім нічия.
    # Якщо перевіряти навпаки, слабкий збіг із двома однаково поганими кандидатами
    # виглядає як «є кілька варіантів», хоча насправді немає жодного:
    # запит «закон збереження імпульсу» так проходив як неоднозначність
    # замість чесної відмови.
    if not results[0].confident:
        return Resolution(message=(
            f"Точного збігу за запитом '{query}' немає.\n"
            f"Найближче за змістом:\n{render_options(results)}\n\n"
            "Якщо потрібної формули тут немає, значить її немає в базі."
        ), options=results)

    if is_ambiguous(results):
        close = [r for r in results
                 if abs(r.lexical_score - results[0].lexical_score) < AMBIGUITY_TOLERANCE]
        return Resolution(options=close)

    return Resolution(formula=results[0].formula, found_by=results[0].found_by)


# --- інструменти ----------------------------------------------------------
@tool
@requires(query="Вкажи назву формули або тему, наприклад: 'кінетична енергія'.")
def formula_lookup(query: str) -> str:
    """Шукає формулу з математики, фізики або хімії у перевіреній базі StudyMate.

    Використовуй ЗАВЖДИ, коли студент питає формулу, просить пригадати, як щось
    обчислити, або хоче пояснення змінних. Ніколи не наводь формулу з власної пам'яті:
    тільки те, що повернув цей інструмент.

    Args:
        query: Назва формули або тема українською, наприклад "кінетична енергія".

    Returns:
        Картку формули з умовою застосовності, або перелік варіантів, якщо запит
        неоднозначний, або чесне повідомлення, що формули в базі немає.
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return f"Запит '{query}' закороткий, назви тему повністю."

    resolution = resolve_formula(query)
    if resolution.ok:
        return render_card(resolution.formula, resolution.found_by)
    if resolution.options and not resolution.message:
        return (
            f"За запитом '{query}' підходять кілька формул:\n"
            f"{render_options(resolution.options)}\n\n"
            "Уточни, яка саме потрібна, або опиши умову задачі: "
            "тоді я перевірю, котра з них підходить."
        )
    return resolution.message


def find_alternative(rejected: Sequence[Formula], task: str) -> Optional[Formula]:
    """Шукає формулу тієї ж теми, придатну для цієї задачі.

    Потрібна саме тоді, коли перевірка дала «не придатна»: сказати студенту,
    що формула не підходить, і не назвати правильну, означає лишити його
    рівно там, звідки він прийшов.
    """
    rejected_uids = {f.uid for f in rejected}
    base_stems = stems(rejected[0].name)
    for other in FORMULAS:
        if other.uid in rejected_uids or other.subject != rejected[0].subject:
            continue
        if not (stems(other.name) & base_stems):
            continue
        if check_applicability(other, task).applicable is True:
            return other
    return None


@tool
@requires(
    formula_name="Вкажи назву формули, яку треба перевірити.",
    task_description="Наведи умову задачі, інакше перевірити застосовність неможливо.",
)
def check_formula_for_task(formula_name: str, task_description: str) -> str:
    """Перевіряє, чи можна застосувати формулу до КОНКРЕТНОЇ умови задачі.

    Використовуй ОБОВ'ЯЗКОВО, коли студент описує свою задачу і збирається
    застосувати формулу: наприклад, кидає тіло з даху, працює з непрямокутним
    трикутником або зі змінним опором. Це найважливіша перевірка в системі:
    формула може бути правильною сама по собі і невірною для цієї задачі.

    Args:
        formula_name: Назва формули, наприклад "дальність польоту".
        task_description: Повний текст умови задачі студента.

    Returns:
        Вердикт про придатність із поясненням причини і, за потреби, вказівкою
        на правильну альтернативу.
    """
    resolution = resolve_formula(formula_name)
    if resolution.message and not resolution.ok:
        return (
            f"Формули '{formula_name}' немає в базі StudyMate, тому перевіряти нічого.\n"
            "Я не перевіряю формул, яких немає в базі."
        )

    # Неоднозначність тут не привід обирати за студента: перевіряємо ВСІ близькі
    # варіанти й показуємо, який з них підходить саме до цієї задачі.
    candidates = ([r.formula for r in resolution.options] if resolution.options
                  else [resolution.formula])

    lines = [f"Умова задачі: {task_description[:130]}", ""]
    if len(candidates) > 1:
        lines += [f"За назвою '{formula_name}' у базі є {len(candidates)} формули, "
                  "перевіряю кожну:", ""]

    suitable = []
    for formula in candidates:
        verdict = check_applicability(formula, task_description)
        lines += render_verdict(formula, verdict) + [""]
        if verdict.applicable is True:
            suitable.append(formula)

    if suitable:
        if len(candidates) > 1:
            lines += [f"Бери: {suitable[0].name}", f"  {suitable[0].expression}"]
    else:
        alternative = find_alternative(candidates, task_description)
        if alternative:
            lines += [f"Для цієї задачі підходить: {alternative.name}",
                      f"  {alternative.expression}"]

    return "\n".join(lines).rstrip()


# Таблиця конвертацій: усі перетворення лінійні, тому зберігаємо коефіцієнти
# (множник і зсув), а не готові значення й не lambda-функції. Так дані лишаються
# даними: їх видно очима, можна перевірити й розширити одним рядком.
PHYSICS_CONVERSIONS = {
    ("км/год", "м/с"): (1 / 3.6, 0), ("м/с", "км/год"): (3.6, 0),
    ("л", "м³"): (0.001, 0), ("м³", "л"): (1000, 0),
    ("C", "F"): (9 / 5, 32), ("F", "C"): (5 / 9, -32 * 5 / 9),
    ("C", "K"): (1, 273.15), ("K", "C"): (1, -273.15),
    ("атм", "Па"): (101_325, 0), ("Па", "атм"): (1 / 101_325, 0),
    ("бар", "Па"): (100_000, 0), ("Па", "бар"): (1 / 100_000, 0),
    ("Дж", "кал"): (1 / 4.184, 0), ("кал", "Дж"): (4.184, 0),
    ("г", "кг"): (0.001, 0), ("кг", "г"): (1000, 0),
}

# Аліаси одиниць з відмінковими формами. Без них найчастіший запит демо
# «скільки кубічних метрів у 40 літрАХ» не розпізнавався взагалі.
UNIT_ALIASES = {
    "км/год": ["км/год", "км/г", "kmh", "кілометрів на годину"],
    "м/с": ["м/с", "m/s", "метрів на секунду"],
    "л": ["літрах", "літрів", "літри", "літра", "літр", "л"],
    "м³": ["кубічних метрів", "кубічний метр", "кубометр", "м³", "кубічн", "м3"],
    "C": ["цельсія", "цельсій", "цельсіях", "цельсієм", "°c", "°с"],
    "F": ["фаренгейтах", "фаренгейтів", "фаренгейти", "фаренгейт", "°f"],
    "K": ["кельвінах", "кельвінів", "кельвіни", "кельвін", "k"],
    "атм": ["атмосферах", "атмосфер", "атм"],
    "Па": ["паскалях", "паскалів", "паскаль", "паскал", "па"],
    "бар": ["барах", "бар"],
    "Дж": ["джоулях", "джоулів", "джоуль", "джоул", "дж"],
    "кал": ["калоріях", "калорій", "калорія", "калор", "кал"],
    "г": ["грамах", "грамів", "грами", "грам", "г"],
    "кг": ["кілограмах", "кілограмів", "кілограми", "кілограм", "кг"],
}

_LETTER = "а-яіїєґёa-z"


def _find_units(text: str) -> list:
    """Знаходить одиниці по МЕЖАХ СЛОВА разом з позицією.

    Межі слова тут не педантизм: без них «кал» знаходиться всередині «шкала»,
    а «па» всередині «пару», і система відмовляє на цілком нормальному запиті.
    """
    found = []
    for canonical, aliases in UNIT_ALIASES.items():
        positions = []
        for alias in aliases:
            pattern = rf"(?<![{_LETTER}]){re.escape(alias)}(?![{_LETTER}])"
            positions += [m.start() for m in re.finditer(pattern, text)]
        if positions:
            found.append((min(positions), canonical))
    return sorted(found)


def _format_number(value: float) -> str:
    """Число для школяра, а не для інженера: без експонент і без втрати значущих цифр."""
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    if abs(value) >= 10_000:
        return f"{value:,.2f}".rstrip("0").rstrip(".").replace(",", " ")
    if abs(value) >= 0.001:
        return f"{value:.6g}"
    return f"{value:.10f}".rstrip("0")


@tool
def convert_units(query: str) -> str:
    """Переводить фізичні одиниці: швидкість, температуру, об'єм, тиск, енергію, масу.

    Використовуй для будь-якого переведення одиниць. Не рахуй конвертацію самостійно,
    навіть якщо вона здається простою: помилка на порядок тут найчастіша.

    Args:
        query: Запит із числом і двома одиницями, наприклад "40 літрів у м³".

    Returns:
        Результат конвертації або пояснення, чого забракло в запиті.
    """
    if not query or not query.strip():
        return "Вкажи значення та одиниці, наприклад: '40 літрів у м³'."

    padded = f" {query.lower()} "
    match = re.search(r"-?\d+(?:[.,]\d+)?", padded)
    if not match:
        return "Не бачу числа в запиті. Приклад: '100 км/год у м/с'."

    value = float(match.group().replace(",", "."))
    number_pos = match.start()
    units = _find_units(padded)

    if len(units) < 2:
        pairs = ", ".join(f"{a}→{b}" for a, b in list(PHYSICS_CONVERSIONS)[:6])
        return (
            f"Не розпізнав обидві одиниці в '{query}'.\n"
            f"Підтримуються, наприклад: {pairs}."
        )

    # Вихідна одиниця це ПЕРША після числа: у живій мові величину називають одразу
    # за числом («40 літрів»), а цільову виносять уперед або в кінець.
    after = [u for pos, u in units if pos > number_pos]
    from_unit = after[0] if after else min(units, key=lambda p: abs(p[0] - number_pos))[1]
    rest = [u for _, u in units if u != from_unit]
    if not rest:
        return f"{_format_number(value)} {from_unit} це вже і є {from_unit}."
    to_unit = rest[0]

    pair = PHYSICS_CONVERSIONS.get((from_unit, to_unit))
    if pair is None:
        return f"Конвертацію з '{from_unit}' у '{to_unit}' не підтримано."

    factor, offset = pair
    result = value * factor + offset
    return f"{_format_number(value)} {from_unit} = {_format_number(result)} {to_unit}"


NUM = r"(-?\d+(?:[.,]\d+)?)"
MAX_REALISTIC_HOURS = 12


def _first_match(patterns: list, text: str) -> Optional[float]:
    """Перший збіг із переліку шаблонів, у порядку пріоритету."""
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


@tool
def plan_exam_prep(query: str) -> str:
    """Будує план підготовки до іспиту за кількістю тем, днів і годин на день.

    Використовуй, коли студент планує підготовку або питає, чи вистачає часу.

    Args:
        query: Запит із кількістю тем, днів і годин, наприклад "20 тем, 10 днів, 3 години на день".

    Returns:
        План з розрахунком навантаження або пояснення, яких даних бракує.
    """
    if not query or not query.strip():
        return "Вкажи кількість тем, днів і годин на день."

    text = query.lower()
    # Прикметники між числом і словом: «20 важких тем», «15 складних тем».
    topics = _first_match([rf"{NUM}\s*(?:\w+\s+){{0,2}}тем", rf"тем\w*\s*[-:]?\s*{NUM}"], text)
    weeks = _first_match([rf"{NUM}\s*тижн"], text)
    days = weeks * 7 if weeks is not None else _first_match(
        [rf"{NUM}\s*(?:дн|день|доб)", rf"дн\w*\s*[-:]?\s*{NUM}"], text)
    hours = _first_match([
        rf"{NUM}\s*год\w*\s*(?:на день|щодня|за день|в день)",
        rf"{NUM}\s*(?:год|час)", rf"год\w*\s*[-:]?\s*{NUM}",
    ], text)

    missing = [n for n, v in [("тем", topics), ("днів", days), ("годин на день", hours)] if v is None]
    if missing:
        return f"Бракує даних: {', '.join(missing)}. Приклад: '20 тем, 10 днів, 3 години на день'."

    topics, days = int(topics), int(days)
    if topics <= 0 or days <= 0 or hours <= 0:
        return "Усі значення мають бути більші за нуль."
    if hours > MAX_REALISTIC_HOURS:
        return (
            f"{hours:g} годин на день нереалістично: на сон і відпочинок не лишається часу.\n"
            f"Більше за {MAX_REALISTIC_HOURS} годин планувати немає сенсу."
        )

    # «прост» тут навмисно немає: воно ловило вставне слово «просто»
    # і мовчки міняло оцінку навантаження на третину.
    if "важк" in text or "склад" in text:
        difficulty, hours_per_topic = "важка", 2.5
    elif "легк" in text or "нескладн" in text:
        difficulty, hours_per_topic = "легка", 1.0
    else:
        difficulty, hours_per_topic = "середня", 1.5
    needed, available = round(topics * hours_per_topic), round(days * hours)
    per_day = math.ceil(topics / days)

    lines = [
        "ПЛАН ПІДГОТОВКИ", "",
        f"Тем: {topics}, днів: {days}, годин на день: {hours:g}",
        f"Складність матеріалу: {difficulty} ({hours_per_topic:g} год на тему)",
        f"Потрібно годин: {needed}, доступно: {available}",
    ]
    lines.append(
        f"Резерв {available - needed} год на повторення" if available >= needed
        else f"Дефіцит {needed - available} год: додай годин або скороти обсяг"
    )
    lines += ["", f"Темп: {per_day} теми на день", "", "Розподіл:"]
    covered = 0
    for day in range(1, min(days, 7) + 1):
        if covered >= topics:
            break
        end = min(covered + per_day, topics)
        lines.append(f"  День {day}: теми {covered + 1}-{end}")
        covered = end
    if covered < topics:
        lines.append(f"  Далі тим самим темпом, решта: {covered + 1}-{topics}")
    return "\n".join(lines)


TOOLS = [formula_lookup, check_formula_for_task, convert_units, plan_exam_prep]

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """Ти StudyMate, освітній асистент для студентів з математики, фізики та хімії.

## Роль
Ти не видаєш відповіді, ти пояснюєш. Студент має зрозуміти, звідки береться результат.
Пояснюй простою мовою, як однокласник, який добре розібрався в темі.

## Правила виклику інструментів
1. formula_lookup: ЗАВЖДИ, коли питають формулу. Формулу з власної пам'яті наводити заборонено.
2. check_formula_for_task: ОБОВ'ЯЗКОВО, коли студент описує свою задачу і збирається
   застосувати формулу. Формула може бути правильною і при цьому непридатною саме тут.
3. convert_units: для будь-якого переведення одиниць, навіть очевидного.
4. plan_exam_prep: коли планують підготовку.

## Жорсткі межі
- Якщо інструмент не знайшов формулу, ТАК І СКАЖИ. Краще чесне «немає в базі»,
  ніж правдоподібна вигадка: студент не зможе відрізнити одне від одного.
- Якщо в картці є умова застосовності, ОБОВ'ЯЗКОВО передай її студенту.
- Якщо інструмент повернув кілька варіантів, не обирай сам: перепитай студента.
- Якщо перевірка показала, що формула непридатна, прямо скажи про це і назви правильний шлях.
- Не розв'язуй контрольну за студента без пояснення кроків.
- Якщо питання не про точні науки, ввічливо поясни свій профіль.

## Формат
Стисло: спочатку суть, потім пояснення, за потреби приклад. Не більше кількох абзаців.
"""

def build_agent():
    """Створює агента. Виклик відкладено, щоб імпорт модуля не вимагав ключа:
    інструменти це чисті функції, і їх треба вміти тестувати офлайн."""
    model = ChatOpenAI(model=LLM_MODEL, temperature=0.2, max_tokens=900)
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


_AGENT = None


def get_agent():
    """Ледаче створення агента: імпорт модуля не має вимагати ключа."""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


ERROR_PREFIX = "⚠️ Помилка виклику агента:"


def extract_tool_calls(messages: list) -> list:
    """Назви інструментів, які агент викликав. Без цього неможливо відрізнити
    відповідь з бази від відповіді з пам'яті моделі."""
    called = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                called.append(name)
    return called


def ask(user_input: str, history: Optional[list] = None, verbose: bool = True):
    """Один хід діалогу з урахуванням контексту.

    history це список messages, який передається в агента цілком на кожному кроці.
    Саме він робить можливими уточнення на кшталт «а якщо з даху?».
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": user_input})
    try:
        result = get_agent().invoke({"messages": messages})
    except Exception as error:
        return f"{ERROR_PREFIX} {type(error).__name__}: {error}", messages, []

    updated = result["messages"]
    answer = updated[-1].content
    tools_used = extract_tool_calls(updated[len(messages):])

    if verbose:
        print(f"👤 {user_input}")
        if tools_used:
            print(f"🔧 {', '.join(tools_used)}")
        print(f"🤖 {answer}\n" + "─" * 78)
    return answer, updated, tools_used


