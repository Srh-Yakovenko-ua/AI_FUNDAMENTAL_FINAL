"""Інструменти агента.

Кожен інструмент **звужує** свободу моделі, а не розширює її: формула приходить
з бази, придатність вирішує код, одиниці переводить таблиця, план рахує алгоритм.
Моделі лишається мова.

Самі `@tool` тут навмисно короткі: вони склеюють готові шари й нічого не
обчислюють. Уся логіка живе в `search`, `applicability`, `converters`, `planner`,
а форматування у чистих функціях `render_*`.
"""
from functools import wraps
from typing import Sequence

from langchain_core.tools import tool

from .applicability import ApplicabilityChecker
from .converters import UnitConverter
from .data import FORMULAS
from .models import ApplicabilityVerdict, Formula, RetrievalResult
from .planner import StudyPlanner
from .search import MIN_QUERY_LENGTH, HybridSearch

# Спільні екземпляри: стан у них лише кешований (ембеддінги бази),
# тому тримати по одному на процес безпечно й дешево.
SEARCH = HybridSearch()
CHECKER = ApplicabilityChecker()
CONVERTER = UnitConverter()
PLANNER = StudyPlanner()


def requires(**field_prompts):
    """Декоратор: перевіряє, що названі аргументи не порожні.

    Раніше ці три рядки перевірки повторювалися в кожному інструменті.
    Декоратор прибирає дублювання і робить вимогу видимою поруч із сигнатурою:
    одразу зрозуміло, який аргумент обовʼязковий і що побачить студент,
    якщо його забути.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import inspect

            bound = inspect.signature(func).bind_partial(*args, **kwargs).arguments
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
    lines = [f"{formula.name.upper()} ({formula.subject})", "",
             f"Формула: {formula.expression}", "", "Змінні:"]
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
    return [formula.name, f"  {verdict.label}. {verdict.reason}"]


# --- інструменти ----------------------------------------------------------
@tool
@requires(query="Вкажи назву формули або тему, наприклад: 'кінетична енергія'.")
def formula_lookup(query: str) -> str:
    """Шукає формулу з математики, фізики або хімії у перевіреній базі StudyMate.

    Використовуй ЗАВЖДИ, коли студент питає формулу, просить пригадати, як щось
    обчислити, або хоче пояснення змінних. Ніколи не наводь формулу з власної
    пам'яті: тільки те, що повернув цей інструмент.

    Args:
        query: Назва формули або тема українською, наприклад "кінетична енергія".

    Returns:
        Картку формули з умовою застосовності, перелік варіантів для уточнення
        або чесне повідомлення, що формули в базі немає.
    """
    if len(query.strip()) < MIN_QUERY_LENGTH:
        return f"Запит '{query}' закороткий, назви тему повністю."

    resolution = SEARCH.resolve(query)
    if resolution.ok:
        return render_card(resolution.formula, resolution.found_by)
    if resolution.options and not resolution.message:
        return (f"За запитом '{query}' підходять кілька формул:\n"
                f"{render_options(resolution.options)}\n\n"
                "Уточни, яка саме потрібна, або опиши умову задачі: "
                "тоді я перевірю, котра з них підходить.")
    return resolution.message


@tool
@requires(
    formula_name="Вкажи назву формули, яку треба перевірити.",
    task_description="Наведи умову задачі, інакше перевірити застосовність неможливо.",
)
def check_formula_for_task(formula_name: str, task_description: str) -> str:
    """Перевіряє, чи можна застосувати формулу до КОНКРЕТНОЇ умови задачі.

    Використовуй ОБОВ'ЯЗКОВО, коли студент описує свою задачу і збирається
    застосувати формулу: кидає тіло з даху, працює з непрямокутним трикутником
    або зі змінним опором. Це найважливіша перевірка в системі: формула може
    бути правильною сама по собі і невірною саме для цієї задачі.

    Args:
        formula_name: Назва формули, наприклад "дальність польоту".
        task_description: Повний текст умови задачі студента.

    Returns:
        Вердикт про придатність із поясненням причини і, за потреби, вказівкою
        на правильну альтернативу.
    """
    if len(formula_name.strip()) < MIN_QUERY_LENGTH:
        return f"Назва '{formula_name}' закоротка, напиши формулу повністю."

    resolution = SEARCH.resolve(formula_name)
    if resolution.message and not resolution.ok:
        return (f"Формули '{formula_name}' немає в базі StudyMate, "
                "тому перевіряти нічого.\n"
                "Я не перевіряю формул, яких немає в базі.")

    # Неоднозначність тут не привід обирати за студента: перевіряємо ВСІ
    # близькі варіанти й показуємо, який підходить саме до цієї задачі.
    candidates = ([r.formula for r in resolution.options] if resolution.options
                  else [resolution.formula])

    lines = [f"Умова задачі: {task_description[:130]}", ""]
    if len(candidates) > 1:
        lines += [f"За назвою '{formula_name}' у базі є {len(candidates)} формули, "
                  "перевіряю кожну:", ""]

    verdicts = [(f, CHECKER.check(f, task_description)) for f in candidates]
    for formula, verdict in verdicts:
        lines += render_verdict(formula, verdict) + [""]

    suitable = [f for f, v in verdicts if v.applicable is True]
    rejected = [f for f, v in verdicts if v.applicable is False]

    if suitable and len(candidates) > 1:
        lines += [f"Бери: {suitable[0].name}", f"  {suitable[0].expression}"]
    elif not suitable and rejected:
        # Альтернативу шукаємо ЛИШЕ коли формулу справді відхилено. Раніше сюди
        # потрапляв і вердикт «не перевірено», і система радила потенціальну
        # енергію там, де задача ідеально лягала на кінетичну.
        alternative = CHECKER.find_alternative(rejected, task_description)
        if alternative:
            lines += [f"Для цієї задачі підходить: {alternative.name}",
                      f"  {alternative.expression}"]
    return "\n".join(lines).rstrip()


@tool
@requires(query="Вкажи значення та одиниці, наприклад: '40 літрів у м³'.")
def convert_units(query: str) -> str:
    """Переводить фізичні одиниці: швидкість, температуру, обʼєм, тиск, енергію, масу.

    Використовуй для будь-якого переведення одиниць. Не рахуй конвертацію
    самостійно, навіть якщо вона здається простою: помилка на порядок
    тут найчастіша.

    Args:
        query: Запит із числом і двома одиницями, наприклад "40 літрів у м³".

    Returns:
        Результат конвертації або пояснення, чого забракло в запиті.
    """
    return CONVERTER.explain(query)


@tool
@requires(query="Вкажи кількість тем, днів і годин на день.")
def plan_exam_prep(query: str) -> str:
    """Будує план підготовки до іспиту за кількістю тем, днів і годин на день.

    Використовуй, коли студент планує підготовку або питає, чи вистачає часу.

    Args:
        query: Запит із кількістю тем, днів і годин, наприклад
            "20 тем, 10 днів, 3 години на день".

    Returns:
        План з розрахунком навантаження або пояснення, яких даних бракує.
    """
    return PLANNER.build(query)


TOOLS = [formula_lookup, check_formula_for_task, convert_units, plan_exam_prep]
