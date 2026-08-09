"""StudyMate: освітній асистент, який не вигадує формул.

Пакет розбитий за відповідальностями, і межі проведені свідомо:

    models.py         типи даних, без логіки й без залежностей
    data.py           довідник формул і таблиці перетворень
    text.py           стемінг і міра схожості, чисті функції
    search.py         лексичний, семантичний шари та їхнє злиття
    applicability.py  фільтр застосовності: рішення ухвалює код, не модель
    converters.py     переведення одиниць
    planner.py        планування підготовки
    tools.py          тонкі @tool-обгортки над готовими шарами
    agent.py          агент, системний промпт і діалог з контекстом

Швидкий старт:

    from studymate import StudyMateAgent, SEARCH

    SEARCH.build()                 # опційно: вмикає семантичний шар
    agent = StudyMateAgent()
    print(agent.ask("Яка формула кінетичної енергії?").answer)

Інструменти можна використовувати й окремо від агента, без ключа і без мережі:

    from studymate import formula_lookup
    print(formula_lookup.invoke({"query": "кінетична енергія"}))
"""
from .agent import ERROR_PREFIX, SYSTEM_PROMPT, StudyMateAgent, Turn
from .applicability import ApplicabilityChecker, detect_conditions
from .converters import UnitConverter
from .data import FORMULAS, FORMULAS_BY_UID, PHYSICS_CONVERSIONS, SUBJECTS
from .models import ApplicabilityVerdict, Formula, Resolution, RetrievalResult
from .planner import StudyPlanner
from .search import HybridSearch, LexicalSearch, SemanticSearch
from .text import format_number, lexical_score, stem_word, stems
from .tools import (CHECKER, CONVERTER, PLANNER, SEARCH, TOOLS,
                    check_formula_for_task, convert_units, formula_lookup,
                    plan_exam_prep, render_card)

__all__ = [
    "StudyMateAgent", "Turn", "SYSTEM_PROMPT", "ERROR_PREFIX",
    "Formula", "RetrievalResult", "ApplicabilityVerdict", "Resolution",
    "FORMULAS", "FORMULAS_BY_UID", "PHYSICS_CONVERSIONS", "SUBJECTS",
    "LexicalSearch", "SemanticSearch", "HybridSearch",
    "ApplicabilityChecker", "detect_conditions",
    "UnitConverter", "StudyPlanner",
    "TOOLS", "formula_lookup", "check_formula_for_task", "convert_units",
    "plan_exam_prep", "render_card",
    "SEARCH", "CHECKER", "CONVERTER", "PLANNER",
    "stems", "stem_word", "lexical_score", "format_number",
]
