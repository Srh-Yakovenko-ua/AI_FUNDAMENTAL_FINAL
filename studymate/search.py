"""Пошук формул: лексичний шар, семантичний шар і їхнє злиття.

Кожен шар оформлено окремим класом з власною відповідальністю, а `HybridSearch`
лише поєднує їх. Завдяки цьому семантичний шар можна вимкнути, підмінити
чи протестувати окремо, не чіпаючи решту системи.
"""
from typing import Optional, Sequence

import numpy as np

from .data import FORMULAS, FORMULAS_BY_UID, SUBJECTS
from .models import Formula, Resolution, RetrievalResult
from .text import lexical_score

LEXICAL_THRESHOLD = 0.6      # нижче цього збіг вважається слабким
AMBIGUITY_TOLERANCE = 0.1    # ближче за це два кандидати вважаються рівними
MIN_QUERY_LENGTH = 4         # коротший запит не несе змісту
RRF_K = 60                   # стандартна константа згладжування


class LexicalSearch:
    """Пошук за словами. Працює завжди: без мережі, без ключа, без затримки."""

    def __init__(self, formulas: Sequence[Formula] = FORMULAS):
        self.formulas = list(formulas)

    def search(self, query: str, top_k: int = 5) -> list:
        """Повертає [(uid, score), ...] за спаданням схожості.

        Оцінка береться як найкраща серед назви та синонімів: студент рідко
        називає формулу так, як вона записана в довіднику.
        """
        scored = []
        for formula in self.formulas:
            best = max([lexical_score(query, formula.name)]
                       + [lexical_score(query, s) for s in formula.synonyms])
            if best > 0:
                scored.append((formula.uid, best))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class SemanticSearch:
    """Пошук за змістом на ембеддінгах OpenAI.

    Будується ліниво і вимикається сам, якщо ключа немає або API недоступне.
    Деградація має бути явною: система повідомляє, що працює лише на лексиці,
    а не тихо гіршає.
    """

    def __init__(self, formulas: Sequence[Formula] = FORMULAS,
                 model: str = "text-embedding-3-small"):
        self.formulas = list(formulas)
        self.model = model
        self.available = False
        self.tokens_used = 0
        self._vectors = None
        self._client = None

    def build(self) -> bool:
        """Рахує ембеддінги бази. Повертає True, якщо шар доступний."""
        try:
            from openai import OpenAI

            self._client = OpenAI()
            # Ембеддимо назву, синоніми і примітку: саме вони несуть зміст,
            # а не сам вираз формули, що складається з символів.
            texts = [f"{f.name}. {' '.join(f.synonyms)}. {f.note}".strip()
                     for f in self.formulas]
            response = self._client.embeddings.create(model=self.model, input=texts)
            self.tokens_used += response.usage.total_tokens
            self._vectors = np.array([item.embedding for item in response.data])
            self.available = True
        except Exception:
            self.available = False
        return self.available

    def search(self, query: str, top_k: int = 5) -> list:
        """Пошук за змістом. Порожній список, якщо шар недоступний."""
        if not self.available:
            return []
        try:
            response = self._client.embeddings.create(model=self.model, input=[query])
            self.tokens_used += response.usage.total_tokens
            vector = np.array(response.data[0].embedding)
            # Вектори OpenAI нормалізовані, тому косинус це скалярний добуток.
            similarities = self._vectors @ vector
            order = np.argsort(similarities)[::-1][:top_k]
            return [(self.formulas[i].uid, float(similarities[i])) for i in order]
        except Exception:
            return []


def reciprocal_rank_fusion(*rankings: list, k: int = RRF_K) -> list:
    """Зливає кілька ранжувань за формулою score = Σ 1/(k + rank).

    Перевага перед складанням оцінок у тому, що шкали шарів не треба зводити
    докупи: RRF працює з порядком, а не зі значеннями.
    """
    scores = {}
    for ranking in rankings:
        for rank, (uid, _) in enumerate(ranking, start=1):
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridSearch:
    """Поєднує лексичний і семантичний шари та вирішує, чи можна вірити знайденому."""

    def __init__(self, lexical: Optional[LexicalSearch] = None,
                 semantic: Optional[SemanticSearch] = None):
        self.lexical = lexical or LexicalSearch()
        self.semantic = semantic or SemanticSearch()

    def build(self) -> bool:
        return self.semantic.build()

    def search(self, query: str, top_k: int = 3) -> list:
        """Знаходить кандидатів і проставляє кожному прапорець впевненості."""
        lex = self.lexical.search(query, top_k=5)
        sem = self.semantic.search(query, top_k=5)

        lex_uids, sem_uids = {u for u, _ in lex}, {u for u, _ in sem}
        lex_scores = dict(lex)
        fused = reciprocal_rank_fusion(lex, sem) if sem else lex

        results = []
        for uid, score in fused[:top_k]:
            source = ("both" if uid in lex_uids and uid in sem_uids
                      else "lexical" if uid in lex_uids else "semantic")
            lex_score = lex_scores.get(uid, 0.0)
            results.append(RetrievalResult(
                formula=FORMULAS_BY_UID[uid],
                score=score,
                found_by=source,
                lexical_score=lex_score,
                # Впевненість дає або сильний лексичний збіг, або підтвердження
                # обома шарами: коли на картку вказали і слова, і зміст.
                confident=lex_score >= LEXICAL_THRESHOLD or source == "both",
            ))
        return results

    @staticmethod
    def is_ambiguous(results: Sequence[RetrievalResult],
                     tolerance: float = AMBIGUITY_TOLERANCE) -> bool:
        """Чи занадто близькі два найкращі результати, щоб обирати самостійно.

        Нічия це не збій, а сигнал. Запит «формула дальності польоту» однаково
        описує кидання з рівної поверхні й кидання з висоти, а різниця між ними
        вирішальна: одна формула для іншої задачі дає відповідь, меншу в півтора
        рази. Там, де система не має підстав для вибору, вона має перепитати.
        """
        if len(results) < 2:
            return False
        return abs(results[0].lexical_score - results[1].lexical_score) < tolerance

    def resolve(self, query: str, top_k: int = 3) -> Resolution:
        """Шукає формулу і вирішує, чи можна віддавати її як відповідь.

        Ця логіка потрібна двом інструментам: пошуку і перевірці застосовності.
        Коли вона була продубльована, гейт впевненості стояв лише в одному з них,
        і другий на запит «закон збереження імпульсу» впевнено віддавав закон Ома.
        Спільний резолвер робить таке розходження неможливим.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return Resolution(message=(
                f"Формули '{query}' немає в базі StudyMate.\n"
                f"Доступні предмети: {', '.join(SUBJECTS)}.\n"
                "Я не можу навести формулу, якої немає в базі."
            ))

        # Порядок перевірок принциповий: спершу впевненість, потім нічия.
        # Інакше слабкий збіг із двома однаково поганими кандидатами виглядає
        # як «є кілька варіантів», хоча насправді немає жодного.
        if not results[0].confident:
            options = "\n".join(f"  • {r.formula.name} ({r.formula.subject})"
                                for r in results)
            return Resolution(message=(
                f"Точного збігу за запитом '{query}' немає.\n"
                f"Найближче за змістом:\n{options}\n\n"
                "Якщо потрібної формули тут немає, значить її немає в базі."
            ), options=results)

        if self.is_ambiguous(results):
            close = [r for r in results
                     if abs(r.lexical_score - results[0].lexical_score) < AMBIGUITY_TOLERANCE]
            return Resolution(options=close)

        return Resolution(formula=results[0].formula, found_by=results[0].found_by)
