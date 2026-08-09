"""StudyMate: веб-інтерфейс на Streamlit.

Запуск:
    export OPENAI_API_KEY="ваш ключ"
    streamlit run app.py

Інтерфейс навмисно показує не лише відповідь, а й ЯК вона отримана: які інструменти
викликано і що вирішив фільтр застосовності. Для освітнього продукту походження
відповіді важливіше за її формулювання, тому це винесено в саму взаємодію,
а не сховано в логи.
"""
import os

import streamlit as st

st.set_page_config(page_title="StudyMate", page_icon="📐", layout="wide")


# --- Ключ -------------------------------------------------------------------
def ensure_api_key() -> bool:
    """Ключ беремо зі змінної середовища або зі скриньки Streamlit."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    key = st.sidebar.text_input("OpenAI API key", type="password",
                                help="Ключ не зберігається, живе лише в цій сесії.")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        return True
    return False


st.title("📐 StudyMate")
st.caption("Освітній асистент з точних наук. Формули приходять з перевіреної бази, "
           "а придатність формули для твоєї задачі перевіряє код, а не модель.")

if not ensure_api_key():
    st.warning("Введи ключ OpenAI у бічній панелі, щоб почати.")
    st.stop()


# --- Ядро -------------------------------------------------------------------
@st.cache_resource(show_spinner="Готую базу формул і пошук...")
def load_core():
    """Ядро вантажимо один раз на сесію: ембеддінги бази не варто рахувати щоразу."""
    import studymate_core as core
    # build() викликається один раз на сесію завдяки cache_resource.
    # Модуль сам його не викликає, щоб імпорт лишався дешевим і не потребував мережі.
    core.semantic.build()
    return core


core = load_core()

# --- Бічна панель: стан системи ---------------------------------------------
with st.sidebar:
    st.subheader("Стан системи")
    st.metric("Формул у базі", len(core.FORMULAS))
    st.metric("З умовами застосовності", sum(1 for f in core.FORMULAS if f.predicates))
    st.write("**Пошук:**", "гібридний (лексика + семантика)" if core.semantic.available
             else "лексичний (семантика недоступна)")
    st.write("**Модель:**", core.LLM_MODEL)

    st.divider()
    st.subheader("Спробуй запитати")
    examples = [
        "Яка формула кінетичної енергії?",
        "Кидаю камінь з даху 12 м під кутом 30°. Чи підійде формула дальності польоту?",
        "Скільки кубічних метрів у 40 літрах?",
        "У мене 20 тем і 10 днів по 3 години. Встигну?",
        "Яка формула ентропії Гіббса?",
    ]
    for i, example in enumerate(examples):
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example

    st.divider()
    if st.button("Очистити діалог", use_container_width=True):
        st.session_state.messages = []
        st.session_state.visible = []
        st.rerun()

# --- Стан діалогу -----------------------------------------------------------
st.session_state.setdefault("messages", [])   # історія для агента
st.session_state.setdefault("visible", [])    # історія для показу

for item in st.session_state.visible:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("tools"):
            st.caption(f"🔧 Викликано: {', '.join(item['tools'])}")

prompt = st.chat_input("Запитай про формулу, одиниці або план підготовки")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.visible.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Шукаю в базі..."):
            answer, history, tools_used = core.ask(
                prompt, st.session_state.messages, verbose=False
            )
        st.session_state.messages = history
        st.markdown(answer)
        if tools_used:
            st.caption(f"🔧 Викликано: {', '.join(tools_used)}")
        else:
            # Порожній список це теж інформація: відповідь дана без звернення до бази.
            st.caption("🔧 Інструменти не викликались: відповідь без звернення до бази")

    st.session_state.visible.append(
        {"role": "assistant", "content": answer, "tools": tools_used}
    )
