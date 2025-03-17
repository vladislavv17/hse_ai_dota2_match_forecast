import streamlit as st
import plotly.express as px
from utils.logger import logger

def app():
    st.title("Информация о модели и кривые обучения")
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Модель не обучена. Перейдите в раздел 'Обучение модели'")
        return

    st.subheader("Параметры модели")
    st.write(st.session_state.model)
    
    st.subheader("Сохранённые эксперименты")
    if 'experiment_results' in st.session_state and st.session_state.experiment_results:
        exp_names = list(st.session_state.experiment_results.keys())
        selected_exp = st.selectbox("Выберите эксперимент", exp_names)
        exp_data = st.session_state.experiment_results[selected_exp]
        st.write("Гиперпараметры:", exp_data["params"])
        st.write("Точность модели:", exp_data["accuracy"])
        fig = px.line(x=exp_data["train_sizes"], y=exp_data["train_scores"],
                      labels={"x": "Размер обучающей выборки", "y": "Score"},
                      title=f"Кривая обучения ({selected_exp})")
        fig.add_scatter(x=exp_data["train_sizes"], y=exp_data["val_scores"], mode='lines', name="Валидация")
        st.plotly_chart(fig)
        logger.info("Отображена информация для эксперимента %s", selected_exp)
    else:
        st.info("Нет сохранённых экспериментов.")