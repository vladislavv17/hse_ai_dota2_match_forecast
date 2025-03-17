import streamlit as st
import plotly.express as px
from utils.logger import logger

def app():
    st.title("Сравнение экспериментов")
    if 'experiment_results' not in st.session_state or not st.session_state.experiment_results:
        st.info("Экспериментов для сравнения ещё нет.")
        return

    exp_names = list(st.session_state.experiment_results.keys())
    selected_exps = st.multiselect("Выберите эксперименты для сравнения", exp_names)
    if selected_exps:
        fig = None
        for exp in selected_exps:
            data = st.session_state.experiment_results[exp]
            if fig is None:
                fig = px.line(x=data["train_sizes"], y=data["train_scores"],
                              labels={"x": "Размер обучающей выборки", "y": "Score"},
                              title="Сравнение кривых обучения")
                fig.add_scatter(x=data["train_sizes"], y=data["val_scores"], mode='lines', name=f"{exp} (валидация)")
                fig.add_scatter(x=data["train_sizes"], y=data["train_scores"], mode='lines', name=f"{exp} (обучение)")
            else:
                fig.add_scatter(x=data["train_sizes"], y=data["train_scores"], mode='lines', name=f"{exp} (обучение)")
                fig.add_scatter(x=data["train_sizes"], y=data["val_scores"], mode='lines', name=f"{exp} (валидация)")
        st.plotly_chart(fig)
        logger.info("Сравнение экспериментов: %s", selected_exps)
