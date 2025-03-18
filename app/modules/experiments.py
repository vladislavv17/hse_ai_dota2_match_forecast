import streamlit as st
import pickle
import glob
import plotly.express as px
from utils.logger import logger

def app():
    st.title("Сравнение экспериментов")

    # Получаем список файлов экспериментов из папки models
    experiment_files = glob.glob("models/*.pkl")
    if not experiment_files:
        st.warning("Файлы экспериментов не найдены в папке 'models'.")
        logger.warning("Нет файлов экспериментов в папке 'models'.")
        return

    # По умолчанию выбираем все эксперименты
    selected_files = st.multiselect("Выберите эксперименты для сравнения", experiment_files, default=experiment_files)
    logger.info("Пользователь выбрал эксперименты для сравнения: %s", selected_files)
    if selected_files:
        fig = None
        for exp_file in selected_files:
            with open(exp_file, "rb") as f:
                experiment = pickle.load(f)
            lc = experiment["learning_curve"]
            exp_label = f"{experiment['model']} (ID {experiment['experiment_id']})"
            # Для сравнения оставляем только кривые валидации (train curve исключаем)
            if fig is None:
                fig = px.line(x=lc["train_sizes"], y=lc["val_scores"],
                              labels={"x": "Размер обучающей выборки", "y": "Score"},
                              title="Сравнение кривых обучения (валидация)")
                fig.add_scatter(x=lc["train_sizes"], y=lc["val_scores"], mode="lines", name=exp_label)
            else:
                fig.add_scatter(x=lc["train_sizes"], y=lc["val_scores"], mode="lines", name=exp_label)
        st.plotly_chart(fig)
        logger.info("Построена сравнительная кривая обучения (валидация) для выбранных экспериментов")
    else:
        st.info("Выберите хотя бы один эксперимент для сравнения.")
        logger.info("Пользователь не выбрал эксперименты для сравнения")

if __name__ == "__main__":
    app()