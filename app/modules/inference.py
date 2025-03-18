import streamlit as st
import pickle
import glob
import pandas as pd
from utils.logger import logger

def app():
    st.title("Инференс с использованием обученной модели")
    # Получаем список файлов экспериментов (моделей) из папки models
    experiment_files = glob.glob("models/*.pkl")
    if not experiment_files:
        st.warning("Нет доступных моделей для инференса.")
        logger.warning("Нет доступных моделей для инференса в папке 'models'.")
        return

    selected_model_file = st.selectbox("Выберите модель для инференса", experiment_files)
    logger.info("Пользователь выбрал модель для инференса: %s", selected_model_file)
    with open(selected_model_file, "rb") as f:
        experiment = pickle.load(f)
    
    # Предполагаем, что эксперимент содержит ключ "pipeline" с обученным pipeline
    if "pipeline" not in experiment:
        st.error("В выбранном эксперименте не найден сохраненный pipeline для инференса.")
        logger.error("Отсутствует ключ 'pipeline' в эксперименте: %s", selected_model_file)
        return
    model_pipeline = experiment["pipeline"]

    st.write("**Выбранная модель:**", experiment["model"])
    st.write("**Гиперпараметры:**", experiment["params"])

    uploaded_file = st.file_uploader("Загрузите CSV с данными для инференса", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Данные для инференса:")
        st.dataframe(new_data)
        try:
            predictions = model_pipeline.predict(new_data)
            new_data["isRadiant_win"] = predictions
            st.write("Результаты инференса (столбец 'isRadiant_win' закреплен справа):")
            st.dataframe(new_data)
            logger.info("Инференс выполнен с моделью: %s", experiment["model"])
        except Exception as e:
            st.error("Ошибка при выполнении инференса: " + str(e))
            logger.error("Ошибка при инференсе с моделью %s: %s", experiment["model"], str(e))

if __name__ == "__main__":
    app()