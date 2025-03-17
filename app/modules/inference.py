import streamlit as st
import pandas as pd
import numpy as np
from utils.logger import logger

def app():
    st.title("Инференс с обученной моделью")
    if 'model' not in st.session_state or st.session_state.model is None:
        st.warning("Модель не обучена. Сначала обучите модель в разделе 'Обучение модели'")
        return
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Сначала загрузите датасет в разделе 'Загрузка датасета'")
        return

    df = st.session_state.df
    # Используем те же числовые признаки, что и при обучении
    features = df.drop(columns=["radiant_win"]).select_dtypes(include=[np.number]).columns.tolist()
    st.write("Введите значения для следующих признаков:")
    input_data = {}
    for col in features:
        default_val = float(df[col].mean()) if not np.isnan(df[col].mean()) else 0.0
        input_data[col] = st.number_input(col, value=default_val)
    if st.button("Сделать предсказание"):
        input_df = pd.DataFrame([input_data])
        pred = st.session_state.model.predict(input_df)
        st.success(f"Предсказание (radiant_win): {bool(pred[0])}")
        logger.info("Инференс выполнен. Входные данные: %s, результат: %s", input_data, bool(pred[0]))
