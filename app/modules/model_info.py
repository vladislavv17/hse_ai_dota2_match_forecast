# pages/model_info.py
import streamlit as st
import pickle
import glob
import plotly.express as px
import plotly.graph_objects as go
from utils.logger import logger

def app():
    st.title("Информация о модели и кривые обучения")

    # Получаем список файлов экспериментов из папки models
    experiment_files = glob.glob("models/*.pkl")
    if not experiment_files:
        st.warning("Файлы экспериментов не найдены в папке 'models'.")
        logger.warning("Нет файлов экспериментов в папке 'models'.")
        return

    selected_file = st.selectbox("Выберите эксперимент", experiment_files)
    logger.info("Пользователь выбрал эксперимент: %s", selected_file)
    with open(selected_file, "rb") as f:
        experiment = pickle.load(f)

    st.write("**Модель:**", experiment["model"])
    st.write("**Гиперпараметры:**", experiment["params"])
    st.write("**Доля правильных ответов (Accuracy):**", experiment["test_score"])
    # Из отчёта по классификации извлекаем precision (weighted average)
    precision = experiment["classification_report"].get('weighted avg', {}).get('precision', None)
    if precision is not None:
        st.write("**Точность (Precision):**", precision)
    else:
        st.write("**Отчет по классификации:**", experiment["classification_report"])
    st.write("**Время эксперимента:**", experiment["timestamp"])
    st.write("**ID эксперимента:**", experiment["experiment_id"])
    logger.info("Отображены детали эксперимента ID %s", experiment["experiment_id"])

    # Построение матрицы ошибок с аннотациями (для бинарной классификации)
    cm = experiment["confusion_matrix"]
    if len(cm) == 2 and len(cm[0]) == 2:
        TN, FP = cm[0]
        FN, TP = cm[1]
        annotations = [[f"TN\n{TN}", f"FP\n{FP}"],
                       [f"FN\n{FN}", f"TP\n{TP}"]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale='Blues',
            showscale=True
        ))
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(dict(
                    x=["Predicted Negative", "Predicted Positive"][j],
                    y=["Actual Negative", "Actual Positive"][i],
                    text=annotations[i][j],
                    showarrow=False,
                    font=dict(color="black")
                ))
        fig_cm.update_layout(title="Матрица ошибок")
    else:
        fig_cm = px.imshow(cm,
                           text_auto=True,
                           color_continuous_scale='Blues',
                           labels={'x': "Predicted", 'y': "Actual"},
                           title="Матрица ошибок")
    st.plotly_chart(fig_cm)
    logger.info("Отображена матрица ошибок для эксперимента ID %s", experiment["experiment_id"])

    # Построение кривой обучения (отображаем линии для train и validation)
    lc = experiment["learning_curve"]
    fig = px.line(x=lc["train_sizes"], y=lc["train_scores"],
                  labels={"x": "Размер обучающей выборки", "y": "Score"},
                  title="Кривая обучения")
    fig.add_scatter(x=lc["train_sizes"], y=lc["val_scores"], mode="lines", name="Validation")
    fig.add_scatter(x=lc["train_sizes"], y=lc["train_scores"], mode="lines", name="Train")
    st.plotly_chart(fig)
    logger.info("Построена кривая обучения для эксперимента ID %s", experiment["experiment_id"])

if __name__ == "__main__":
    app()