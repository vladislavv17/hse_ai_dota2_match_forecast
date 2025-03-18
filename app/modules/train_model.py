import streamlit as st
import os
import pickle
import datetime
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from utils.logger import logger
import plotly.express as px
import plotly.graph_objects as go

def app():
    st.title("Обучение/Дообучение модели")

    # Проверяем, что датасет загружен
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Сначала загрузите датасет в разделе 'Загрузка датасета'")
        return

    # Загружаем датафрейм из session_state
    df = st.session_state.df.copy()
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Фильтруем столбцы по паттернам
    patterns = [
        "draft",
        "account_id_",
        "party_id",
        "hero_variant",
        "name_",
        "isRadiant_",
        "rank_tier_",
        "game_mode",
        "lobby_type",
        "start_time",
        "lane_", 
        "is_roaming",
        "version",
        "series_type",
        "patch",
        "region",
        "radiant_win"
    ]
    cols_to_keep = [col for col in df.columns if any(pattern in col for pattern in patterns)]
    df = df[cols_to_keep]

    if 'radiant_win' not in df.columns:
        st.error("В датасете отсутствует столбец 'radiant_win'")
        return

    X = df.drop(columns=['radiant_win'])
    y = df['radiant_win']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Режим обучения: "Обучение с нуля" или "Дообучение существующей модели"
    mode = st.radio("Выберите режим обучения", ["Обучение с нуля", "Дообучение существующей модели"])
    st.write("Режим:", mode)

    if mode == "Обучение с нуля":
        st.subheader("Настройка модели")
        classifier_choice = st.selectbox("Выберите классификатор", 
                                         ["Logistic Regression", "Decision Tree", "SGDClassifier"])
        model_params = {}
        model_name = ""
        if classifier_choice == "Logistic Regression":
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            C = st.number_input("C (обратная регуляризация)", value=1.0, step=0.1)
            solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear", "newton-cg"])
            model_params = {"penalty": penalty, "C": C, "solver": solver, "max_iter": 200}
            model_name = "LogisticRegression"
        elif classifier_choice == "Decision Tree":
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            max_depth = st.slider("Max Depth", 1, 50, 10)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            model_params = {"criterion": criterion, "max_depth": max_depth, "min_samples_split": min_samples_split}
            model_name = "DecisionTreeClassifier"
        elif classifier_choice == "SGDClassifier":
            loss = st.selectbox("Loss", ["hinge", "log", "modified_huber", "squared_loss"])
            alpha = st.number_input("Alpha", value=0.0001, step=0.0001, format="%.5f")
            penalty = st.selectbox("Penalty", ["l2", "l1", "elasticnet"])
            learning_rate = st.selectbox("Learning rate", ["optimal", "constant", "invscaling"])
            model_params = {"loss": loss, "alpha": alpha, "penalty": penalty, "learning_rate": learning_rate, "max_iter": 1000, "tol": 1e-3}
            model_name = "SGDClassifier"

        if st.button("Запустить обучение"):
            with st.spinner("Идет обучение модели..."):
                if model_name == "LogisticRegression":
                    from sklearn.linear_model import LogisticRegression
                    classifier = LogisticRegression(**model_params)
                elif model_name == "DecisionTreeClassifier":
                    from sklearn.tree import DecisionTreeClassifier
                    classifier = DecisionTreeClassifier(**model_params)
                elif model_name == "SGDClassifier":
                    from sklearn.linear_model import SGDClassifier
                    classifier = SGDClassifier(**model_params)
                
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', classifier)
                ])
                
                # Вычисляем learning curve (10 точек)
                X_train_processed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
                train_sizes, train_scores, val_scores = learning_curve(
                    pipeline.named_steps['classifier'], X_train_processed, y_train,
                    cv=3, train_sizes=np.linspace(0.1, 1.0, 10)
                )
                train_scores_mean = np.mean(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                
                pipeline.fit(X_train, y_train)
                test_score = pipeline.score(X_test, y_test)
                y_pred = pipeline.predict(X_test)
                conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                experiment = {
                    "model": model_name,
                    "params": classifier.get_params(),
                    "learning_curve": {
                        "train_sizes": train_sizes.tolist(),
                        "train_scores": train_scores_mean.tolist(),
                        "val_scores": val_scores_mean.tolist(),
                    },
                    "test_score": test_score,
                    "confusion_matrix": conf_matrix,
                    "classification_report": class_report,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                pkl_bytes = pickle.dumps(experiment)
            st.success("Обучение завершено!")
            st.write("Доля правильных ответов (Accuracy):", test_score)
            precision = experiment["classification_report"].get('weighted avg', {}).get('precision', None)
            if precision:
                st.write("Точность (Precision):", precision)
            st.download_button("Скачать модель (pkl)", pkl_bytes, file_name=f"{model_name}_{experiment['experiment_id']}.pkl")
            logger.info("Обучена модель: %s с параметрами %s, режим: %s, Accuracy: %.4f", 
                        model_name, model_params, mode, test_score)
    else:  # Дообучение существующей модели
        st.subheader("Выберите модель для дообучения")
        available_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
        if not available_files:
            st.warning("Нет сохраненных моделей для дообучения.")
            st.stop()
        selected_file = st.selectbox("Выберите модель для дообучения", available_files)
        with open(os.path.join("models", selected_file), "rb") as f:
            experiment = pickle.load(f)
        if "pipeline" not in experiment:
            st.error("В выбранном эксперименте отсутствует сохранённый pipeline для дообучения.")
            st.stop()
        pipeline = experiment["pipeline"]
        st.write("Выбрана модель для дообучения:", experiment["model"])
        st.write("Гиперпараметры:", experiment["params"])
        if st.button("Запустить дообучение"):
            with st.spinner("Идет дообучение модели..."):
                pipeline.fit(X_train, y_train)
                test_score = pipeline.score(X_test, y_test)
                y_pred = pipeline.predict(X_test)
                conf_matrix = confusion_matrix(y_test, y_pred).tolist()
                class_report = classification_report(y_test, y_pred, output_dict=True)
                # Для learning curve используем только трансформированные данные
                X_train_processed = pipeline.named_steps['preprocessor'].transform(X_train)
                train_sizes, train_scores, val_scores = learning_curve(
                    pipeline.named_steps['classifier'], X_train_processed, y_train,
                    cv=3, train_sizes=np.linspace(0.1, 1.0, 10)
                )
                train_scores_mean = np.mean(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                experiment_update = {
                    "model": experiment["model"],
                    "params": pipeline.named_steps['classifier'].get_params(),
                    "learning_curve": {
                        "train_sizes": train_sizes.tolist(),
                        "train_scores": train_scores_mean.tolist(),
                        "val_scores": val_scores_mean.tolist(),
                    },
                    "test_score": test_score,
                    "confusion_matrix": conf_matrix,
                    "classification_report": class_report,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "experiment_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "pipeline": pipeline
                }
                pkl_bytes = pickle.dumps(experiment_update)
            st.success("Дообучение завершено!")
            st.write("Доля правильных ответов (Accuracy):", test_score)
            precision = experiment_update["classification_report"].get('weighted avg', {}).get('precision', None)
            if precision:
                st.write("Точность (Precision):", precision)
            st.download_button("Скачать дообученную модель (pkl)", pkl_bytes, file_name=f"{experiment['model']}_{experiment_update['experiment_id']}.pkl")
            logger.info("Дообучена модель: %s, исходные параметры: %s, режим: %s, Accuracy: %.4f", 
                        experiment["model"], experiment["params"], mode, test_score)

if __name__ == "__main__":
    app()