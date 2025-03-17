import streamlit as st
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils.logger import logger

def app():
    st.title("Выбор модели и гиперпараметров")
    st.write("Выберите модель, настройте гиперпараметры и определите стратегию обучения.")

    # Выбор стратегии обучения: дообучение или новое обучение
    training_strategy = st.radio("Выберите стратегию обучения", 
                                 ["Дообучение существующей модели", "Обучение с нуля"])

    st.sidebar.title("Настройки модели")
    classifier_choice = st.sidebar.selectbox("Выберите классификатор", 
                                               ["Logistic Regression", "Decision Tree", "Random Forest", "SGDClassifier"])
    
    if classifier_choice == "Logistic Regression":
        penalty = st.sidebar.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
        C = st.sidebar.number_input("C (обратная регуляризация)", value=1.0, step=0.1)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "saga", "liblinear", "newton-cg"])
        model_params = {"penalty": penalty, "C": C, "solver": solver}
        model_name = "Logistic Regression"
    
    elif classifier_choice == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        model_params = {"criterion": criterion, "max_depth": max_depth, "min_samples_split": min_samples_split}
        model_name = "Decision Tree"
    
    elif classifier_choice == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        max_depth = st.sidebar.slider("Max Depth", 1, 50, 10)
        min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}
        model_name = "Random Forest"
    
    elif classifier_choice == "SGDClassifier":
        loss = st.sidebar.selectbox("Loss", ["hinge", "log", "modified_huber", "squared_loss"])
        alpha = st.sidebar.number_input("Alpha", value=0.0001, step=0.0001, format="%.5f")
        penalty = st.sidebar.selectbox("Penalty", ["l2", "l1", "elasticnet"])
        learning_rate = st.sidebar.selectbox("Learning rate", ["optimal", "constant", "invscaling"])
        model_params = {"loss": loss, "alpha": alpha, "penalty": penalty, "learning_rate": learning_rate}
        model_name = "SGDClassifier"
    
    if st.button("Выбрать модель"):
        # Сохраняем выбранную информацию в session_state
        st.session_state.model_info = {
            "model": model_name, 
            "params": model_params,
            "training_strategy": training_strategy
        }
        st.success("Модель выбрана!")
        logger.info("Модель выбрана: %s с параметрами %s, стратегия обучения: %s", 
                    model_name, model_params, training_strategy)
        st.write("Выбранная модель:", model_name)
        st.write("Гиперпараметры:", model_params)
        st.write("Стратегия обучения:", training_strategy)