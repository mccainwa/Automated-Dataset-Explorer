import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def train_and_evaluate(df):
    import streamlit as st

    st.header("🤖 Train a Machine Learning Model")

    # 1. Column selection
    st.subheader("📌 Feature and Target Selection")
    all_columns = df.columns.tolist()

    target = st.selectbox("Select the target column (what you're trying to predict)", all_columns)
    features = st.multiselect("Select the feature columns (inputs to the model)", [col for col in all_columns if col != target])

    if (not features) or (not target):
        st.warning("Please select at least one feature and a target.")
        return None, None

    X = df[features]
    y = df[target]

    # 2. Train-test split
    test_size = st.slider("Test Set Size (%)", 10, 50, 20)
    random_state = int(st.number_input("Random State (for reproducibility)", value=42))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)

    # 3. Model selection
    st.subheader("⚙️ Choose a Model")
    model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression", "KNN", "XGBoost"])

    if model_name == "KNN":
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    else:
        model = RandomForestClassifier(random_state=random_state)

    # 4. Train model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # 5. Evaluation
    st.subheader("📊 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**Precision (weighted):**", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    st.write("**Recall (weighted):**", recall_score(y_test, y_pred, average="weighted"))
    st.write("**F1 Score (weighted):**", f1_score(y_test, y_pred, average="weighted"))

    # 6. Feature Importance
    st.subheader("🔍 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df)

        fig2, ax2 = plt.subplots(figsize=(6, max(3, 0.3 * len(importance_df))))
        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.set_xlabel("Importance Score")
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("This model does not support built-in feature importance.")

    # 7. Confusion Matrix
    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # 8. SHAP Explanation (optional)
    st.subheader("🧠 SHAP Explanation (Optional)")
    shap_fig = None

    if not SHAP_AVAILABLE:
        st.warning("SHAP library not installed. Run `pip install shap` to enable SHAP explanations.")
        return model, shap_fig

    if model_name not in ["Random Forest", "XGBoost"]:
        st.info("SHAP explanations available only for tree-based models (Random Forest, XGBoost).")
        return model, shap_fig

    try:
        max_shap_samples = 200
        if X_test.shape[0] > max_shap_samples:
            X_shap = X_test.sample(n=max_shap_samples, random_state=1)
        else:
            X_shap = X_test.copy()

        explainer = shap.TreeExplainer(model, check_additivity=False)
        shap_values = explainer.shap_values(X_shap)

        shap_fig = plt.figure(figsize=(8, 6))
        if isinstance(shap_values, list):  # multiclass
            shap.summary_plot(shap_values, X_shap, show=False)
        else:
            shap.summary_plot(shap_values, X_shap, show=False)
        plt.tight_layout()
        st.pyplot(shap_fig)
    except Exception as e:
        st.warning(f"SHAP explanation not available for this model. Error: {e}")

    return model, shap_fig
