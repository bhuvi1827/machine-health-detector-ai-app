import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import io
from sklearn.utils.multiclass import unique_labels

st.set_page_config(page_title="Machine Health Detector", page_icon="🦾", layout="centered")
st.title("🦾 Machine Health Detector AI App")

st.write("""
Upload your machine's sensor CSV data, train an ML model, see accuracy, predict on new data, and download the model or predictions!
""")

uploaded_file = st.file_uploader("Upload your machine sensor data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### First few rows of your data:", data.head())
    with st.expander("Show Data Columns / Structure"):
        st.write(data.info())

    all_columns = list(data.columns)
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Choose Result Label column", all_columns, index=len(all_columns)-1)
    with col2:
        feature_columns = st.multiselect("Select feature columns", [c for c in all_columns if c != target_column], default=[c for c in all_columns if c != target_column])

    model_option = st.selectbox("Choose ML Model", ("Random Forest", "SVM", "Logistic Regression"))
    predictions = None    # To save output

    if len(feature_columns) < 1 or not target_column:
        st.warning("Please select features and result column.")
    else:
        # Prepare
        X = data[feature_columns]
        y = data[target_column].apply(lambda x: 1 if str(x).lower() == 'healthy' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection
        if model_option == "Random Forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_option == "SVM":
            clf = SVC(probability=True)
        else:
            clf = LogisticRegression(max_iter=1000)
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model accuracy: {accuracy*100:.2f}%")

        with st.expander("Show Classification Report"):
            labels = unique_labels(y_test, y_pred)
            names_map = {0: "FAULTY", 1: "HEALTHY"}
            target_names = [names_map[l] for l in labels]
            st.text(classification_report(y_test, y_pred, target_names=target_names))

        # Feature importance (only for tree-based classifiers)
        if hasattr(clf, 'feature_importances_'):
            st.write("#### Feature Importance")
            fig, ax = plt.subplots()
            pd.Series(clf.feature_importances_, index=feature_columns).sort_values().plot(
                kind='barh', ax=ax, color='skyblue')
            st.pyplot(fig)

        # Download classifier button
        buf = io.BytesIO()
        joblib.dump(clf, buf)
        buf.seek(0)
        st.download_button(
            "Download Trained Model (.pkl)",
            data=buf,
            file_name=f"{model_option.lower().replace(' ', '_')}_classifier.pkl",
            mime="application/octet-stream"
        )

        # Custom data prediction
        st.write("---")
        st.header("Try a Prediction")
        input_data = {}
        for col in feature_columns:
            val = st.number_input(
                f"Enter value for {col}",
                float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            input_data[col] = val

        if st.button("Predict Machine Health"):
            input_df = pd.DataFrame([input_data])
            result = clf.predict(input_df)[0]
            st.markdown(f"## Prediction: {'🟢 HEALTHY' if result == 1 else '🔴 FAULTY'}", unsafe_allow_html=True)
            predictions = pd.DataFrame([input_data])
            predictions['Prediction'] = 'HEALTHY' if result == 1 else 'FAULTY'
        else:
            predictions = None

        # Download predictions (if available)
        if predictions is not None:
            out_csv = predictions.to_csv(index=False)
            st.download_button(
                "Download Prediction as CSV",
                data=out_csv,
                file_name="your_prediction.csv",
                mime="text/csv"
            )
        st.write("---")
        st.write("Tip: You can retrain with a new file or different features anytime!")

else:
    st.info("👆 Upload your CSV file to get started!")

st.markdown("---\nMade for AIML Internship 🚀")