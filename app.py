import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Machine Health Detector", page_icon="🦾", layout="centered")

st.title("🦾 Machine Health Detector AI App")
st.write("""
A simple AI-powered app to detect if your machine is 'HEALTHY' or 'FAULTY' using sensor data and machine learning.
Upload your CSV, train the model, and try predictions!
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your machine sensor data CSV", type=["csv"])

if uploaded_file:
    # 1. Read Data
    data = pd.read_csv(uploaded_file)
    st.write("### First few rows of your data:", data.head())

    with st.expander("Show Data Columns / Structure"):
        st.write(data.info())

    # 2. Select Features and Target
    all_columns = list(data.columns)
    col1, col2 = st.columns(2)
    with col1:
        target_column = st.selectbox("Choose your Result/Label column", all_columns, index=len(all_columns)-1)
    with col2:
        feature_columns = st.multiselect("Select sensor feature columns", [c for c in all_columns if c != target_column], default=[c for c in all_columns if c != target_column])
    
    if len(feature_columns) < 1 or not target_column:
        st.warning("Please select features and result column to proceed.")
    else:
        # 3. Prepare data
        X = data[feature_columns]
        y = data[target_column].apply(lambda x: 1 if str(x).lower() == 'healthy' else 0)  # positive=HEALTHY

        # 4. Split & Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # 5: Evaluate/Test Results
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"Model Accuracy: {accuracy*100:.2f}%")
        with st.expander("Show Classification Report"):
            st.text(classification_report(y_test, y_pred, target_names=['FAULTY', 'HEALTHY']))

        # 6. Feature Importance (plot)
        st.write("#### Feature Importance")
        fig, ax = plt.subplots()
        pd.Series(clf.feature_importances_, index=feature_columns).sort_values().plot(kind='barh', ax=ax, color='skyblue')
        st.pyplot(fig)

        # 7. Try your own prediction!
        st.write("---")
        st.header("Try a Prediction")
        input_data = {}
        for col in feature_columns:
            val = st.number_input(f"Enter value for {col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            input_data[col] = val
        if st.button("Predict Machine Health"):
            input_df = pd.DataFrame([input_data])
            result = clf.predict(input_df)[0]
            st.markdown(f"## Prediction: {'🟢 HEALTHY' if result == 1 else '🔴 FAULTY'}", unsafe_allow_html=True)

        st.write("---")
        st.write("*Tip: You can retrain with a new file or different features anytime!*")

else:
    st.info("👆 Upload your CSV file to get started!")

st.markdown("---\nMade by [Your Name] for AIML Internship 🚀")