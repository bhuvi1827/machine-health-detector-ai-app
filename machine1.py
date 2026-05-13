import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load your data
data = pd.read_csv('machine_data.csv')

# 2. Features and target
feature_columns = ['temperature', 'vibration', 'pressure']  # add your real sensor columns here
X = data[feature_columns]
y = data['status'].apply(lambda x: 1 if x == 'HEALTHY' else 0)  # 1=HEALTHY, 0=FAULTY

# 3. Split the data for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['FAULTY', 'HEALTHY']))

# 6. Use on new/unseen data
def predict_machine_health(sensor_dict):
    # sensor_dict = {'temperature': 75, 'vibration': 2, 'pressure': 30}
    input_df = pd.DataFrame([sensor_dict])
    pred = clf.predict(input_df)[0]
    return 'HEALTHY' if pred == 1 else 'FAULTY'

# Example Prediction
new_sensor_data = {'temperature': 75, 'vibration': 2, 'pressure': 30}
status = predict_machine_health(new_sensor_data)
print("Predicted Machine Status:", status)