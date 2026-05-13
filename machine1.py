import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data
data = pd.read_csv('machine_data.csv')

# 2. Feature columns
feature_columns = ['temperature', 'vibration', 'pressure']

# 3. Features
X = data[feature_columns]

# Convert everything safely into numbers
X = X.apply(pd.to_numeric, errors='coerce')

# Fill missing values
X = X.fillna(X.mean())

# 4. Target column
y = data['status'].apply(
    lambda x: 1 if str(x).strip().upper() == 'HEALTHY' else 0
)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train)

# 7. Evaluate model
y_pred = clf.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=['FAULTY', 'HEALTHY']
    )
)

# 8. Prediction function
def predict_machine_health(sensor_dict):

    input_df = pd.DataFrame([sensor_dict])

    # Convert safely
    input_df = input_df.apply(
        pd.to_numeric,
        errors='coerce'
    )

    # Fill missing values
    input_df = input_df.fillna(0)

    pred = clf.predict(input_df)[0]

    return 'HEALTHY' if pred == 1 else 'FAULTY'

# 9. Example prediction
new_sensor_data = {
    'temperature': 75,
    'vibration': 2,
    'pressure': 30
}

status = predict_machine_health(new_sensor_data)

print("Predicted Machine Status:", status)