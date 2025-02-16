from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import numpy as np

# Load and prep data as before
data = joblib.load('../preprocessed_data.joblib')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']

# Select and scale features
top_features = [0, 1, 2]
X_train_top = X_train[:, top_features]
X_val_top = X_val[:, top_features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top)
X_val_scaled = scaler.transform(X_val_top)
# Add feature interactions
X_train_interact = np.column_stack([
    X_train_scaled,
    X_train_scaled[:, 1] * X_train_scaled[:, 2],  # Interaction between Feature 1 and 2
    X_train_scaled[:, 0] * X_train_scaled[:, 1],  # Interaction between Feature 0 and 1
    X_train_scaled[:, 0] * X_train_scaled[:, 2]  # Interaction between Feature 0 and 2
])

X_val_interact = np.column_stack([
    X_val_scaled,
    X_val_scaled[:, 1] * X_val_scaled[:, 2],
    X_val_scaled[:, 0] * X_val_scaled[:, 1],
    X_val_scaled[:, 0] * X_val_scaled[:, 2]
])

# Modified model
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,  # Further reduced
    min_samples_split=30,
    min_samples_leaf=15,
    class_weight={0: 1, 1: 3},  # Slightly reduced class weight
    random_state=42
)

rf_model.fit(X_train_interact, y_train)
y_pred = rf_model.predict(X_val_interact)
y_pred_proba = rf_model.predict_proba(X_val_interact)[:, 1]

print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print(f"\nROC AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}")

# Feature importance including interactions
importance = rf_model.feature_importances_
feature_names = ['Feature_0', 'Feature_1', 'Feature_2',
                 'F1*F2_interact', 'F0*F1_interact', 'F0*F2_interact']

for name, imp in zip(feature_names, importance):
    print(f"{name} importance: {imp:.4f}")

