import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("data/loan_data.csv", header=1)
print(data.head())
print("\nDataset Shape:")
print(data.shape)
print("\nColumn Names:")
print(data.columns)
# -------------------------------
# Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(data.isnull().sum())

# -------------------------------
# Rename Target Column
# -------------------------------
data = data.rename(
    columns={"default payment next month": "Default"}
)

print("\nUpdated Columns:")
print(data.columns)

# -------------------------------
# Separate Features and Target
# -------------------------------
X = data.drop(["Default", "ID"], axis=1)
y = data["Default"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# -------------------------------
# Train-Test Split  ✅ FIRST
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# -------------------------------
# Feature Scaling  ✅ SECOND
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Feature Selection  ✅ THIRD
# -------------------------------
selector = SelectKBest(score_func=f_classif, k=15)

X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# -------------------------------
# Logistic Regression Model
# -------------------------------
model = LogisticRegression(
    max_iter=5000,
    solver="lbfgs",
    C=2.0
)

model.fit(X_train, y_train)
print("\nModel Training Completed ✅")
# -------------------------------
# Predictions
# -------------------------------
y_pred = model.predict(X_test)
# -------------------------------
# Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
y_prob = model.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", roc_score)
import joblib
# Save model and scaler
import joblib

joblib.dump(model, "model/loan_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(selector, "model/selector.pkl")

print("\nModel Saved Successfully ✅")

