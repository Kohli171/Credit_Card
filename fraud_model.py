import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================
# STEP 1: Load the Dataset
# ============================
df = pd.read_csv('/Users/kashishkohli/Downloads/creditcard.csv')  # adjust path if needed

print("✅ File loaded successfully!\n")
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ============================
# STEP 2: Basic Exploration
# ============================
print("\nMissing values in each column:\n", df.isnull().sum())
print("\nClass distribution (0 = Not Fraud, 1 = Fraud):\n", df['Class'].value_counts())

fraud_count = df['Class'].value_counts()[1]
total = df.shape[0]
print(f"\n⚠️ Percentage of fraud transactions: {round(fraud_count / total * 100, 4)}%")

# ============================
# STEP 3: Preprocessing
# ============================
data = df.copy()

# Scale 'Amount' and 'Time'
scaler = StandardScaler()
data[['scaled_amount', 'scaled_time']] = scaler.fit_transform(data[['Amount', 'Time']])
data.drop(['Amount', 'Time'], axis=1, inplace=True)

# Rearranging columns (optional)
columns = ['scaled_time', 'scaled_amount'] + [col for col in data.columns if col not in ['scaled_time', 'scaled_amount', 'Class']] + ['Class']
data = data[columns]

# Feature matrix and target
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\n✅ Data preprocessing complete.")
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# ============================
# STEP 4: Model Training
# ============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================
# STEP 5: Evaluation
# ============================
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]  # probabilities for positive class

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ============================
# STEP 6: Plot Precision-Recall Curve
# ============================
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision, color='green', linewidth=2, label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()