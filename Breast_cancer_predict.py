
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Import dataset
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print("Dataset:", X.shape)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define algorithms
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier())
}

# 4. Train and compare
results = []
trained = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    trained[name] = model

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_test, pred) * 100, 2)
    })

result = pd.DataFrame(results).sort_values(
    "Accuracy", ascending=False
)

print("\nAlgorithm Comparison:")
print(result.to_string(index=False))

# 5. Accuracy graph
plt.figure(figsize=(9, 5))
plt.bar(result["Algorithm"], result["Accuracy"])
plt.ylabel("Accuracy (%)")
plt.title("Breast Cancer Algorithm Comparison")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# 6. Best model
best_name = result.iloc[0]["Algorithm"]
best_model = trained[best_name]

print("\nBest Model:", best_name)
print("\nClassification Report:")
print(classification_report(
    y_test, best_model.predict(X_test),
    target_names=data.target_names
))

# 7. Confusion matrix
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=data.target_names
)
plt.title("Best Model Confusion Matrix")
plt.show()
