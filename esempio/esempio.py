import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generazione di dati di esempio
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Divisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione del modello XGBoost
model = xgb.XGBClassifier()

# Addestramento del modello
model.fit(X_train, y_train)

# Predizione sui dati di test
y_pred = model.predict(X_test)

# Valutazione delle prestazioni del modello
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)