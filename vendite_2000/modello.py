import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Caricare i dati dal file CSV
data = pd.read_csv('vendite_cioccolato.csv')

# 2. Preprocessare i dati se necessario (non è necessario in questo caso)

# 3. Dividere i dati in features (X) e target (y)
X = data.drop('Vendite', axis=1)
y = data['Vendite']

# 4. Dividere i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Definire il modello XGBoost per la regressione
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 6. Addestrare il modello
model.fit(X_train, y_train)

# 7. Fare previsioni sui dati di test
y_pred = model.predict(X_test)

# Valutare le prestazioni del modello
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)