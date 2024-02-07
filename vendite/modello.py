import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carica i dati dal file CSV
data = pd.read_csv('sales_data.csv')

# Preprocessa i dati, ad esempio, convertendo variabili categoriche in dummy
data = pd.get_dummies(data)

# Dividi i dati in features (X) e target (y)
X = data.drop('Vendite', axis=1)
y = data['Vendite']

# Dividi i dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisci il modello XGBoost per regressione
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Addestramento del modello
model.fit(X_train, y_train)

# Predizione sui dati di test
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Root Mean Squared Error (RMSE):", rmse)