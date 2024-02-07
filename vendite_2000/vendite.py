import pandas as pd
import numpy as np

# Definizione delle variabili per generare i dati
num_samples = 2000
tipo_cioccolato = ['Cioccolato al latte', 'Cioccolato fondente', 'Cioccolato bianco']
prezzi = [2.5, 3.0, 2.8]
promozioni = ['Sì', 'No']

# Generazione dei dati casuali
np.random.seed(42)
vendite_data = {
    'Prezzo': np.random.choice(prezzi, num_samples),
    'Tipo': np.random.choice(tipo_cioccolato, num_samples),
    'Quantità': np.random.randint(50, 150, num_samples),
    'Promozione': np.random.choice(promozioni, num_samples),
    'Vendite': np.random.randint(100, 400, num_samples)
}

# Creazione del DataFrame pandas
df = pd.DataFrame(vendite_data)

# Salvataggio dei dati in un file CSV
df.to_csv('vendite_cioccolato.csv', index=False)