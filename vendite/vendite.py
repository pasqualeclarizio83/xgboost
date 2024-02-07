import pandas as pd

# Creazione dei dati di esempio
data = {
    'Prezzo': [2.5, 3.0, 2.8, 3.2, 2.7],
    'Tipo': ['Cioccolato al latte', 'Cioccolato fondente', 'Cioccolato bianco', 'Cioccolato al latte', 'Cioccolato fondente'],
    'Quantità': [100, 120, 90, 110, 130],
    'Promozione': ['Sì', 'No', 'Sì', 'No', 'No'],
    'Vendite': [250, 300, 270, 320, 270]
}

# Creazione del DataFrame pandas
df = pd.DataFrame(data)

# Salvataggio dei dati in un file CSV
df.to_csv('sales_data.csv', index=False)