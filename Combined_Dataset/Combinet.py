import pandas as pd

# Ruta completa al archivo CSV
file_path = r'C:\Users\Tu Papi\Desktop\Proyect Algoritmica\Combined_Dataset\Combined_Dataset.csv'

# Cargar el DataFrame desde el archivo CSV
combined_df = pd.read_csv(file_path)

# Eliminar filas con valores nulos
combined_df = combined_df.dropna()

# Define una función para extraer los valores requeridos
def extract_values(row):
    home_team = row['team'].replace(' ', '')
    away_team = row['opponent'].replace(' ', '')
    home_score = row['team_score']
    away_score = row['opponent_score']
    result = row['result']

    # Crea dos filas para cada fila original
    row1 = {
        'match': f"{home_team}_{away_team}",
        'team': home_team,
        'opponent': away_team,
        'team_score': home_score,
        'opponent_score': away_score,
        'result': result
    }
    row2 = {
        'match': f"{away_team}_{home_team}",
        'team': away_team,
        'opponent': home_team,
        'team_score': away_score,
        'opponent_score': home_score,
        'result': 'win' if result == 'loss' else 'loss' if result == 'win' else 'draw'
    }

    return [row1, row2]

# Aplica la función a cada fila del DataFrame y crea un nuevo DataFrame
new_rows = [row for _, row in combined_df.iterrows() for row in extract_values(row)]
new_df = pd.DataFrame(new_rows)

# Guarda el nuevo DataFrame en un archivo CSV
new_df.to_csv(r'C:\Users\Tu Papi\Desktop\Proyect Algoritmica\Combined_Dataset\datos_limpios_1.csv', index=False)

# Mostrar todas las columnas y filas del nuevo DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(new_df)

# También puedes usar .head() para mostrar solo las primeras filas si el DataFrame es muy grande
#


print(new_df.head())
