#original-final para presentar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

# Especificar la ruta absoluta al archivo CSV
file_path = 'C:/Users/Tu Papi/Desktop/Proyect Algoritmica/Combined_Dataset/Combined_Dataset.csv'

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(file_path)

# Limpiar el DataFrame de valores nulos
df.dropna(inplace=True)

# Asignar rondas si no existe una columna 'round'
if 'round' not in df.columns:
    df['round'] = (df.index // 10) + 1

# Verificar las rondas disponibles en el dataset
rounds_available = df['round'].unique()
num_rounds = len(rounds_available)

print(f"Número de rondas disponibles: {num_rounds}")
print(f"Rondas disponibles: {rounds_available}")

# Función para entrenar y predecir el resultado de un partido entre dos equipos
def predict_match_result(team1, team2, round_num):
    # Filtrar los datos para la ronda específica
    df_round = df[df['round'] == round_num]

    if df_round.empty:
        return f"No hay datos disponibles para la ronda {round_num}."

    # Separar los datos en características y etiquetas
    X = df_round[['team', 'opponent', 'round']]
    y = df_round['team_score']

    # Codificar las variables categóricas
    X = pd.get_dummies(X, columns=['team', 'opponent'])

    # Normalizar los datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Random Forest para predicción de goles
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train)

    # Predecir goles
    match_data_goles = pd.DataFrame({'team': [team1], 'opponent': [team2], 'round': [round_num]})
    match_data_goles = pd.get_dummies(match_data_goles, columns=['team', 'opponent'])
    match_data_goles = match_data_goles.reindex(columns=pd.get_dummies(df[['team', 'opponent', 'round']]).columns, fill_value=0)
    match_data_goles = scaler.transform(match_data_goles)
    predicted_goles_team1 = reg_model.predict(match_data_goles)[0]

    match_data_goles = pd.DataFrame({'team': [team2], 'opponent': [team1], 'round': [round_num]})
    match_data_goles = pd.get_dummies(match_data_goles, columns=['team', 'opponent'])
    match_data_goles = match_data_goles.reindex(columns=pd.get_dummies(df[['team', 'opponent', 'round']]).columns, fill_value=0)
    match_data_goles = scaler.transform(match_data_goles)
    predicted_goles_team2 = reg_model.predict(match_data_goles)[0]

    # Derivar probabilidad de victoria
    if predicted_goles_team1 > predicted_goles_team2:
        prob_team1 = 1.0
        prob_team2 = 0.0
    elif predicted_goles_team1 < predicted_goles_team2:
        prob_team1 = 0.0
        prob_team2 = 1.0
    else:
        prob_team1 = 0.5
        prob_team2 = 0.5

    return {
        'round': round_num,
        'team1': team1,
        'team2': team2,
        'predicted_goles_team1': predicted_goles_team1,
        'predicted_goles_team2': predicted_goles_team2,
        'predicted_win_prob_team1': prob_team1,
        'predicted_win_prob_team2': prob_team2
    }

# Predecir resultado de un partido entre dos equipos específicos para las rondas 0 a 8
team1 = 'Barcelona'
team2 = 'RealMadrid'
all_predictions = []
for round_num in range(0, 9):  # Rondas 0 a 8
    prediction = predict_match_result(team1, team2, round_num)
    all_predictions.append(prediction)

# Función para filtrar y mostrar resultados desde el diccionario
def mostrar_resultados(predictions):
    for prediction in predictions:
        if isinstance(prediction, str):
            print(prediction)
            continue
        print(f"Ronda: {prediction['round']}")
        print("Predicción de Goles:")
        print(f"{prediction['team1']}: {prediction['predicted_goles_team1']:.2f} goles")
        print(f"{prediction['team2']}: {prediction['predicted_goles_team2']:.2f} goles")
        print("Probabilidad de Victoria:")
        prob_team1 = prediction['predicted_win_prob_team1']
        prob_team2 = prediction['predicted_win_prob_team2']
        print(f"{prediction['team1']}: {prob_team1*100:.2f}%")
        print(f"{prediction['team2']}: {prob_team2*100:.2f}%")
        if prob_team1 > prob_team2:
            print(f"Equipo con más probabilidad de ganar: {prediction['team1']}")
        else:
            print(f"Equipo con más probabilidad de ganar: {prediction['team2']}")
        print("\n")

# Mostrar la predicción
mostrar_resultados(all_predictions)
