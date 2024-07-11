#Pruebaaaa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Especificar la ruta absoluta al archivo CSV
file_path = 'C:/Users/Tu Papi/Desktop/Proyect Algoritmica/Combined_Dataset/datos_limpios_1.csv'

# Verifica si el archivo existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {file_path} no se encuentra.")
else:
    # Cargar el archivo CSV en un DataFrame
    df = pd.read_csv(file_path)

    # Separar los datos en características y etiquetas
    # Para predicción de goles
    X_goles = df[['team', 'opponent']]
    y_goles = df['team_score']

    # Para predicción de victorias
    X_victoria = df[['team', 'opponent', 'team_score', 'opponent_score']]
    y_victoria = df['result']

    # Codificar las variables categóricas
    X_goles = pd.get_dummies(X_goles, columns=['team', 'opponent'])
    X_victoria = pd.get_dummies(X_victoria, columns=['team', 'opponent'])

    # Normalizar los datos
    scaler_goles = StandardScaler()
    X_goles = scaler_goles.fit_transform(X_goles)

    scaler_victoria = StandardScaler()
    X_victoria = scaler_victoria.fit_transform(X_victoria)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train_goles, X_test_goles, y_train_goles, y_test_goles = train_test_split(X_goles, y_goles, test_size=0.2, random_state=42)
    X_train_victoria, X_test_victoria, y_train_victoria, y_test_victoria = train_test_split(X_victoria, y_victoria, test_size=0.2, random_state=42)

    # Modelo de Random Forest para predicción de goles
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_goles, y_train_goles)

    # Modelo de Random Forest para predicción de victorias
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_victoria, y_train_victoria)

    # Función para predecir el resultado de un partido entre dos equipos
    def predict_match_result(team1, team2):
        # Predecir goles
        match_data_goles = pd.DataFrame({'team': [team1], 'opponent': [team2]})
        match_data_goles = pd.get_dummies(match_data_goles, columns=['team', 'opponent'])
        match_data_goles = match_data_goles.reindex(columns=pd.get_dummies(df[['team', 'opponent']]).columns, fill_value=0)
        match_data_goles = scaler_goles.transform(match_data_goles)
        predicted_goles_team1 = reg_model.predict(match_data_goles)[0]

        match_data_goles = pd.DataFrame({'team': [team2], 'opponent': [team1]})
        match_data_goles = pd.get_dummies(match_data_goles, columns=['team', 'opponent'])
        match_data_goles = match_data_goles.reindex(columns=pd.get_dummies(df[['team', 'opponent']]).columns, fill_value=0)
        match_data_goles = scaler_goles.transform(match_data_goles)
        predicted_goles_team2 = reg_model.predict(match_data_goles)[0]

        # Predecir resultado del partido
        match_data_victoria = pd.DataFrame({'team': [team1], 'opponent': [team2], 'team_score': [predicted_goles_team1], 'opponent_score': [predicted_goles_team2]})
        match_data_victoria = pd.get_dummies(match_data_victoria, columns=['team', 'opponent'])
        match_data_victoria = match_data_victoria.reindex(columns=pd.get_dummies(df[['team', 'opponent', 'team_score', 'opponent_score']]).columns, fill_value=0)
        match_data_victoria = scaler_victoria.transform(match_data_victoria)
        predicted_result_team1 = clf_model.predict_proba(match_data_victoria)[0]

        match_data_victoria = pd.DataFrame({'team': [team2], 'opponent': [team1], 'team_score': [predicted_goles_team2], 'opponent_score': [predicted_goles_team1]})
        match_data_victoria = pd.get_dummies(match_data_victoria, columns=['team', 'opponent'])
        match_data_victoria = match_data_victoria.reindex(columns=pd.get_dummies(df[['team', 'opponent', 'team_score', 'opponent_score']]).columns, fill_value=0)
        match_data_victoria = scaler_victoria.transform(match_data_victoria)
        predicted_result_team2 = clf_model.predict_proba(match_data_victoria)[0]

        return {
            'team1': team1,
            'team2': team2,
            'predicted_goles_team1': predicted_goles_team1,
            'predicted_goles_team2': predicted_goles_team2,
            'predicted_win_prob_team1': predicted_result_team1,
            'predicted_win_prob_team2': predicted_result_team2
        }

    # Predecir resu ltado de un partido entre dos equipos específicos
    team1 = 'Barcelona'
    team2 = 'RealMadrid'
    prediction = predict_match_result(team1, team2)

    # Mostrar la predicción
    prediction_result = {
        "Predicción de goles": {
            team1: prediction['predicted_goles_team1'],
            team2: prediction['predicted_goles_team2']
        },
        "Probabilidad de victoria": {
            team1: prediction['predicted_win_prob_team1'][1]*100,
            team2: prediction['predicted_win_prob_team2'][1]*100
        }
    }
    print(prediction_result)
