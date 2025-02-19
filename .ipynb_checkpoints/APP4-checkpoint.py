import requests
import pandas as pd
import sqlite3
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow import keras

st.title("📈 Prédiction des Prix des Actions")

# Entrée pour le symbole boursier et l'horizon de prévision
ticker = st.text_input("Entrez le symbole boursier (ex: AAPL, TSLA)", "AAPL")
horizon = st.slider("Choisissez l'horizon de prévision (jours)", 1, 30, 7)

@st.cache
def scrape_historique(ticker):
    # Calculer la date de début (il y a 3 ans)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    # Convertir les dates en timestamp Unix
    period1 = int(start_date.timestamp())
    period2 = int(end_date.timestamp())

    # Construire l'URL avec les paramètres de date
    url = f"https://finance.yahoo.com/quote/{ticker}/history?period1={period1}&period2={period2}&interval=1d&filter=history&frequency=1d"

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")

        if tables:
            df = pd.read_html(str(tables[0]))[0]
            df = df.dropna()

            # Convertir en format Date
            df["Date"] = pd.to_datetime(df["Date"])
            return df
    return None

# Fonction pour afficher un graphique de la donnée historique
def plot_historical_data(df):
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Close"], label="Prix de clôture", color='b', linestyle='-')
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture (USD)")
    plt.title(f"Évolution du prix de l'action {ticker}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot()

# Fonction pour effectuer la prédiction et l'affichage des résultats
def predict_and_plot(df):
    # Normalisation des prix
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Close"] = scaler.fit_transform(df[["Close"]])

    # Créer les séquences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    seq_length = 50
    data = df["Close"].values
    X, y = create_sequences(data, seq_length)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Modèle LSTM
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units=64))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Prédictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Visualisation des résultats
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), scaler.inverse_transform(data.reshape(-1, 1)), label='Données Historiques', color='blue')
    plt.plot(range(len(data) - len(predictions), len(data)), predictions, label='Prévisions', color='red', linestyle='--')
    plt.title(f'Prévisions de Prix de l\'Action {ticker}')
    plt.xlabel('Index')
    plt.ylabel('Prix de Clôture (USD)')
    plt.legend()
    plt.grid()
    st.pyplot()

# Scraper les données boursières
st.write("Récupération des données historiques...")

df = scrape_historique(ticker)

if df is not None:
    st.write("Premières données après traitement :")
    st.dataframe(df.head())

    # Afficher un graphique de l'évolution des prix
    plot_historical_data(df)

    # Bouton pour lancer la prédiction
    if st.button("Lancer la Prédiction"):
        with st.spinner("Prédiction en cours..."):
            predict_and_plot(df)

else:
    st.write(f"Impossible de récupérer les données pour {ticker}. Assurez-vous que le symbole est correct.")
