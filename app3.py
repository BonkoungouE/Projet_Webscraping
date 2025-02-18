import streamlit as st
import requests
import pandas as pd
import numpy as np
import sqlite3
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fonction pour scraper les données de Yahoo Finance
def scrape_historique(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
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

            # Connexion SQLite
            conn = sqlite3.connect("prix_historique.db")
            df.to_sql(ticker, conn, if_exists="replace", index=False)
            conn.close()
            
            return df
        else:
            st.error("Tableau introuvable sur Yahoo Finance.")
    else:
        st.error("Échec du scraping des données.")
    return None

# Fonction pour préparer les données
def prepare_data(df):
    df.rename(columns={df.columns[4]: "Close"}, inplace=True)
    df = df[["Date", "Close"]].dropna()

    # Filtrer les lignes contenant des valeurs non numériques dans la colonne "Close"
    df = df[pd.to_numeric(df["Close"], errors='coerce').notnull()]

    df["Close"] = df["Close"].astype(float)  # Convertir en float après nettoyage
    df.set_index("Date", inplace=True)

    # Normalisation des données
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["Close"] = scaler.fit_transform(df[["Close"]])

    return df, scaler

# Fonction pour créer des séquences temporelles
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Interface utilisateur Streamlit
st.title("📈 Prédiction des Prix des Actions")

ticker = st.text_input("Entrez le symbole boursier (ex: AAPL, TSLA)", "AAPL")
horizon = st.slider("Choisissez l'horizon de prévision (jours)", 1, 30, 7)

if st.button("Lancer la Prédiction"):
    with st.spinner("Récupération des données..."):
        df = scrape_historique(ticker)
    
    if df is not None:
        df, scaler = prepare_data(df)
        seq_length = 50
        data = df["Close"].values

        X, y = create_sequences(data, seq_length)
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        # Adapter les dimensions pour la régression linéaire
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        # Entraînement du modèle
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prédictions futures
        future_predictions = []
        last_sequence = data[-seq_length:].reshape(1, -1)

        for _ in range(horizon):
            next_pred = model.predict(last_sequence)[0]
            future_predictions.append(next_pred)
            last_sequence = np.append(last_sequence[:, 1:], next_pred).reshape(1, -1)

        # Inverser la normalisation
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Affichage des résultats
        st.subheader("📊 Visualisation des Prédictions")
        
        # Créer un graphique avec les données historiques et les prévisions futures
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, scaler.inverse_transform(df[["Close"]]), label="Données Historiques", color="blue")

        # Étendre l'index des dates pour inclure les prévisions futures
        future_dates = pd.date_range(df.index[-1], periods=horizon + 1, freq='D')[1:]

        # Ajouter les prévisions futures au graphique
        ax.plot(future_dates, future_predictions, label="Prévisions Futures", color="red", linestyle="--")

        # Personnalisation du graphique
        ax.set_xlabel("Date")
        ax.set_ylabel("Prix de Clôture (USD)")
        ax.legend()
        ax.grid()

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)

        st.success("Prédictions générées avec succès ! ✅")
