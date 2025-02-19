import streamlit as st
import requests
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Importer pour la mise en forme
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #4CAF50;">📈 Prédiction des Prix des Actions</h1>
        <p style="font-size: 18px;">Développé par <b>BONKOUNGOU Emmanuel, MANTHO Livine Larissa
et MISSENGUE MOULOMBO Exaucée </b></p>
        <p style="font-size: 18px;">Sous la supervision de <b>M. Serge NDOUMIN </b></p>
    </div>
""", unsafe_allow_html=True)




st.sidebar.title("🔍 Options")
st.sidebar.info("Bienvenue sur notre application de prédiction boursière. \nSélectionnez vos paramètres et analysez les tendances.")



# Fonction pour scraper les données de Yahoo Finance
def scrape_historique(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)

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

            # Convertir la colonne "Date" en datetime
            df["Date"] = pd.to_datetime(df["Date"])

            # 📌 **CORRECTION : Réordonner les dates du plus ancien au plus récent**
            df = df.sort_values(by="Date").reset_index(drop=True)

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
#st.title("📈 Prédiction des Prix des Actions")

ticker = st.text_input("Entrez le symbole boursier (ex: AAPL, TSLA)", "AAPL")
horizon = st.slider("Choisissez l'horizon de prévision (jours)", 1, 180, 120)

if st.button("Lancer l'analyse", key="scraper_button"):
    with st.spinner("Récupération des données..."):
        df = scrape_historique(ticker)
        
        if df is not None:
            df, scaler = prepare_data(df)
            st.success("✅ Données récupérées et traitées avec succès !")

            # 1️⃣ Affichage des premières lignes des données
            st.subheader("📊 Données après traitement")
            st.dataframe(df.head())
            st.markdown("---")  # Ajoute une ligne horizontale
            st.subheader("📊 Données après traitement")

            # 2️⃣ Visualisation des données historiques
            st.subheader("📈 Visualisation des données")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["Date"], df["Close"], label="Prix de clôture normalisé", color='b', linestyle='-')
            ax.set_xlabel("Date")
            ax.set_ylabel("Prix de clôture (normalisé)")
            ax.set_title(f"Évolution du prix de l'action {ticker}")
            ax.legend()
            ax.grid()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.markdown("---")  # Ajoute une ligne horizontale
            st.subheader("📊 Données après traitement")

            # 3️⃣ Prédictions
            st.subheader("🔮 Prédictions des prix")

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

            # 📌 **Correction de l'ordre des dates pour l'affichage**
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["Date"], scaler.inverse_transform(df[["Close"]]), label="Données Historiques", color="blue")

            # Générer les dates futures à partir de la dernière date connue
            future_dates = pd.date_range(df["Date"].iloc[-1], periods=horizon + 1, freq='D')[1:]

            # Ajouter les prévisions futures au graphique
            ax.plot(future_dates, future_predictions, label="Prévisions Futures", color="red", linestyle="--")

            # Personnalisation du graphique
            ax.set_xlabel("Date")
            ax.set_ylabel("Prix de Clôture (USD)")
            ax.legend()
            ax.grid()
            plt.xticks(rotation=45)

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
            st.success("✅ Prédictions générées avec succès !")
            
st.markdown("""
    <hr>
    <p style="text-align: center; font-size: 14px;">
        🚀 Projet réalisé dans le cadre du cours de Webscrapping | © 2025 - Tous droits réservés
    </p>
""", unsafe_allow_html=True)

