import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Local API URL
API_URL = 'http://127.0.0.1:8000'

def get_prediction(client_id):
    data = {"client_id": client_id}
    response = requests.post(f"{API_URL}/prediction", json=data)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        st.error(f"Erreur lors de l'obtention de la prédiction: {response.text}")
        return None

def get_client_info(client_id):
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de l'obtention des informations du client: {response.text}")
        return None

def main():
    st.title("Prédiction de remboursement de prêt")

    # Saisie du client_id par l'utilisateur
    client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

    # Bouton pour obtenir la prédiction
    if st.button('Obtenir la prédiction de défaut de prêt'):
        prediction = get_prediction(client_id)
        if prediction is not None:
            st.write(f"La probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")
            if prediction < 0.5:
                st.success("Le prêt est probablement approuvé.")
            else:
                st.error("Le prêt est probablement refusé.")

    # Afficher les informations descriptives du client
    if client_id:
        client_info = get_client_info(client_id)
        if client_info:
            st.subheader("Informations du client")
            st.write(client_info)

            # Graphiques comparatifs
            st.subheader("Comparaison des informations du client avec les autres clients")
            feature = st.selectbox("Sélectionnez une variable pour la comparaison", options=client_info.keys())
            if feature:
                train_data = pd.read_csv('train_mean_imputed.csv')  # Charger les données d'entraînement pour comparaison
                fig = px.histogram(train_data, x=feature, title=f"Distribution de {feature}")
                fig.add_vline(x=client_info[feature], line_dash="dash", line_color="red")
                st.plotly_chart(fig)

if __name__ == '__main__':
    main()
