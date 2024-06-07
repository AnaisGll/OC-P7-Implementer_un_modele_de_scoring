import streamlit as st
import pandas as pd
import requests

#local
API_URL = 'http://127.0.0.1:8000'
#API_URL = 'https://pret-a-depenser.azurewebsites.net'

def get_prediction(client_id):
    data = {"client_id": client_id}
    response = requests.post(f"{API_URL}/prediction", json=data)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        st.error(f"Erreur lors de l'obtention de la prédiction: {response.text}")
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

if __name__ == '__main__':
    main()