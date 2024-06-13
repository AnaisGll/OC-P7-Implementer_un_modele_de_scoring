import streamlit as st
import pandas as pd
import requests

#local
#API_URL = 'http://127.0.0.1:8000'
API_URL = 'https://pret-a-depenser.azurewebsites.net'

def get_prediction(client_id):
    data = {"client_id": client_id}
    try:
        response = requests.post(f"{API_URL}/prediction", json=data)
        response.raise_for_status()  # Lève une exception pour les statuts d'erreur HTTP
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la prédiction: {e}")
        return None

    return response.json().get("prediction")

def main():
    st.title("Prédiction de remboursement de prêt")

    # Saisie du client_id par l'utilisateur
    client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

    # Bouton pour obtenir la prédiction
    if st.button('Obtenir la prédiction de défaut de prêt'):
        prediction = get_prediction(client_id)
        if prediction is not None:
            st.write(f"La probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")
        else:
            st.write("Aucune prédiction n'a pu être obtenue.")

if __name__ == '__main__':
    main()
