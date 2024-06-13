import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import matplotlib.pyplot as plt

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

def get_shap_values(client_id):
    data = {"client_id": client_id}
    try:
        response = requests.post(f"{API_URL}/shap_values", json=data)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention des valeurs SHAP: {e}")
        return None

    return response.json()
    
def main():
    st.title("Prédiction de remboursement de prêt")

    # Saisie du client_id par l'utilisateur
    client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

    # Bouton pour obtenir la prédiction
    if st.button('Obtenir la prédiction de défaut de prêt'):
        prediction = get_prediction(client_id)
        shap_data = get_shap_values(client_id)
        
        if prediction is not None and shap_data is not None:
            st.write(f"La probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")

            # Jauge de la probabilité de défaut
            st.progress(prediction)

            # Afficher les valeurs SHAP
            shap_values = np.array(shap_data['shap_values'])
            base_value = shap_data['base_value']
            feature_names = shap_data['feature_names']
            
            st.subheader("Importance des caractéristiques")
            shap.summary_plot(shap_values, feature_names, plot_type="bar", show=False)
            st.pyplot(bbox_inches='tight')

            st.subheader("Explications SHAP pour le client")
            shap.force_plot(base_value, shap_values, feature_names, matplotlib=True)
            st.pyplot(bbox_inches='tight')
        else:
            st.write("Aucune prédiction n'a pu être obtenue.")

if __name__ == '__main__':
    main()
