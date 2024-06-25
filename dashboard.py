import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import IPython
from PIL import Image
import io
import base64

# Local API URL
#API_URL = 'http://127.0.0.1:8000'
API_URL = 'https://pret-a-depenser.azurewebsites.net'

def get_prediction(client_id):
    data = {"client_id": client_id}
    try:
        response = requests.post(f"{API_URL}/prediction", json=data)
        response.raise_for_status()
        return response.json().get("prediction")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la prédiction: {e}")
        return None

def get_client_info(client_id):
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de l'obtention des informations du client: {response.text}")
        return None

def update_client_info(client_id, data):
    response = requests.put(f"{API_URL}/client_info/{client_id}", json=data)
    if response.status_code == 200:
        st.success("Informations du client mises à jour")
        return True
    else:
        st.error(f"Erreur lors de la mise à jour des informations du client: {response.text}")
        return False
        
def get_shap_summary_plot(client_id):
    try:
        response = requests.get(f"{API_URL}/shap_summary_plot/{client_id}")
        response.raise_for_status()
        shap_plot_data = response.json().get("shap_summary_plot")
        if shap_plot_data:
            return base64.b64decode(shap_plot_data)
        else:
            st.error("Erreur lors de la génération du SHAP summary plot.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention du SHAP summary plot: {e}")
        return None

def get_global_feature_importance():
    try:
        response = requests.get(f"{API_URL}/global_feature_importance")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la feature importance globale: {e}")
        return None

def get_local_feature_importance(client_id):
    try:
        response = requests.get(f"{API_URL}/local_feature_importance/{client_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la feature importance locale: {e}")
        return None
        
def main():
    st.title("Prédiction de remboursement de prêt")

    client_id = st.number_input("Entrez l'ID du client", min_value=1, step=1)

    if st.button('Obtenir la prédiction de défaut de prêt'):
        prediction = get_prediction(client_id)
        if prediction is not None:
            st.write(f"La probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")
            st.progress(prediction)
            if prediction < 0.5:
                st.success("Le prêt est approuvé.")
            else:
                st.error("Le prêt est refusé.")
                
            # Obtenir et afficher l'importance des caractéristiques locales
            local_feature_importance = get_local_feature_importance(client_id)
            if local_feature_importance:
                st.subheader("Importance des caractéristiques locales")
                local_importance_df = pd.DataFrame(list(local_feature_importance.items()), columns=['feature', 'importance'])
                fig_local = px.bar(local_importance_df.head(10), x='feature', y='importance', title="Feature Importance Locale (Top 10)")
                st.plotly_chart(fig_local)

            # Obtenir et afficher l'importance des caractéristiques globales
            global_feature_importance = get_global_feature_importance()
            if global_feature_importance:
                st.subheader("Feature Importance Globale")
                global_importance_df = pd.DataFrame(list(global_feature_importance.items()), columns=['feature', 'importance'])
                fig_global = px.bar(global_importance_df, x='feature', y='importance', title="Feature Importance Globale")
                st.plotly_chart(fig_global)
                
    if client_id:
        client_info = get_client_info(client_id)
        if client_info:
            st.subheader("Informations du client")
            with st.expander("Voir les informations du client"):
                st.json(client_info)

            st.subheader("Modifier les informations du client")
            update_data = {k: st.text_input(k, str(v)) for k, v in client_info.items() if k != 'client_id'}

            st.subheader("Comparaison des informations du client avec les autres clients")
            feature = st.selectbox("Sélectionnez une variable pour la comparaison", options=client_info.keys())
            if feature:
                train_data = pd.read_csv('train_mean_sample.csv')
                fig = px.histogram(train_data, x=feature, title=f"Distribution de {feature}")
                fig.add_vline(x=float(client_info[feature]), line_dash="dash", line_color="red", annotation_text="Client")
                st.plotly_chart(fig)

            feature1 = st.selectbox("Sélectionnez la première variable pour l'analyse bi-variée", options=client_info.keys())
            feature2 = st.selectbox("Sélectionnez la deuxième variable pour l'analyse bi-variée", options=client_info.keys())

            if feature1 and feature2:
                st.subheader(f"Analyse bi-variée entre {feature1} et {feature2}")
                fig_bivariate = px.scatter(train_data, x=feature1, y=feature2, title=f"Analyse bi-variée entre {feature1} et {feature2}")
                st.plotly_chart(fig_bivariate)

            if st.button('Mettre à jour les informations'):
                if update_client_info(client_id, update_data):
                    prediction = get_prediction(client_id)
                    if prediction is not None:
                        st.write(f"Nouvelle probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")
                        st.progress(prediction)
                        if prediction < 0.5:
                            st.success("Le prêt est approuvé.")
                        else:
                            st.error("Le prêt est refusé.")

if __name__ == '__main__':
    main()
