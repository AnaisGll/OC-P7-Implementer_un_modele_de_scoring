import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from PIL import Image
import io
import base64

# Local API URL
# API_URL = 'http://127.0.0.1:8000'
API_URL = 'https://pret-a-depenser.azurewebsites.net'

# Fonction pour obtenir la prédiction
def get_prediction(client_id):
    data = {"client_id": client_id}
    try:
        response = requests.post(f"{API_URL}/prediction", json=data)
        response.raise_for_status()
        return response.json().get("prediction")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la prédiction: {e}")
        return None

# Fonction pour obtenir les informations du client
def get_client_info(client_id):
    response = requests.get(f"{API_URL}/client_info/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de l'obtention des informations du client: {response.text}")
        return None

# Fonction pour mettre à jour les informations du client
def update_client_info(client_id, data):
    response = requests.put(f"{API_URL}/client_info/{client_id}", json=data)
    if response.status_code == 200:
        st.success("Informations du client mises à jour")
        return True
    else:
        st.error(f"Erreur lors de la mise à jour des informations du client: {response.text}")
        return False

# Fonction pour obtenir le SHAP summary plot
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

# Fonction pour obtenir l'importance des caractéristiques globales
def get_global_feature_importance():
    try:
        response = requests.get(f"{API_URL}/global_feature_importance")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la feature importance globale: {e}")
        return None

# Fonction pour obtenir l'importance des caractéristiques locales
def get_local_feature_importance(client_id):
    try:
        response = requests.get(f"{API_URL}/local_feature_importance/{client_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'obtention de la feature importance locale: {e}")
        return None

def main():
    # Titre de la page
    st.set_page_config(page_title="Prédiction de remboursement de prêt")
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
                fig_local.update_layout(title_text='Feature Importance Locale (Top 10)', title_x=0.5)
                fig_local.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
                st.plotly_chart(fig_local, use_container_width=True)
                st.write("Le graphique ci-dessus montre l'importance des différentes caractéristiques locales. Chaque barre représente une caractéristique avec son niveau d'importance.")

            # Obtenir et afficher l'importance des caractéristiques globales
            global_feature_importance = get_global_feature_importance()
            if global_feature_importance:
                st.subheader("Feature Importance Globale")
                global_importance_df = pd.DataFrame(global_feature_importance.items(), columns=['feature', 'importance'])
                global_importance_df['importance'] = global_importance_df['importance'].astype(float)
                fig_global = px.bar(global_importance_df.head(10), x='feature', y='importance', title="Feature Importance Globale")
                fig_global.update_layout(title_text='Feature Importance Globale', title_x=0.5)
                fig_global.update_traces(marker_color='rgb(123,204,196)', marker_line_color='rgb(4,77,51)', marker_line_width=1.5, opacity=0.6)
                st.plotly_chart(fig_global, use_container_width=True)
                st.write("Le graphique ci-dessus montre l'importance des différentes caractéristiques globales. Chaque barre représente une caractéristique avec son niveau d'importance.")

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
                fig.update_layout(title_text=f"Distribution de {feature}", title_x=0.5)
                fig.update_traces(marker_color='rgb(246,207,113)', marker_line_color='rgb(205,102,0)', marker_line_width=1.5, opacity=0.6)
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Le graphique ci-dessus montre la distribution de la variable {feature}. La ligne rouge indique la position de ce client.")

            feature1 = st.selectbox("Sélectionnez la première variable pour l'analyse bi-variée", options=client_info.keys())
            feature2 = st.selectbox("Sélectionnez la deuxième variable pour l'analyse bi-variée", options=client_info.keys())

            if feature1 and feature2:
                st.subheader(f"Analyse bi-variée entre {feature1} et {feature2}")
                fig_bivariate = px.scatter(train_data, x=feature1, y=feature2, title=f"Analyse bi-variée entre {feature1} et {feature2}")
                fig_bivariate.update_layout(title_text=f"Analyse bi-variée entre {feature1} et {feature2}", title_x=0.5)
                fig_bivariate.update_traces(marker_color='rgb(229,152,102)', marker_line_color='rgb(174,49,0)', marker_line_width=1.5, opacity=0.6)
                st.plotly_chart(fig_bivariate, use_container_width=True)
                st.write(f"Le graphique ci-dessus montre la relation entre {feature1} et {feature2} pour différents clients. Chaque point représente un client.")

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

