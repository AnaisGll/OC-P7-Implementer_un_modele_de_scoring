import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Local API URL
#API_URL = 'http://127.0.0.1:8000'
API_URL = 'https://pret-a-depenser.azurewebsites.net'

# Charger les données d'entraînement pour le SHAP explainer
train_data = pd.read_csv('train_mean_sample.csv')
X_train = train_data.drop(['TARGET', 'client_id'], axis=1)

# Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entraîner un modèle pour le SHAP explainer (exemple)
model = LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=31, force_col_wise=True)
model.fit(X_train_scaled, train_data['TARGET'])

# Créer un explainer SHAP basé sur le modèle entraîné
explainer = shap.Explainer(model, X_train_scaled)

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
        
def get_shap_values(client_id):
    response = requests.get(f"{API_URL}/shap_values/{client_id}")
    if response.status_code == 200:
        return response.json().get("shap_values")
    else:
        st.error(f"Erreur lors de l'obtention des valeurs SHAP: {response.text}")
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
                            
            # Affichage des SHAP values
            st.subheader("SHAP Values")
            shap_values = get_shap_values(client_id)
            if shap_values:
                st.write("Valeurs SHAP pour ce client :")
                st.json(shap_values)
                
                # Affichage du SHAP summary plot
                st.subheader("SHAP Summary Plot")
                shap_values_global = explainer(X_train_scaled)  # Utiliser les données déjà normalisées
                shap.summary_plot(shap_values_global.values, X_train, show=False)
                st.pyplot(plt.gcf())

if __name__ == '__main__':
    main()

