import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import shap
import matplotlib.pyplot as plt

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
        
def get_shap_values_local(client_id):
    response = requests.get(f"{API_URL}/shap_values/{client_id}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de l'obtention des valeurs SHAP: {response.text}")
        return None

def st_shap(plot, height=None):
    """ Helper function to display a SHAP plot in Streamlit """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)
    
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
            shap_values_local = get_shap_values_local(client_id)
            if shap_values_local:
                with st.expander("Voir les valeurs SHAP pour ce client"):
                    st.json(shap_values_local)
                    shap_values = shap_values_local['shap_values']
                    base_value = shap_values_local['base_value']
                    data = shap_values_local['data'][0]  # Prendre la première ligne des données du client
                    feature_names = shap_values_local['feature_names']

                    # Create a SHAP force plot
                    shap.initjs()
                    shap_values_obj = shap.Explanation(values=shap_values, base_values=base_value, data=data, feature_names=feature_names)
                    st_shap(shap.force_plot(base_value, shap_values, data, feature_names=feature_names))

                # Create a summary plot with the top 10 selected features
                st.write("Summary Plot des 10 variables les plus importantes")
                shap_values_df = pd.DataFrame({
                    'shap_values': shap_values,
                    'feature_names': feature_names
                })
                top_features = shap_values_df.nlargest(10, 'shap_values')['feature_names']
                
                # Using matplotlib to create the summary plot
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values_obj[:, top_features], plot_type="bar", show=False)
                st.pyplot(fig)
    
if __name__ == '__main__':
    main()
