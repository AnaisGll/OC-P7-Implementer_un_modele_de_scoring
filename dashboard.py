import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

# URL de l'API déployée
API_URL = 'https://pret-a-depenser.azurewebsites.net'

def get_prediction(client_id):
    data = {"client_id": client_id}
    try:
        response = requests.post(f"{API_URL}/prediction", json=data)
        response.raise_for_status()
        return response.json().get("prediction")
    except requests.exceptions.HTTPError:
        return None

def get_client_data(client_id):
    try:
        response = requests.get(f"{API_URL}/check_client/{client_id}")
        response.raise_for_status()
        if response.json():
            return response.json()
        else:
            return None
    except requests.exceptions.HTTPError:
        return None

def main():
    st.title("Dashboard de Prédiction de Remboursement de Prêt")
    
    st.sidebar.header("Menu")
    client_id = st.sidebar.number_input("Entrez l'ID du client", min_value=1, step=1)
    
    if st.sidebar.button("Obtenir les informations du client"):
        client_data = get_client_data(client_id)
        
        if client_data:
            st.write(f"**Informations du Client {client_id}**")
            st.json(client_data)
            
            st.write(f"**Probabilité de défaut de prêt**")
            prediction = get_prediction(client_id)
            st.write(f"La probabilité de défaut de prêt pour le client {client_id} est de {prediction:.2f}")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction * 100,
                title={'text': "Probabilité de défaut (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgreen"},
                           {'range': [50, 100], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
            st.plotly_chart(fig)
            
            st.write(f"**Comparaison avec d'autres clients**")
            feature = st.selectbox("Choisissez une variable pour comparer", options=client_data['features'])
            
            fig = px.histogram(client_data['data'], x=feature, title=f"Distribution de {feature} parmi tous les clients")
            st.plotly_chart(fig)
            
            fig = px.histogram(client_data['data'][client_data['data']['client_id'] == client_id], x=feature, title=f"Valeur de {feature} pour le client {client_id}")
            st.plotly_chart(fig)
        else:
            st.error("Client non trouvé")

if __name__ == '__main__':
    main()
