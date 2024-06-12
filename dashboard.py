import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

# URL de l'API déployée
API_URL = 'https://pret-a-depenser.azurewebsites.net'


def get_prediction(client_id):
    """Récupère la probabilité de défaut du client via l'API."""
    data = {"client_id": client_id}
    best_threshold = 0.54

    try:
        response = requests.post(f"{API_URL}/prediction", json=data)
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP

        # Extraction de la prédiction depuis la réponse JSON
        prediction = response.json().get("prediction")
        if prediction is None:
            return None 

        # Conversion de la prédiction en float
        proba_default = round(float(prediction), 3)

        # Détermination de la décision basée sur le seuil
        if proba_default >= best_threshold:
            decision = "Refusé"
        else:
            decision = "Accordé"

        return proba_default, decision

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
        return None
    except (ValueError, TypeError) as e:
        print(f"Erreur lors du traitement de la réponse : {e}")
        return None


def jauge_score(proba):
    """Construit une jauge indiquant le score du client.
    :param: proba (float).
    """
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 54], 'color': "Orange"},
                   {'range': [54, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}}))

    st.plotly_chart(fig)


def plot_client_features(client_id):
    """Récupère les caractéristiques du client via l'API et les visualise."""
    try:
        response = requests.get(f"{API_URL}/client_features/{client_id}")
        response.raise_for_status()  # Lève une exception pour les erreurs HTTP

        # Extraction des données de caractéristiques depuis la réponse JSON
        client_features = response.json()
        if not client_features:
            st.write("Aucune donnée de caractéristiques disponible pour ce client.")
            return

        # Création d'un dataframe à partir des données de caractéristiques
        df_client_features = pd.DataFrame(client_features)

        # Visualisation avec plotly
        st.header("Caractéristiques du Client")
        fig = px.bar(df_client_features, x=df_client_features.columns, y=df_client_features.iloc[0], 
                     labels={'x': 'Caractéristique', 'y': 'Valeur'}, 
                     title=f"Caractéristiques du Client {client_id}")
        st.plotly_chart(fig)

    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
    except (ValueError, TypeError) as e:
        print(f"Erreur lors du traitement de la réponse : {e}")


def main():
    st.title('Dashboard pour l\'analyse des prêts')

    # Section pour afficher le score de crédit
    st.header('Score de Crédit')
    client_id = st.number_input('Entrez l\'ID du client:', min_value=1)
    proba_default, decision = get_prediction(client_id)
    if proba_default is not None:
        st.write(f"Probabilité de défaut du client: {proba_default}")
        st.write(f"Décision: {decision}")
        jauge_score(proba_default)

    # Section pour afficher les caractéristiques du client
    st.header('Caractéristiques du Client')
    plot_client_features(client_id)


if __name__ == '__main__':
    main()
