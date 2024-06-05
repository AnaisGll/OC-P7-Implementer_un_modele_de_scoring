import streamlit as st
import pandas as pd
import requests

# Chargement du fichier CSV contenant les données
df = pd.read_csv('test_mean_sample.csv')

# Liste des noms de vos variables
variable_names = [
    'SK_ID_CURR', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'PAYMENT_RATE', 'DAYS_BIRTH', 'EXT_SOURCE_1', 'DAYS_EMPLOYED', 
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'INSTAL_DBD_MEAN', 'ANNUITY_INCOME_PERC', 'AMT_ANNUITY', 
    'DAYS_LAST_PHONE_CHANGE', 'ACTIVE_DAYS_CREDIT_UPDATE_MEAN', 'REGION_POPULATION_RELATIVE', 
    'INSTAL_DAYS_ENTRY_PAYMENT_MAX', 'INCOME_CREDIT_PERC', 'INSTAL_AMT_PAYMENT_MIN', 'CLOSED_DAYS_CREDIT_MAX', 
    'ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'PREV_APP_CREDIT_PERC_VAR', 'INSTAL_DBD_MAX', 'INSTAL_DBD_SUM', 
    'BURO_DAYS_CREDIT_VAR', 'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'ACTIVE_DAYS_CREDIT_MAX', 'INCOME_PER_PERSON', 
    'PREV_HOUR_APPR_PROCESS_START_MEAN', 'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN', 'CLOSED_AMT_CREDIT_SUM_MEAN', 
    'POS_NAME_CONTRACT_STATUS_Active_MEAN', 'ACTIVE_DAYS_CREDIT_VAR', 'CLOSED_DAYS_CREDIT_UPDATE_MEAN', 
    'PREV_CNT_PAYMENT_MEAN', 'PREV_APP_CREDIT_PERC_MEAN', 'INSTAL_AMT_PAYMENT_MEAN', 'INSTAL_DPD_MEAN', 
    'PREV_AMT_ANNUITY_MIN', 'PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN', 'INSTAL_AMT_INSTALMENT_MAX', 
    'HOUR_APPR_PROCESS_START', 'TOTALAREA_MODE', 'INSTAL_PAYMENT_PERC_VAR', 'PREV_NAME_TYPE_SUITE_nan_MEAN', 
    'PREV_NAME_YIELD_GROUP_middle_MEAN', 'PREV_RATE_DOWN_PAYMENT_MEAN'
]

def extract_data(sk_id_curr):
    # Extraction des données correspondant à SK_ID_CURR
    data = df[df['SK_ID_CURR'] == sk_id_curr].drop(columns=['SK_ID_CURR'])
    return data

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {"dataframe_records": data}
    response = requests.post(model_uri, headers=headers, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()["predictions"]

def main():
    #local
    #API_URL = 'http://127.0.0.1:5000/invocations'
    API_URL = 'https://pret-a-depenser.azurewebsites.net'

    st.title("Prédiction de remboursement de prêt")

    # Saisie du SK_ID_CURR par l'utilisateur
    sk_id_curr = st.number_input('Entrez le SK_ID_CURR :', format='%d')

    predict_btn = st.button('Prédire')
    if predict_btn:
        # Extraction des données à partir du fichier CSV
        data = extract_data(sk_id_curr)
        
        # Préparation des données pour la prédiction
        data_for_prediction = data[variable_names].values.tolist()
        
        # Appel à l'API de prédiction
        pred = request_prediction(API_URL, data_for_prediction)
        
        # Affichage de la prédiction
        st.write(f"La prédiction est : {'Oui' if pred[0] == 1 else 'Non'}")

if __name__ == '__main__':
    main()
