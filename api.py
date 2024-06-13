from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)

# Charger les données et la pipeline
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')
pipeline = joblib.load('pipeline_LGBM_prediction.joblib')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Convertir les données en float64
test_data = test_data.astype('float64')
train_data = train_data.astype('float64')

# Extraire le modèle LightGBM de la pipeline
def extract_model_from_pipeline(pipeline):
    # Assurez-vous que le modèle est nommé 'model' dans votre pipeline
    return pipeline.named_steps['model']

model = extract_model_from_pipeline(pipeline)

# SHAP explainer
explainer = shap.TreeExplainer(model)

@app.route('/')
def home():
    return 'API pour prédire l\'accord d\'un prêt'

@app.route('/check_client/<int:client_id>', methods=['GET'])
def check_client_id(client_id):
    """
    Recherche du client dans la base de données
    :param: client_id (int)
    :return: message (bool).
    """
    if client_id in list(test_data['client_id']):
        return jsonify(True)
    else:
        return jsonify(False)

@app.route('/prediction', methods=['POST'])
def get_prediction():
    """
    Calcule la probabilité de défaut pour un client.
    :return: probabilité de défaut (float).
    """
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    prediction = pipeline.predict_proba(info_client)[0][1]
    return jsonify({"prediction": prediction})

@app.route('/shap_values', methods=['POST'])
def get_shap_values():
    """
    Retourne les valeurs SHAP pour un client spécifique.
    :return: valeurs SHAP (list), base value (float), feature names (list).
    """
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    shap_values = explainer.shap_values(info_client)[1]  # Assuming binary classification
    base_value = explainer.expected_value[1]
    feature_names = info_client.columns.tolist()
    
    return jsonify({
        "shap_values": shap_values.tolist(),
        "base_value": base_value,
        "feature_names": feature_names
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)


