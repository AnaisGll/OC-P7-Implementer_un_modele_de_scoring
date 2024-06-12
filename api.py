from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# Charger les données et la pipeline
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')
pipeline = joblib.load('pipeline_LGBM_prediction.joblib')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Initialiser l'explainer SHAP
explainer = shap.Explainer(pipeline)

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
    Calcule la probabilité de défaut pour un client et les valeurs SHAP.
    :return: probabilité de défaut (float) et valeurs SHAP (dict).
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

    # Calculer les valeurs SHAP
    shap_values = explainer(info_client)
    shap_values_dict = dict(zip(info_client.columns, shap_values.values[0]))

    return jsonify({"prediction": prediction, "shap_values": shap_values_dict})

@app.route('/shaplocal/<int:client_id>', methods=['GET'])
def shap_values_local(client_id):
    """
    Calcule les SHAP values pour un client.
    :param: client_id (int)
    :return: SHAP values du client (JSON).
    """
    client_data = test_data[test_data['client_id'] == client_id]
    client_data = client_data.drop('client_id', axis=1)
    shap_val = explainer.shap_values(client_data)[0][:, 1]

    return jsonify({
        'shap_values': shap_val.tolist(),
        'base_value': shap_val.base_values,
        'data': client_data.values.tolist(),
        'feature_names': client_data.columns.tolist()
    })

@app.route('/shap/', methods=['GET'])
def shap_values():
    """
    Calcule les SHAP values de l'ensemble du jeu de données.
    :return: SHAP values (JSON).
    """
    shap_val = explainer.shap_values(test_data.drop('client_id', axis=1))
    return jsonify({
        'shap_values_0': shap_val[0].tolist(),
        'shap_values_1': shap_val[1].tolist()
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
