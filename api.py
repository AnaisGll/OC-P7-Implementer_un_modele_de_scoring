from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Charger les données et la pipeline
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')
pipeline = joblib.load('pipeline_LGBM_prediction.joblib')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
