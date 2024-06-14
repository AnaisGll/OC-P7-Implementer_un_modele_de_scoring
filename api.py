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
    if client_id in list(test_data['client_id']):
        return jsonify(True)
    else:
        return jsonify(False)
        
@app.route('/client_info/<int:client_id>', methods=['GET'])
def get_client_info(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    return client_data.to_dict(orient='records')[0]

@app.route('/client_info/<int:client_id>', methods=['PUT'])
def update_client_info(client_id):
    data = request.get_json()
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    global test_data
    test_data.loc[test_data['client_id'] == client_id, list(data.keys())] = list(data.values())
    return jsonify({"message": "Client information updated"}), 200

@app.route('/client_info', methods=['POST'])
def submit_new_client():
    data = request.get_json()
    new_client_id = max(test_data['client_id']) + 1
    data['client_id'] = new_client_id
    global test_data
    test_data = pd.concat([test_data, pd.DataFrame([data])], ignore_index=True)
    return jsonify({"message": "New client submitted", "client_id": new_client_id}), 201

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

