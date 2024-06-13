from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import joblib
import shap

app = Flask(__name__)

# Chargement des données et de la pipeline
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Convertir les données en float64
test_data = test_data.astype('float64')
train_data = train_data.astype('float64')

# Définir la pipeline avec LightGBM
def create_pipeline():
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=0)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    lgbm_model = LGBMClassifier()
    lgbm_model.fit(X_train_resampled, y_train_resampled)
    
    pipeline = {
        'scaler': scaler,
        'smote': smote,
        'model': lgbm_model
    }
    
    return pipeline

pipeline = create_pipeline()

# SHAP explainer
explainer = shap.TreeExplainer(pipeline['model'])

@app.route('/')
def home():
    return "API pour prédire l'accord d'un prêt"

@app.route('/check_client/<int:client_id>', methods=['GET'])
def check_client_id(client_id):
    if client_id in test_data['client_id'].values:
        return jsonify(True)
    else:
        return jsonify(False)

@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400
    
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    
    # Prétraitement des données comme dans la pipeline d'entraînement
    X_client = client_data.drop(columns=['client_id'])
    X_client_scaled = pipeline['scaler'].transform(X_client)
    X_client_resampled = pipeline['smote'].sample(X_client_scaled)
    
    # Prédiction avec le modèle
    prediction = pipeline['model'].predict_proba(X_client_resampled)[:, 1]
    
    return jsonify({"prediction": float(prediction)})

@app.route('/shap_values', methods=['POST'])
def get_shap_values():
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400
    
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    
    # Prétraitement des données comme dans la pipeline d'entraînement
    X_client = client_data.drop(columns=['client_id'])
    X_client_scaled = pipeline['scaler'].transform(X_client)
    X_client_resampled = pipeline['smote'].sample(X_client_scaled)
    
    # Calcul des valeurs SHAP
    shap_values = explainer.shap_values(X_client_resampled)[1]  # Assuming binary classification
    base_value = explainer.expected_value[1]
    feature_names = X_client.columns.tolist()
    
    return jsonify({
        "shap_values": shap_values.tolist(),
        "base_value": float(base_value),
        "feature_names": feature_names
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


