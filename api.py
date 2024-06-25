from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import plotly.express as px

app = Flask(__name__)

# Charger les données 
test_data = pd.read_csv('test_mean_sample.csv')
train_data = pd.read_csv('train_mean_sample.csv')

# Ajouter la colonne client_id
train_data['client_id'] = range(1, len(train_data) + 1)
test_data['client_id'] = range(1, len(test_data) + 1)

# Séparer les features et la target dans les données d'entraînement
X_train = train_data.drop(['TARGET', 'client_id'], axis=1)
y_train = train_data['TARGET']

# Appliquer le scaler et SMOTE aux données d'entraînement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Entraîner le modèle
model = LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=31, force_col_wise=True)
model.fit(X_train_resampled, y_train_resampled)

# Créer un explainer SHAP basé sur le modèle entraîné
explainer = shap.Explainer(model, X_train_scaled)

# Calcul de la feature importance globale
shap_values_train = explainer(X_train_scaled, check_additivity=False)
global_shap_values = np.abs(shap_values_train.values).mean(axis=0)
global_shap_importance = pd.DataFrame(list(zip(X_train.columns, global_shap_values)), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)

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
    global test_data  
    data = request.get_json()
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404
    test_data.loc[test_data['client_id'] == client_id, list(data.keys())] = list(data.values())
    return jsonify({"message": "Client information updated"}), 200

@app.route('/client_info', methods=['POST'])
def submit_new_client():
    global test_data
    data = request.get_json()
    new_client_id = len(test_data) + 1
    data['client_id'] = new_client_id
    test_data = pd.concat([test_data, pd.DataFrame(data, index=[0])], ignore_index=True)
    return jsonify({
        "message": "New client submitted",
        "client_id": new_client_id
    }), 201


@app.route('/prediction', methods=['POST'])
def get_prediction():
    data = request.get_json()
    client_id = data.get('client_id')
    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    # Filtrer les colonnes inattendues
    info_client = client_data.drop('client_id', axis=1)
    
    # Appliquer les transformations et prédire
    info_client_scaled = scaler.transform(info_client)
    prediction = model.predict_proba(info_client_scaled)[0][1]
    
    return jsonify({"prediction": prediction})

@app.route('/shap_summary_plot/<int:client_id>', methods=['GET'])
def shap_summary_plot(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    info_client_scaled = scaler.transform(info_client)

    shap_values = explainer(info_client_scaled, check_additivity=False)
    shap.summary_plot(shap_values.values, info_client, plot_type="bar", max_display=10)
    
    # Sauvegarder le plot en tant qu'image dans un buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({"shap_summary_plot": image_base64})

@app.route('/global_feature_importance', methods=['GET'])
def global_feature_importance():
    global_shap_importance_dict = global_shap_importance.set_index('feature')['importance'].to_dict()
    return jsonify(global_shap_importance_dict)

@app.route('/local_feature_importance/<int:client_id>', methods=['GET'])
def local_feature_importance(client_id):
    client_data = test_data[test_data['client_id'] == client_id]
    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    info_client = client_data.drop('client_id', axis=1)
    info_client_scaled = scaler.transform(info_client)

    shap_values = explainer(info_client_scaled, check_additivity=False)
    local_shap_values = np.abs(shap_values.values[0])
    local_shap_importance = pd.DataFrame(list(zip(info_client.columns, local_shap_values)), columns=['feature', 'importance']).sort_values(by='importance', ascending=False)

    local_shap_importance_dict = local_shap_importance.set_index('feature')['importance'].to_dict()
    return jsonify(local_shap_importance_dict)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)


