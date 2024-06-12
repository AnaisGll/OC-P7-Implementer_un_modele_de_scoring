import pytest
from app import app

@pytest.fixture
def client():
    """Fixture pour créer un client de test pour l'application Flask."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    """Test de la route racine."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'API pour prédire l\'accord d\'un prêt' in response.data

def test_check_client_id_exist(client):
    """Test de la recherche d'un client existant."""
    response = client.get('/check_client/1')
    assert response.status_code == 200
    assert response.json == True

def test_check_client_id_not_exist(client):
    """Test de la recherche d'un client inexistant."""
    response = client.get('/check_client/1000')
    assert response.status_code == 200
    assert response.json == False

def test_get_prediction(client):
    """Test de la route de prédiction."""
    data = {"client_id": 1}
    response = client.post('/prediction', json=data)
    assert response.status_code == 200
    assert 'prediction' in response.json
    assert 'shap_values' in response.json

def test_shap_values_local(client):
    """Test de la route des valeurs SHAP locales."""
    response = client.get('/shaplocal/1')
    assert response.status_code == 200
    assert 'shap_values' in response.json
    assert 'base_value' in response.json
    assert 'data' in response.json
    assert 'feature_names' in response.json

def test_shap_values(client):
    """Test de la route des valeurs SHAP globales."""
    response = client.get('/shap/')
    assert response.status_code == 200
    assert 'shap_values_0' in response.json
    assert 'shap_values_1' in response.json

