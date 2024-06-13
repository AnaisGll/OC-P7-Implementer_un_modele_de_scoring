import pytest
from flask import Flask
import json
from api import app 

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "API pour prédire l'accord d'un prêt"

def test_check_client_id_exists(client):
    # Supposons que le client_id 1 existe dans les données test
    response = client.get('/check_client/1')
    assert response.status_code == 200
    assert response.json is True

def test_check_client_id_not_exists(client):
    # Supposons que le client_id 9999 n'existe pas dans les données test
    response = client.get('/check_client/9999')
    assert response.status_code == 200
    assert response.json is False

def test_get_prediction(client):
    # Supposons que le client_id 1 existe dans les données test
    response = client.post('/prediction', json={'client_id': 1})
    assert response.status_code == 200
    assert 'prediction' in response.json

def test_get_prediction_no_client_id(client):
    response = client.post('/prediction', json={})
    assert response.status_code == 400
    assert response.json == {"error": "client_id is required"}

def test_get_prediction_client_not_found(client):
    # Supposons que le client_id 9999 n'existe pas dans les données test
    response = client.post('/prediction', json={'client_id': 9999})
    assert response.status_code == 404
    assert response.json == {"error": "Client not found"}
