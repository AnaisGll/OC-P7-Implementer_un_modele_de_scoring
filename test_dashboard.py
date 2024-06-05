import pytest
import pandas as pd
from streamlit.testing import TestApp
import requests_mock

from my_dashboard import extract_data, request_prediction, main

# Charger les données de test
df = pd.read_csv('test_mean_sample.csv')

@pytest.fixture
def setup_data():
    return df

def test_extract_data_existing_id(setup_data):
    sk_id_curr = setup_data['SK_ID_CURR'].iloc[0]
    data = extract_data(sk_id_curr)
    assert not data.empty
    assert data.shape[0] == 1  

def test_extract_data_non_existing_id(setup_data):
    sk_id_curr = -1
    data = extract_data(sk_id_curr)
    assert data.empty

def test_request_prediction_success(requests_mock):
    model_uri = 'http://fakeapi/predict'
    data = [[1, 2, 3]]
    expected_response = {"predictions": [0]}
    requests_mock.post(model_uri, json=expected_response)

    response = request_prediction(model_uri, data)
    assert response == expected_response["predictions"]

def test_request_prediction_failure(requests_mock):
    model_uri = 'http://fakeapi/predict'
    data = [[1, 2, 3]]
    requests_mock.post(model_uri, status_code=400, text="Bad Request")

    with pytest.raises(Exception) as excinfo:
        request_prediction(model_uri, data)
    assert "Request failed with status 400" in str(excinfo.value)

@pytest.fixture
def streamlit_app():
    return TestApp(main)

def test_streamlit_ui(streamlit_app):
    sk_id_curr = df['SK_ID_CURR'].iloc[0]

    streamlit_app.input('Entrez le SK_ID_CURR :', sk_id_curr)
    streamlit_app.button('Prédire')

    result = streamlit_app.get_last_output()
    assert 'La prédiction est : Oui' in result or 'La prédiction est : Non' in result
