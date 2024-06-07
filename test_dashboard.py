import pytest
import requests
from dashboard import get_prediction

# Mocking the API response for testing purposes
def test_get_prediction(monkeypatch):
    # Simulation d'une réponse réussie de l'API
    def mock_post_success(*args, **kwargs):
        # classe MockResponse qui simule la réponse de l'API
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data

            def json(self):
                return self.json_data

        return MockResponse(200, {"prediction": 0.75})

    monkeypatch.setattr(requests, 'post', mock_post_success)

    # Test de récupération de prédiction réussie
    assert get_prediction(123) == 0.75

    # Simulation d'une réponse d'erreur de l'API
    def mock_post_error(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        return MockResponse(404, "Client ID not found")

    monkeypatch.setattr(requests, 'post', mock_post_error)

    # Test de la gestion des erreurs
    assert get_prediction(456) is None
