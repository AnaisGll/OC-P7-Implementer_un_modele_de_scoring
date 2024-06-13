import pytest
import requests
from dashboard import get_prediction, get_shap_values

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

            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")
                    
        return MockResponse(200, {"prediction": 0.75})

    monkeypatch.setattr(requests, 'post', mock_post_success)

    # Test de récupération de prédiction réussie
    assert get_prediction(123) == 0.75

    # Simulation d'une réponse échouée de l'API
    def mock_post_failure(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data

            def json(self):
                return self.json_data
            
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")

        return MockResponse(400, {"error": "Invalid request"})

    monkeypatch.setattr(requests, 'post', mock_post_failure)

    # Test de récupération de prédiction échouée
    assert get_prediction(123) is None

def test_get_shap_values(monkeypatch):
    # Simulation d'une réponse réussie de l'API pour SHAP values
    def mock_post_success(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data

            def json(self):
                return self.json_data

            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")
                    
        return MockResponse(200, {
            "shap_values": [0.1, 0.2, -0.1],
            "base_value": 0.5,
            "feature_names": ["feature1", "feature2", "feature3"]
        })

    monkeypatch.setattr(requests, 'post', mock_post_success)

    # Test de récupération des valeurs SHAP réussie
    response = get_shap_values(123)
    assert response is not None
    assert "shap_values" in response
    assert "base_value" in response
    assert "feature_names" in response

    # Simulation d'une réponse échouée de l'API pour SHAP values
    def mock_post_failure(*args, **kwargs):
        class MockResponse:
            def __init__(self, status_code, json_data):
                self.status_code = status_code
                self.json_data = json_data

            def json(self):
                return self.json_data
            
            def raise_for_status(self):
                if self.status_code != 200:
                    raise requests.exceptions.HTTPError(f"{self.status_code} Error")

        return MockResponse(400, {"error": "Invalid request"})

    monkeypatch.setattr(requests, 'post', mock_post_failure)

    # Test de récupération des valeurs SHAP échouée
    assert get_shap_values(123) is None

if __name__ == '__main__':
    pytest.main()


