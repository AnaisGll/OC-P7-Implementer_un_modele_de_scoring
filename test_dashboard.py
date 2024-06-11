import pytest
import requests
import pandas as pd
from dashboard import get_prediction, get_client_data

# Mock data to simulate test dataset
test_data = pd.DataFrame({
    'client_id': [123, 456, 789],
    'feature1': [10, 20, 30],
    'feature2': [100, 200, 300]
})

# Mock response class for requests
class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.json_data = json_data

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError(f"{self.status_code} Error")

# Test get_prediction function
def test_get_prediction(monkeypatch):
    # Mock successful API response
    def mock_post_success(*args, **kwargs):
        return MockResponse(200, {"prediction": 0.75})

    monkeypatch.setattr(requests, 'post', mock_post_success)
    assert get_prediction(123) == 0.75

    # Mock failed API response
    def mock_post_failure(*args, **kwargs):
        return MockResponse(400, {"error": "Invalid request"})

    monkeypatch.setattr(requests, 'post', mock_post_failure)
    assert get_prediction(123) is None

# Test get_client_data function
def test_get_client_data(monkeypatch):
    # Mock successful API response
    def mock_get_success(*args, **kwargs):
        return MockResponse(200, True)

    monkeypatch.setattr(requests, 'get', mock_get_success)
    client_data = get_client_data(123, test_data)
    assert client_data['client_id'] == 123
    assert client_data['feature1'] == 10

    # Mock failed API response
    def mock_get_failure(*args, **kwargs):
        return MockResponse(404, False)

    monkeypatch.setattr(requests, 'get', mock_get_failure)
    client_data = get_client_data(123, test_data)
    assert client_data is None

if __name__ == '__main__':
    pytest.main()



