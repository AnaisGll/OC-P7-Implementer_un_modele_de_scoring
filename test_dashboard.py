import pytest
from unittest.mock import patch
from dashboard import get_prediction, jauge_score, plot_client_features

# Mock des réponses de l'API
mock_api_response = {
    "prediction": 0.7
}

@pytest.mark.parametrize("client_id, expected_proba_default, expected_decision", [
    (1, 0.7, "Refusé"),
    (2, None, None)
])
@patch("dashboard.requests.post")
def test_get_prediction(mock_post, client_id, expected_proba_default, expected_decision):
    # Configuration du mock de la réponse de l'API
    mock_post.return_value.json.return_value = mock_api_response

    # Appel de la fonction get_prediction avec le client_id
    proba_default, decision = get_prediction(client_id)

    # Vérification des résultats
    assert proba_default == expected_proba_default
    assert decision == expected_decision

def test_jauge_score():
    # Pas de véritable test pour cette fonction, car elle ne retourne pas de valeur mais appelle plotly_chart
    pass

def test_plot_client_features():
    # À implémenter en fonction des tests nécessaires pour la fonction plot_client_features
    pass

    pytest.main()



