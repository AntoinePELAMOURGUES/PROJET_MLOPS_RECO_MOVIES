import os
import pytest
from unittest import mock
import importlib.util
import sys

# Définir le chemin du fichier à tester
module_file_path = 'docker/python_scrapping/scrapping.py'

# Charger le module
spec = importlib.util.spec_from_file_location("scrapping_py", module_file_path)
scrapping_py = importlib.util.module_from_spec(spec)
sys.modules["scrapping_py"] = scrapping_py
spec.loader.exec_module(scrapping_py)

@pytest.fixture
def mock_env_variables(monkeypatch):
    """Fixture pour simuler les variables d'environnement nécessaires."""
    monkeypatch.setenv("TMDB_TOKEN", "fake_token")
    monkeypatch.setenv("AIRFLOW_POSTGRESQL_SERVICE_HOST", "localhost")
    monkeypatch.setenv("DATABASE", "test_db")
    monkeypatch.setenv("USER", "test_user")
    monkeypatch.setenv("PASSWORD", "test_password")

def test_scrape_imdb_first_page(mock_env_variables):
    """Test de la fonction scrape_imdb_first_page."""
    with mock.patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b'<html><body><a class="ipc-title-link-wrapper" href="/title/tt1234567/">Film 1</a></body></html>'

        result = scrapping_py.scrape_imdb_first_page()
        assert result == ['1234567']


def test_genres_request(mock_env_variables):
    """Test de la fonction genres_request."""
    with mock.patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "genres": [
                {"id": 28, "name": "Action"},
                {"id": 35, "name": "Comedy"}
            ]
        }

        result = scrapping_py.genres_request()
        assert result == {'28': 'Action', '35': 'Comedy'}

def test_api_tmdb_request(mock_env_variables):
    """Test de la fonction api_tmdb_request."""
    with mock.patch('requests.get') as mock_get:
        # Mocking the response for the genres request
        mock_get.side_effect = [
            mock.Mock(status_code=200, json=lambda: {"genres": [{"id": 28, "name": "Action"}]}),
            mock.Mock(status_code=200, json=lambda: {
                "movie_results": [{
                    "id": 123,
                    "title": "Test Movie",
                    "release_date": "2024-01-01",
                    "genre_ids": [28]
                }]
            })
        ]

        result = scrapping_py.api_tmdb_request()
        assert isinstance(result, dict)
        assert '0' in result
        assert result['0']['title'] == 'Test Movie'
        assert result['0']['year'] == '2024'

@mock.patch('your_module.create_engine')
def test_insert_data_movies(mock_create_engine, mock_env_variables):
    """Test de la fonction insert_data_movies."""
    # Simuler la connexion à la base de données
    mock_conn = mock.Mock()
    mock_create_engine.return_value.__enter__.return_value = mock_conn

    # Simuler le retour de l'API TMDB
    with mock.patch('your_module.api_tmdb_request') as mock_api:
        mock_api.return_value = {
            '0': {
                'title': 'Test Movie',
                'genres': ['Action'],
                'imbd_id': '1234567',
                'tmdb_id': 123,
                'year': '2024'
            }
        }

        scrapping_py.insert_data_movies()

        # Vérifier que les méthodes d'insertion ont été appelées
        assert mock_conn.execute.call_count > 0

if __name__ == "__main__":
    pytest.main()