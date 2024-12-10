
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from docker.python_train_models.train_models import fetch_ratings, train_SVD_model, train_matrix_model

@pytest.fixture
def sample_ratings():
    data = {
        'userId': [1, 2, 3, 4, 5],
        'movieId': [10, 20, 30, 40, 50],
        'rating': [4.0, 3.5, 5.0, 2.0, 4.5]
    }
    return pd.DataFrame(data)

@patch('train_models.connect')
@patch('train_models.load_config')
def test_fetch_ratings(mock_load_config, mock_connect, sample_ratings):
    mock_load_config.return_value = {
        'host': 'localhost',
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_password'
    }
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor().fetchall.return_value = sample_ratings.values.tolist()
    mock_conn.cursor().description = [('userid',), ('movieid',), ('rating',)]

    df = fetch_ratings('ratings')
    assert df.equals(sample_ratings)

@patch('train_models.mlflow')
@patch('train_models.pickle')
def test_train_SVD_model(mock_pickle, mock_mlflow, sample_ratings, tmpdir):
    data_directory = tmpdir.mkdir("data")
    train_SVD_model(sample_ratings, str(data_directory))

    model_path = data_directory.join("model_SVD.pkl")
    assert model_path.check(file=1)

@patch('train_models.mlflow')
@patch('train_models.pickle')
def test_train_matrix_model(mock_pickle, mock_mlflow, sample_ratings, tmpdir):
    data_directory = tmpdir.mkdir("data")
    train_matrix_model(sample_ratings, str(data_directory))

    model_path = data_directory.join("model_KNN.pkl")
    assert model_path.check(file=1)