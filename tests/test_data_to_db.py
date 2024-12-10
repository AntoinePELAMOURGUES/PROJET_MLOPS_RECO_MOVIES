
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sqlalchemy import table, column
from docker.python_load_data.data_to_db import load_config, connect, execute_query_psql, upsert_to_psql

# Définition d'une table de test
test_table = table('test_table',
    column('id'),
    column('name')
)

@pytest.fixture
def config():
    return {
        'host': 'localhost',
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_password'
    }

def test_load_config(monkeypatch):
    monkeypatch.setenv('AIRFLOW_POSTGRESQL_SERVICE_HOST', 'localhost')
    monkeypatch.setenv('DATABASE', 'test_db')
    monkeypatch.setenv('USER', 'test_user')
    monkeypatch.setenv('PASSWORD', 'test_password')
    config = load_config()
    assert config['host'] == 'localhost'
    assert config['database'] == 'test_db'
    assert config['user'] == 'test_user'
    assert config['password'] == 'test_password'

def test_connect(config):
    with patch('psycopg2.connect') as mock_connect:
        mock_connect.return_value = MagicMock()
        conn = connect(config)
        assert conn is not None
        mock_connect.assert_called_once_with(**config)

def test_execute_query_psql(config):
    query = "SELECT 1"
    with patch('sqlalchemy.create_engine') as mock_create_engine:
        mock_conn = MagicMock()
        mock_create_engine.return_value.begin.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.rowcount = 1
        rowcount = execute_query_psql(query, config)
        assert rowcount == 1
        mock_conn.execute.assert_called_once_with(query)

def test_upsert_to_psql(config):
    df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
    with patch('data_to_db.execute_query_psql') as mock_execute_query_psql:
        mock_execute_query_psql.return_value = 2
        upsert_to_psql(test_table, df)
        assert mock_execute_query_psql.called
        assert mock_execute_query_psql.call_args[0][0].table.name == 'test_table'
        assert mock_execute_query_psql.call_args[0][1] == config