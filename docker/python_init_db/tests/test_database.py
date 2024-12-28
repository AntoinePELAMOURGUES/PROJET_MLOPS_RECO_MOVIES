import os
import pytest
import psycopg
from app.init_db import load_config, connect, create_tables
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="module")
def postgres():
    """Fixture pour démarrer un conteneur PostgreSQL."""
    with PostgresContainer("postgres:14-alpine") as postgres_container:
        # Configurer les variables d'environnement pour la connexion
        os.environ["DB_HOST"] = postgres_container.get_container_host_ip()
        os.environ["DB_PORT"] = postgres_container.get_exposed_port(5432)
        os.environ["DB_USERNAME"] = postgres_container.username
        os.environ["DB_PASSWORD"] = postgres_container.password
        os.environ["DB_NAME"] = postgres_container.dbname  # Nom de la base de données

        # Attendre que PostgreSQL soit prêt à accepter des connexions
        postgres_container.start()

        yield postgres_container


def test_database_connection(postgres):
    """Test de la connexion à la base de données."""
    conn_url = f"postgresql://{postgres.username}:{postgres.password}@{postgres.get_container_host_ip()}:{postgres.get_exposed_port(5432)}/{postgres.dbname}"

    # Vérifiez que l'URL de connexion est disponible
    assert conn_url is not None

    # Testez la connexion
    conn = psycopg.connect(conn_url)
    assert conn is not None  # Vérifiez que la connexion est établie
    conn.close()  # Fermez la connexion après le test


def test_load_config(postgres, monkeypatch):
    """Test de la fonction load_config."""
    monkeypatch.setenv("POSTGRES_HOST", postgres.get_container_host_ip())
    monkeypatch.setenv("POSTGRES_DB", postgres.dbname)
    monkeypatch.setenv("POSTGRES_USER", postgres.username)
    monkeypatch.setenv("POSTGRES_PASSWORD", postgres.password)

    config = load_config()

    assert config["host"] == postgres.get_container_host_ip()
    assert config["database"] == postgres.dbname
    assert config["user"] == postgres.username
    assert config["password"] == postgres.password


def test_create_tables(postgres):
    """Test de la création des tables."""
    conn_url = f"postgresql://{postgres.username}:{postgres.password}@{postgres.get_container_host_ip()}:{postgres.get_exposed_port(5432)}/{postgres.dbname}"

    conn = psycopg.connect(conn_url)
    cursor = conn.cursor()

    # Créer les tables (vous devez appeler create_tables ici)
    create_tables(conn)

    # Vérifier si les tables ont été créées
    cursor.execute("SELECT to_regclass('movies');")
    movies_exists = cursor.fetchone()[0] is not None

    cursor.execute("SELECT to_regclass('ratings');")
    ratings_exists = cursor.fetchone()[0] is not None

    cursor.execute("SELECT to_regclass('links');")
    links_exists = cursor.fetchone()[0] is not None

    cursor.execute("SELECT to_regclass('users');")
    users_exists = cursor.fetchone()[0] is not None

    assert movies_exists is True
    assert ratings_exists is True
    assert links_exists is True
    assert users_exists is True

    cursor.close()
    conn.close()


def test_connect(postgres):
    """Test de la fonction connect."""
    config = {
        "host": postgres.get_container_host_ip(),
        "database": postgres.dbname,
        "user": postgres.username,
        "password": postgres.password,
    }

    conn = connect(config)

    assert conn is not None
