import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, table, column
from sqlalchemy.dialects.postgresql import insert
import datetime


def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


def connect(config):
    """Connecte au serveur PostgreSQL et retourne la connexion."""
    try:
        conn = psycopg2.connect(**config)
        print("Connected to the PostgreSQL server.")
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Connection error: {error}")
        return None


def fetch_table(table):
    """Récupère lignes d'une table et retourne un DataFrame."""
    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            query = f"""
                SELECT *
                FROM {table};
            """
            df = pd.read_sql_query(query, conn)
            print(f"Data {table} fetched successfully.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            conn.close()  # Assurez-vous de fermer la connexion
    else:
        print("Failed to connect to the database.")
        return None


def save_data(df, data_directory, table_name):
    """
    Enregistre les DataFrames traités dans des fichiers CSV avec un nom basé sur la date et l'heure.
    """
    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    try:
        # Générer un horodatage pour le nom du fichier
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_{table_name}_{timestamp}.csv"

        # Enregistrement des fichiers CSV
        df.to_csv(os.path.join(data_directory, filename), index=False)

        print(f"{filename} loaded in {data_directory}")

    except IOError as e:
        print(f"Error saving files: {e}")


if __name__ == "__main__":
    data_directory = "/root/mountfile/raw/silver"

    # Chargement des données à partir du chemin spécifié et enregistrement avec horodatage
    processed_ratings = fetch_table("ratings")
    if processed_ratings is not None:
        save_data(
            processed_ratings, data_directory=data_directory, table_name="ratings"
        )

    processed_movies = fetch_table("movies")
    if processed_movies is not None:
        save_data(processed_movies, data_directory=data_directory, table_name="movies")

    processed_links = fetch_table("links")
    if processed_links is not None:
        save_data(processed_links, data_directory=data_directory, table_name="links")
