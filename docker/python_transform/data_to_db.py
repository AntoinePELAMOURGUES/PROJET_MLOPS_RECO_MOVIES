import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, table, column
from sqlalchemy.dialects.postgresql import insert

# Définition des tables SQLAlchemy pour les opérations d'upsert
table_movies = table('movies',
    column('movieid'),
    column('title'),
    column('genres'),
    column('year')
)

table_ratings = table('ratings',
    column('userid'),
    column('movieid'),
    column('rating'),
    column('timestamp'),
    column('bayesian_mean')
)

table_links = table('links',
    column('movieid'),
    column('imdbid'),
    column('tmdbid')
)

table_users = table('users',
    column('userid'),
    column('username'),
    column('email'),
    column('hached_password')
)

def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        'host': os.getenv('AIRFLOW_POSTGRESQL_SERVICE_HOST'),
        'database': os.getenv('DATABASE'),
        'user': os.getenv('USER'),
        'password': os.getenv('PASSWORD')
    }

def connect(config):
    """Connecte au serveur PostgreSQL et retourne la connexion."""
    try:
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Connection error: {error}")
        return None

def execute_query_psql(query, config):
    """Exécute une requête SQL pour insérer des données dans une table."""
    conn_string = f'postgresql://{config["user"]}:{config["password"]}@{config["host"]}/{config["database"]}'

    try:
        with create_engine(conn_string).begin() as conn:
            res = conn.execute(query)
            return res.rowcount  # Retourne le nombre de lignes affectées par la requête
    except Exception as e:
        print(f"Error executing query: {e}")
        return 0

def upsert_to_psql(table, df):
    """Insère ou met à jour des enregistrements dans une table.

    Args:
        table: La table SQLAlchemy cible.
        df (pd.DataFrame): DataFrame contenant les données à insérer ou mettre à jour.

    """

    # Préparation des données pour l'upsert
    dict_data = df.where(pd.notnull(df), None).to_dict(orient='records')

    # Création de l'instruction d'insertion avec gestion des conflits
    insert_stmt = insert(table).values(dict_data)

    do_update_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[table.c.id],  # Remplacez par la clé primaire appropriée
        set_={col.name: insert_stmt.excluded[col.name] for col in table.columns}
    )

    rowcount = execute_query_psql(do_update_stmt, config)

    if rowcount > 0:
        print(f'{rowcount} rows have been inserted or updated in {table.name}')
    else:
        print(f'No rows were inserted or updated in {table.name}')

if __name__ == '__main__':
    data_directory = 'app/data'

    config = load_config()

    # Chargement et traitement des fichiers CSV par morceaux
    for filename, table in [
        (f'{data_directory}/to_ingest/silver/processed_ratings.csv', table_ratings),
        (f'{data_directory}/to_ingest/silver/processed_movies.csv', table_movies),
        (f'{data_directory}/to_ingest/silver/processed_links.csv', table_links),
        (f'{data_directory}/to_ingest/silver/users.csv', table_users)
    ]:
        try:
            for chunk in pd.read_csv(filename, chunksize=1000):  # Lire par morceaux de 1000 lignes
                upsert_to_psql(table, chunk)
            print(f"Finished processing {filename}")

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except pd.errors.EmptyDataError as e:
            print(f"No data in file: {e}")
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")