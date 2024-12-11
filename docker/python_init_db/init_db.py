import os
import psycopg2
from psycopg2 import OperationalError

def create_connection():
    """Crée une connexion à la base de données PostgreSQL."""
    try:
        conn = psycopg2.connect(
            database=os.getenv("DATABASE"),
            host=os.getenv("AIRFLOW_POSTGRESQL_SERVICE_HOST"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            port=os.getenv("AIRFLOW_POSTGRESQL_SERVICE_PORT")
        )
        print("Connexion à la base de données réussie.")
        return conn
    except OperationalError as e:
        print(f"Erreur lors de la connexion à la base de données: {e}")
        return None


sql_create_movies_table = """
CREATE TABLE IF NOT EXISTS movies (
    movieid SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    genres VARCHAR,
    year INTEGER
);"""

sql_create_ratings_table = """
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    userid INTEGER,
    movieid INTEGER REFERENCES movies(movieid),
    rating FLOAT NOT NULL,
    timestamp INTEGER,
    bayesian_mean FLOAT NOT NULL
);"""

sql_create_links_table = """
CREATE TABLE IF NOT EXISTS links (
    id SERIAL PRIMARY KEY,
    movieid INTEGER REFERENCES movies(movieid),
    imdbid INTEGER,
    tmdbid INTEGER
);"""

sql_create_users_table = """
CREATE TABLE IF NOT EXISTS users (
    userid SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(300) NOT NULL
);"""

def create_tables(conn):
    """Crée les tables dans la base de données."""
    cursor = conn.cursor()
    cursor.execute(sql_create_movies_table)
    cursor.execute(sql_create_ratings_table)
    cursor.execute(sql_create_links_table)
    cursor.execute(sql_create_users_table)
    conn.commit()
    cursor.close()
    print("Tables créées avec succès.")

def main():
    conn = create_connection()
    if conn is not None:
        create_tables(conn)
        conn.close()
    else:
        print("Impossible de se connecter à la base de données.")

if __name__ == "__main__":
    main()

