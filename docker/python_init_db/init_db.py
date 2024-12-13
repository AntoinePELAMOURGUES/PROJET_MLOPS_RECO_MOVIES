import os
import psycopg2


def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        'host': os.getenv('POSTGRES_HOST'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
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


sql_create_movies_table = """
CREATE TABLE IF NOT EXISTS movies (
    id SERIAL PRIMARY KEY,
    movieid INTEGER UNIQUE NOT NULL,
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
    hached_password VARCHAR(300) NOT NULL
);"""

def create_tables(conn):
    """Crée les tables dans la base de données."""
    cursor = conn.cursor()
    try:
        cursor.execute(sql_create_movies_table)
        cursor.execute(sql_create_ratings_table)
        cursor.execute(sql_create_links_table)
        cursor.execute(sql_create_users_table)
        conn.commit()
        print("Tables créées avec succès.")
    except Exception as e:
        print(f"Erreur lors de la création des tables: {e}")
    finally:
        cursor.close()

def main():
    print("Initialisation de la base de données...")
    config = load_config()
    conn = connect(config)
    if conn is not None:
        print("Connexion réussie à la base de données.")
        create_tables(conn)
        conn.close()
    else:
        print("Impossible de se connecter à la base de données.")

if __name__ == "__main__":
    main()

