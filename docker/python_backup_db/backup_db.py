import os
import psycopg2
from psycopg2 import sql

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

def backup_database(conn, data_directory, backup_file):
    """Effectue une sauvegarde de la base de données dans un fichier SQL."""
    try:
        os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire
        # Chemin complet du fichier de sauvegarde
        backup_path = os.path.join(data_directory, backup_file)
        with conn.cursor() as cursor:
            # Exécute la commande pour sauvegarder la base de données
            with open(backup_path, 'w') as f:
                # Utilise pg_dump pour créer le backup
                cursor.copy_expert(sql.SQL("COPY (SELECT * FROM pg_catalog.pg_tables) TO STDOUT WITH CSV HEADER"), f)
            print(f"Backup successful! Saved to {backup_path}")
    except Exception as e:
        print(f"Error during backup: {e}")

def main():
    config = load_config()
    conn = connect(config)

    if conn is not None:
        data_directory = '/root/mount_file/'
        backup_file = f"{config['database']}_backup.sql"  # Nom du fichier de sauvegarde
        backup_database(conn, data_directory, backup_file)
        conn.close()

if __name__ == "__main__":
    main()