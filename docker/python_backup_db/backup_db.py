import os
import psycopg2
import subprocess

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

def backup_database(config, data_directory, backup_file):
    """Effectue une sauvegarde de la base de données dans un fichier SQL en utilisant pg_dump."""
    try:
        os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire
        # Chemin complet du fichier de sauvegarde
        backup_path = os.path.join(data_directory, backup_file)

        # Commande pg_dump
        dump_command = [
            "pg_dump",
            "-h", config['host'],
            "-U", config['user'],
            "-d", config['database'],
            "-f", backup_path,
            "--no-password"  # Pour ne pas demander le mot de passe ici
        ]

        # Exécute pg_dump
        env = os.environ.copy()
        env['PGPASSWORD'] = config['password']  # Définit le mot de passe dans l'environnement

        subprocess.run(dump_command, env=env, check=True)  # Lancer la commande pg_dump
        print(f"Backup successful! Saved to {backup_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error during backup: {e}")
    except Exception as e:
        print(f"General error during backup: {e}")

def main():
    config = load_config()
    conn = connect(config)

    if conn is not None:
        data_directory = '/root/mount_file/'  # Chemin où le fichier de sauvegarde sera enregistré
        backup_file = f"{config['database']}_backup.sql"  # Nom du fichier de sauvegarde
        backup_database(config, data_directory, backup_file)
        conn.close()

if __name__ == "__main__":
    main()
