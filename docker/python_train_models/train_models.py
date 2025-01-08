import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import psycopg2
import mlflow
from sklearn.metrics.pairwise import cosine_similarity


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


def fetch_ratings(table):
    """Récupère 15 % des dernières lignes de la table ratings et retourne un DataFrame."""
    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            # Étape 1: Comptez le nombre total de lignes dans la table
            count_query = f"SELECT COUNT(*) FROM {table};"
            total_count = pd.read_sql_query(count_query, conn).iloc[0, 0]

            # Étape 2: Calculez 15 % du nombre total de lignes
            limit = int(total_count * 0.15)

            # Étape 3: Récupérez les dernières lignes
            query = f"""
                SELECT userid, movieid, rating
                FROM {table}
                ORDER BY userid DESC
                LIMIT {limit};
            """
            df = pd.read_sql_query(query, conn)
            print("Data fetched successfully.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            conn.close()  # Assurez-vous de fermer la connexion
    else:
        print("Failed to connect to the database.")
        return None

def fetch_movies(table):
    """Récupère la table movies et retourne un DataFrame."""
    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            query = f"""
                SELECT movieid, genres
                FROM {table};
            """
            df = pd.read_sql_query(query, conn)
            print("Data fetched successfully.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            conn.close()  # Assurez-vous de fermer la connexion
    else:
        print("Failed to connect to the database.")
        return None


# Chargement du dernier modèle
def load_model(model_name):
    """Charge le modèle à partir du répertoire monté."""
    model_path = f"/models/{model_name}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle {model_name} n'existe pas dans {model_path}."
        )
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        print(f"Modèle chargé depuis {model_path}")
    return model


def train_SVD_model(df, data_directory) -> tuple:
    """Entraîne un modèle SVD de recommandation et sauvegarde le modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes userId, movieId et rating.
    """

    # Démarrer une nouvelle expérience MLflow
    mlflow.start_run()

    start_time = datetime.now()  # Démarrer la mesure du temps

    # Préparer les données pour Surprise
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userid', 'movieid', 'rating']], reader=reader)

    # Diviser les données en ensembles d'entraînement et de test
    trainset, testset = train_test_split(data, test_size=0.10)

    # Créer et entraîner le modèle SVD
    model = load_model('model_SVD.pkl')
    model.fit(trainset)

    # Tester le modèle sur l'ensemble de test et calculer RMSE
    predictions = model.test(testset)
    acc = accuracy.rmse(predictions)

    # Arrondir à 2 chiffres après la virgule
    acc_rounded = round(acc, 2)

    print("Valeur de l'écart quadratique moyen (RMSE) :", acc_rounded)

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement du modèle avec pickle
    with open(f'{data_directory}/model_SVD.pkl', 'wb') as f:
        pickle.dump(model, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_SVD.pkl.")

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("RMSE", acc_rounded)

    # Enregistrer le modèle avec MLflow
    mlflow.sklearn.log_model(model, "model_SVD")

    end_time = datetime.now()

    duration = end_time - start_time
    print(f"Durée de l'entraînement : {duration}")

    # Finir l'exécution de l'expérience MLflow
    mlflow.end_run()

def train_cosine_similarity(movies, data_directory):
    """
    Calcule la similarité cosinus entre les films en utilisant les genres des films dans le cadre d'un démarrage à froid.
    """
     # Démarrer une nouvelle expérience MLflow
    mlflow.start_run()
    # Vérifier les colonnes et le contenu
    start_time = datetime.now()  # Démarrer la mesure du temps
    # Supprimer les espaces dans les genres
    if "genres" in movies.columns:

        # Supprimer les espaces autour des virgules
        movies["genres"] = movies["genres"].str.replace(
            " ", ""
        )
        # Nettoyer les genres en supprimant les espaces au début et à la fin
        movies["genres"] = movies["genres"].str.strip()

        # Créer des variables indicatrices pour les genres
        genres = movies["genres"].str.get_dummies(sep=",")

        # Calculer la similarité cosinus
        cosine_sim = cosine_similarity(genres, genres)

        end_time = datetime.now()

        duration = end_time - start_time
        print(f'Durée de l\'entraînement : {duration}')
        print(f"Dimensions de notre matrice de similarité cosinus : {cosine_sim.shape}")

        os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

        # Enregistrement du modèle avec NumPy
        np.save(f"{data_directory}/cosine_similarity_matrix.npy", cosine_sim)
        print(
            f"Matrice de similarité enregistrée avec NumPy sous {data_directory}/cosine_similarity_matrix.npy."
        )
        # Enregistrer le modèle avec MLflow
        mlflow.log_param("training_duration", duration.total_seconds())
        mlflow.sklearn.log_model(cosine_sim, "cosine_similarity_matrix")

        mlflow.end_run()  # Finir l'exécution de l'expérience MLflow
    else:
        print("La colonne 'genres' n'existe pas dans le DataFrame.")


def authenticate_mlflow():
    """Authentifie MLflow en utilisant les variables d'environnement."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Movie Recommendation Models")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_username and mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

if __name__ == "__main__":
    data_directory = '/root/mount_file/models'
    authenticate_mlflow()
    ratings = fetch_ratings('ratings')
    movies = fetch_movies('movies')
    print('Entrainement du modèle SVD')
    train_SVD_model(ratings, data_directory)
    print('Création de notre matrice cosinus')
    train_cosine_similarity(movies, data_directory)