import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
import pickle
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import numpy as np
import psycopg2
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from surprise.model_selection import cross_validate


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


def authenticate_mlflow():
    """Authentifie MLflow en utilisant les variables d'environnement."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_username and mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password


########" MODELE DE RECOMMANDATION DE FILMS - SANS USERID ########"
def train_TFIDF_model(df, data_directory):
    """
    Entraîne un modèle TF-IDF pour extraire des caractéristiques des genres de films.
    """
    # Démarrer une nouvelle expérience MLflow
    with mlflow.start_run() as run:

        start_time = datetime.now()  # Démarrer la mesure du temps

        # Vérifier les colonnes et le contenu
        print("Colonnes du DataFrame :", df.columns)
        print("Aperçu du DataFrame :")
        print(df.head())

        # Créer une instance de TfidfVectorizer
        tfidf = TfidfVectorizer()

        # Calculer la matrice TF-IDF
        tfidf_matrix = tfidf.fit_transform(df["genres"])

        # Afficher la taille de la matrice
        print(f"Dimensions de notre matrice TF-IDF : {tfidf_matrix.shape}")
        mlflow.log_param("tfidf_matrix_shape", tfidf_matrix.shape)

        # Calculer la similarité cosinus par morceaux
        sim_cosinus = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print(f"Dimensions de la matrice de similarité cosinus : {sim_cosinus.shape}")
        mlflow.log_param("sim_cosinus_shape", sim_cosinus.shape)
        indices = pd.Series(range(0, len(df)), index=df["title"])
        os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

        # Sauvegarder les éléments essentiels
        joblib.dump(tfidf, os.path.join(data_directory, "tfidf_model.joblib"))
        joblib.dump(sim_cosinus, os.path.join(data_directory, "sim_cosinus.joblib"))
        indices.to_pickle(os.path.join(data_directory, "indices.pkl"))

        # Log les modèles avec MLflow
        mlflow.log_artifact(
            os.path.join(data_directory, "tfidf_model.joblib"),
            artifact_path="tfidf_model",
        )
        mlflow.log_artifact(
            os.path.join(data_directory, "sim_cosinus.joblib"),
            artifact_path="sim_cosinus",
        )
        mlflow.log_artifact(
            os.path.join(data_directory, "indices.pkl"), artifact_path="indices"
        )

        return tfidf, sim_cosinus, indices


#### MODELE DE RECOMMANDATION DE FILMS - AVEC USERID ####


def train_model(
    df: pd.DataFrame,
    data_directory: str,
    n_factors: int = 150,
    n_epochs: int = 30,
    lr_all: float = 0.01,
    reg_all: float = 0.05,
) -> SVD:
    """Entraîne le modèle de recommandation sur les données fournies."""
    with mlflow.start_run() as run:

        # Log des hyperparamètres du modèle
        mlflow.log_param("n_factors", n_factors)
        mlflow.log_param("n_epochs", n_epochs)
        mlflow.log_param("lr_all", lr_all)
        mlflow.log_param("reg_all", reg_all)

        # Diviser les données en ensembles d'entraînement et de test
        reader = Reader(rating_scale=(0.5, 5))

        data = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)

        # Extraire le Trainset
        trainset = data.build_full_trainset()

        model = SVD(
            n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all
        )

        # Entraîner le modèle
        model.fit(trainset)

        print("Début de la cross-validation")

        # Effectuer la validation croisée sur le Trainset
        cv_results = cross_validate(
            model, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True
        )

        # Afficher les résultats
        mean_rmse = cv_results["test_rmse"].mean()
        print("Moyenne des RMSE :", mean_rmse)

        # Log les métriques
        mlflow.log_metric("mean_rmse", mean_rmse)

        # Sauvegarder le modèle
        os.makedirs(data_directory, exist_ok=True)

        # Sauvegarder le modèle et le contexte (reader)
        model_path = os.path.join(data_directory, "svd_model_v1.pkl")
        with open(model_path, "wb") as file:
            pickle.dump({"model": model, "reader": reader}, file)
        mlflow.log_artifact(model_path, artifact_path="surprise_model")
        mlflow.log_metric("mean_fit_time", np.mean(cv_results["fit_time"]))
        mlflow.log_metric("mean_test_rmse", np.mean(cv_results["test_rmse"]))

        return model, mean_rmse, reader


if __name__ == "__main__":
    print("########## :hammer: TRAIN MODELS ##########")
    data_directory = "/root/mount_file/models/"
    authenticate_mlflow()
    ratings = fetch_table("ratings")
    movies = fetch_table("movies")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    print("Entrainement du modèle TF-IDF")
    train_TFIDF_model(movies, data_directory)
    print("Entrainement du modèle Surprise SVD")
    train_model(df, data_directory)
