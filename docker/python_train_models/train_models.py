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
import logging
from sqlalchemy import create_engine

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


def connect(config):
    """Connecte au serveur PostgreSQL et retourne l'engine."""
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
        )
        logger.info("Connected to the PostgreSQL server.")
        return engine
    except Exception as error:
        logger.error(f"Connection error: {error}")
        return None


def fetch_table(table):
    """Récupère lignes d'une table et retourne un DataFrame."""
    config = load_config()
    engine = connect(config)

    if engine is not None:
        try:
            query = f"SELECT * FROM {table};"
            df = pd.read_sql_query(query, engine)
            logger.info(f"Data {table} fetched successfully.")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
        finally:
            engine.dispose()  # Dispose of the engine properly
    else:
        logger.error("Failed to connect to the database.")
        return None


def authenticate_mlflow():
    """Authentifie MLflow en utilisant les variables d'environnement."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_username and mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    else:
        logger.error("Variables username et password MLflow absentes.")


########" MODELE DE RECOMMANDATION DE FILMS - SANS USERID ########"
def train_TFIDF_model(df, data_directory):
    """
    Entraîne un modèle TF-IDF pour extraire des caractéristiques des genres de films,
    et enregistre le modèle directement sur MLflow.
    """
    with mlflow.start_run() as run:

        start_time = datetime.now()  # Démarrer la mesure du temps

        # Vérifier les colonnes et le contenu
        logger.info("Colonnes du DataFrame : %s", df.columns)
        logger.info("Aperçu du DataFrame :\n%s", df.head())

        # Créer une instance de TfidfVectorizer
        tfidf = TfidfVectorizer()

        # Calculer la matrice TF-IDF
        tfidf_matrix = tfidf.fit_transform(df["genres"])

        # Afficher la taille de la matrice
        logger.info(f"Dimensions de notre matrice TF-IDF : {tfidf_matrix.shape}")
        mlflow.log_param("tfidf_matrix_shape", tfidf_matrix.shape)

        # Calculer la similarité cosinus par morceaux
        sim_cosinus = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info(
            f"Dimensions de la matrice de similarité cosinus : {sim_cosinus.shape}"
        )
        mlflow.log_param("sim_cosinus_shape", sim_cosinus.shape)

        indices = pd.Series(range(0, len(df)), index=df["title"])

        # Enregistrer le modèle TF-IDF sur MLflow
        mlflow.sklearn.log_model(
            tfidf, artifact_path="tfidf_model", registered_model_name="TFIDFModel"
        )

        # Enregistrer sim_cosinus et indices comme artifacts
        mlflow.log_artifact(
            pd.DataFrame(sim_cosinus).to_csv(header=False, index=False),
            artifact_path="sim_cosinus.csv",
        )

        mlflow.log_artifact(indices.to_csv(header=True), artifact_path="indices.csv")

        # Sauvegadrer modèles vers data_directory ("/root/mount_file/models/")
        with open(f"{data_directory}/tfidf.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        with open(f"{data_directory}/sim_cosinus.pkl", "wb") as f:
            pickle.dump(sim_cosinus, f)
        with open(f"{data_directory}/indices.pkl", "wb") as f:
            pickle.dump(indices, f)

        logger.info("TF-IDF model trained and logged successfully.")

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
    """Entraîne le modèle de recommandation sur les données fournies,
    et enregistre le modèle directement sur MLflow.
    """
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

        logger.info("Début de la cross-validation")

        # Effectuer la validation croisée sur le Trainset
        cv_results = cross_validate(
            model, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True
        )

        # Afficher les résultats
        mean_rmse = cv_results["test_rmse"].mean()

        logger.info("Moyenne des RMSE : %s", mean_rmse)

        # Log les métriques
        mlflow.log_metric("mean_rmse", mean_rmse)

        mlflow.log_metric("mean_fit_time", np.mean(cv_results["fit_time"]))

        mlflow.log_metric("mean_test_rmse", np.mean(cv_results["test_rmse"]))

        # Enregistrer le modèle Surprise SVD sur MLflow
        mlflow.pyfunc.log_model(
            python_model=SurpriseModelWrapper(model=model, reader=reader),
            artifact_path="surprise_svd_model",
            registered_model_name="SurpriseSVDModel",
            code_path=[
                "./"
            ],  # Path to the directory containing the code for your model
            pip_requirements=[
                "scikit-surprise==1.1.3",
                "pandas==2.0.0",
            ],  # Add the library as dependency
        )
        # Sauvegadrer modèles et reader vers data_directory ("/root/mount_file/models/")
        with open(f"{data_directory}/model_svd.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"{data_directory}/reader.pkl", "wb") as f:
            pickle.dump(reader, f)
        logger.info("Surprise SVD model trained and logged successfully.")

        return model, mean_rmse, reader


class SurpriseModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, reader):
        self.model = model
        self.reader = reader

    def predict(self, context, model_input):
        """
        Make predictions using the wrapped Surprise model.
        """
        predictions = []
        for _, row in model_input.iterrows():
            userid = row["userid"]
            movieid = row["movieid"]

            try:
                prediction = self.model.predict(userid, movieid)
                predictions.append(prediction.est)
            except Exception as e:
                logger.error(f"Error predicting for user {userid}, item {movieid}: {e}")
                predictions.append(
                    self.reader.rating_scale[0]
                )  # Or another suitable default

        return predictions


if __name__ == "__main__":
    logger.info("########## TRAIN MODELS ##########")
    data_directory = "/root/mount_file/models/"
    authenticate_mlflow()

    ratings = fetch_table("ratings")
    movies = fetch_table("movies")

    df = pd.merge(ratings, movies, on="movieid", how="left")

    logger.info("Entrainement du modèle TF-IDF")
    train_TFIDF_model(movies, data_directory)

    logger.info("Entrainement du modèle Surprise SVD")
    train_model(df, data_directory)
