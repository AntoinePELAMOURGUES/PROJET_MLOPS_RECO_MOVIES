import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
import pickle
from surprise.model_selection import cross_validate
import logging
from sqlalchemy import create_engine
from prometheus_client import Counter, Histogram, CollectorRegistry, Gauge, start_http_server
import time


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


# METRIQUES PROMETHEUS
TRAIN_DURATION = Histogram('model_training_duration_seconds', 'Duration of model training in seconds')
CROSS_VALIDATION_DURATION = Histogram('cross_validation_duration_seconds', 'Duration of cross-validation in seconds')
RMSE_GAUGE = Gauge('model_rmse', 'Root Mean Square Error of the model')
MAE_GAUGE = Gauge('model_mae', 'Mean Absolute Error of the model')
TRAIN_ITERATIONS = Counter('model_train_iterations_total', 'Total number of training iterations')


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
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    # Mesure du temps d'entraînement
    with TRAIN_DURATION.time():
        model.fit(trainset)

    TRAIN_ITERATIONS.inc(n_epochs)

    logger.info("Début de la cross-validation")
    # Mesure du temps de cross-validation
    with CROSS_VALIDATION_DURATION.time():
        cv_results = cross_validate(
            model, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True
        )

    # Mise à jour des métriques RMSE et MAE
    mean_rmse = cv_results["test_rmse"].mean()
    mean_mae = cv_results["test_mae"].mean()
    RMSE_GAUGE.set(mean_rmse)
    MAE_GAUGE.set(mean_mae)

    logger.info("Moyenne des RMSE : %s", mean_rmse)
    logger.info("Moyenne des MAE : %s", mean_mae)

    # Sauvegarde des modèles et du reader
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        logger.info(f"Created directory: {data_directory}")

    with open(f"{data_directory}/model_svd.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{data_directory}/reader.pkl", "wb") as f:
        pickle.dump(reader, f)
    logger.info("Surprise SVD model trained and logged successfully.")

    return model, reader


if __name__ == "__main__":
    logger.info("########## TRAIN MODELS ##########")
    start_http_server(8000)
    data_directory = "/root/mount_file/models"
    ratings = fetch_table("ratings")
    movies = fetch_table("movies")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    logger.info("Entrainement du modèle Surprise SVD")
    train_model(df, data_directory)
