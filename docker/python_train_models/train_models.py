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

# Création d'un registre de collecteurs Prometheus
collector = CollectorRegistry()

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
        logger.info("Connecté au serveur PostgreSQL.")
        return engine
    except Exception as error:
        logger.error(f"Erreur de connexion : {error}")
        return None

def fetch_table(table):
    """Récupère les lignes d'une table et retourne un DataFrame."""
    config = load_config()
    engine = connect(config)

    if engine is not None:
        try:
            query = f"SELECT * FROM {table};"
            df = pd.read_sql_query(query, engine)
            logger.info(f"Données de {table} récupérées avec succès.")
            return df
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données : {e}")
            return None
        finally:
            engine.dispose()  # Libère proprement l'engine
    else:
        logger.error("Échec de la connexion à la base de données.")
        return None

# MÉTRIQUES PROMETHEUS
# Ajout d'étiquettes pour différencier les modèles et les versions
train_duration_histogram = Histogram('model_training_duration_seconds',
                                     'Durée de l\'entraînement du modèle en secondes',
                                      labelnames=['model_type', 'version'],
                                      registry = collector)

cross_validation_duration_histogram = Histogram('cross_validation_duration_seconds',
                                                'Durée de la validation croisée en secondes',
                                               labelnames=['model_type', 'version'],
                                               registry = collector)

rmse_gauge = Gauge('model_rmse',
                   'Erreur quadratique moyenne du modèle',
                   labelnames=['model_type', 'version'],
                   registry = collector)

mae_gauge = Gauge('model_mae',
                  'Erreur absolue moyenne du modèle',
                  labelnames=['model_type', 'version'],
                  registry = collector)

train_iterations_counter = Counter('model_train_iterations_total',
                                   'Nombre total d\'itérations d\'entraînement',
                                    labelnames=['model_type', 'version'],
                                    registry = collector)

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
    with train_duration_histogram.labels(model_type='SVD', version='1.0').time():
        model.fit(trainset)

    # Incrémentation du compteur d'itérations
    train_iterations_counter.labels(model_type='SVD', version='1.0').inc(n_epochs)

    logger.info("Début de la validation croisée")
    # Mesure du temps de validation croisée
    with cross_validation_duration_histogram.labels(model_type='SVD', version='1.0').time():
        cv_results = cross_validate(
            model, data, measures=["RMSE", "MAE"], cv=5, return_train_measures=True
        )

    # Mise à jour des métriques RMSE et MAE
    mean_rmse = cv_results["test_rmse"].mean()
    mean_mae = cv_results["test_mae"].mean()
    rmse_gauge.labels(model_type='SVD', version='1.0').set(mean_rmse)
    mae_gauge.labels(model_type='SVD', version='1.0').set(mean_mae)

    logger.info("Moyenne des RMSE : %s", mean_rmse)
    logger.info("Moyenne des MAE : %s", mean_mae)

    # Sauvegarde des modèles et du reader
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        logger.info(f"Répertoire créé : {data_directory}")

    with open(f"{data_directory}/model_svd.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{data_directory}/reader.pkl", "wb") as f:
        pickle.dump(reader, f)
    logger.info("Modèle Surprise SVD entraîné et enregistré avec succès.")

    return model, reader

if __name__ == "__main__":
    logger.info("########## ENTRAÎNEMENT DES MODÈLES ##########")
    start_http_server(8000)  # Démarrage du serveur HTTP pour Prometheus
    data_directory = "/root/mount_file/models"
    ratings = fetch_table("ratings")
    movies = fetch_table("movies")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    logger.info("Entraînement du modèle Surprise SVD")
    train_model(df, data_directory)
