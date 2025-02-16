import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
import pickle
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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


########" MODELE DE RECOMMANDATION DE FILMS - SANS USERID ########"
def train_TFIDF_model(df, data_directory):
    """
    Entraîne un modèle TF-IDF pour extraire des caractéristiques des genres de films,
    et enregistre le modèle directement sur MLflow.
    """
    start_time = datetime.now()  # Démarrer la mesure du temps

    # Vérifier les colonnes et le contenu
    logger.info("Colonnes du DataFrame : %s", df.columns)

    # Créer une instance de TfidfVectorizer
    tfidf = TfidfVectorizer()

    # Calculer la matrice TF-IDF
    tfidf_matrix = tfidf.fit_transform(df["genres"])

    # Afficher la taille de la matrice
    logger.info(f"Dimensions de notre matrice TF-IDF : {tfidf_matrix.shape}")

    # Calculer la similarité cosinus par morceaux
    sim_cosinus = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logger.info(f"Dimensions de la matrice de similarité cosinus : {sim_cosinus.shape}")
    indices = pd.Series(range(0, len(df)), index=df["title"])

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
    # Diviser les données en ensembles d'entraînement et de test
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)
    # Extraire le Trainset
    trainset = data.build_full_trainset()
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
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
    # Sauvegadrer modèles et reader vers data_directory ("/root/mount_file/models/")
    with open(f"{data_directory}/model_svd.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{data_directory}/reader.pkl", "wb") as f:
        pickle.dump(reader, f)
    logger.info("Surprise SVD model trained and logged successfully.")

    return model, reader


if __name__ == "__main__":
    logger.info("########## TRAIN MODELS ##########")
    data_directory = "/root/mount_file/models/"
    ratings = fetch_table("ratings")
    movies = fetch_table("movies")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    logger.info("Entrainement du modèle TF-IDF")
    train_TFIDF_model(movies, data_directory)
    logger.info("Entrainement du modèle Surprise SVD")
    train_model(df, data_directory)
