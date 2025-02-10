import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import joblib
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate
import mlflow

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer le répertoire du projet
my_project_directory = os.getenv("MY_PROJECT_DIRECTORY")
print(f"my_project_directory: {my_project_directory}")


def load_data(raw_data_relative_path, filename):
    """
    Charge les données des fichiers CSV dans des DataFrames pandas.

    Args:
        raw_data_relative_path (str): Chemin vers le répertoire contenant les fichiers CSV.

    Returns:
        tuple: DataFrames pour les évaluations, les films et les liens.
    """
    try:
        if "ratings" in filename:
            df = pd.read_csv(
                f"{raw_data_relative_path}/{filename}",
                usecols=["userid", "movieid", "rating"],  # Sélectionner les colonnes
                dtype={"rating": "float32", "userid": str, "movieid": str},
            )
            return df
        elif "movies" in filename:
            df = pd.read_csv(
                f"{raw_data_relative_path}/{filename}",
                usecols=["movieid", "title", "genres"],  # Sélectionner les colonnes
                dtype={"movieid": str, "title": str, "genres": str},
            )
            return df
        print(f"Fichier {filename} chargé avec succès.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"No data: {e}")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")

    # def authenticate_mlflow():
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
    # mlflow.start_run()

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

    # Calculer la similarité cosinus par morceaux
    sim_cosinus = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Dimensions de la matrice de similarité cosinus : {sim_cosinus.shape}")
    indices = pd.Series(range(0, len(df)), index=df["title"])
    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Sauvegarder les éléments essentiels
    joblib.dump(tfidf, os.path.join(data_directory, "tfidf_model.joblib"))
    joblib.dump(sim_cosinus, os.path.join(data_directory, "sim_cosinus.joblib"))
    indices.to_pickle(os.path.join(data_directory, "indices.pkl"))

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
    # Diviser les données en ensembles d'entraînement et de test
    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)

    # Extraire le Trainset
    trainset = data.build_full_trainset()

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

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

    # Sauvegarder le modèle
    os.makedirs(data_directory, exist_ok=True)
    # Sauvegarder le modèle et le contexte (reader)

    with open(os.path.join(data_directory, "svd_model_v1.pkl", "wb")) as file:
        pickle.dump({"model": model, "reader": reader}, file)

    return model, mean_rmse, reader


if __name__ == "__main__":
    print("########## :hammer: TRAIN MODELS ##########")
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/silver")
    data_directory = os.path.join(my_project_directory, "data/models")
    # authenticate_mlflow()
    movies = load_data(raw_data_relative_path, "processed_movies.csv")
    ratings = load_data(raw_data_relative_path, "processed_ratings.csv")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    print("Entrainement du modèle TF-IDF")
    train_TFIDF_model(movies, data_directory)
    print("Entrainement du modèle Surprise SVD")
    train_model(df, data_directory)
