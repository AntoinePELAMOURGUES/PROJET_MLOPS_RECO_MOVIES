import os
import pandas as pd
import pickle
from scipy.sparse import csr_matrix, lil_matrix, save_npz
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# import mlflow

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

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


def filterred_data(df):
    """
    Filtrer les données pour ne conserver que les films ayant reçu au moins 5 évaluations
    et les utilsateurs ayant évalués au moins 10 films.
    """
    user_counts = df["userid"].value_counts()
    users_with_more_than_10_ratings = user_counts[user_counts > 10].index

    # Étape 2 : Compter le nombre de notes par film
    movie_counts = df["movieid"].value_counts()
    movies_with_at_least_5_ratings = movie_counts[movie_counts >= 5].index

    # Étape 3 : Filtrer le DataFrame
    df = df[
        (df["userid"].isin(users_with_more_than_10_ratings))
        & (df["movieid"].isin(movies_with_at_least_5_ratings))
    ]

    return df

    # def authenticate_mlflow():
    """Authentifie MLflow en utilisant les variables d'environnement."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Movie Recommendation Models")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_username and mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password


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

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement de la similarité cosinus avec scipy.sparse
    np.save(f"{data_directory}/cosine_similarity_tfidf.npy", sim_cosinus)
    print(
        f"Matrice de similarité cosinus enregistrée sous {data_directory}/cosine_similarity_tfidf.npy."
    )

    # Enregistrer le modèle avec MLflow
    # mlflow.sklearn.log_model(tfidf, "tfidf_model")

    end_time = datetime.now()

    duration = end_time - start_time

    print(f"Durée de l'entraînement : {duration}")
    # mlflow.log_param("training_duration", duration.total_seconds())

    # mlflow.end_run()  # Finir l'exécution de l'expérience MLflow


def train_matrix_factorization_model(df, data_directory):
    """
    Entraîne un modèle de factorisation matricielle pour prédire les évaluations des utilisateurs.
    """
    # Démarrer une nouvelle expérience MLflow
    # mlflow.start_run()

    start_time = datetime.now()  # Démarrer la mesure du temps

    # df = df.sample(frac=0.08, random_state=42).reset_index(drop=True)
    mat_ratings = pd.pivot_table(
        data=df, values="rating", columns="title", index="userid"
    )
    mat_ratings = (
        mat_ratings + 1
    )  # On ajoute 1 à toutes les notes pour éviter les problèmes de division par 0
    mat_ratings = mat_ratings.fillna(0)
    sparse_ratings = csr_matrix(mat_ratings)
    user_ids = mat_ratings.index.tolist()
    titles = mat_ratings.columns.tolist()
    # Appliquer la factorisation matricielle
    svd = TruncatedSVD(n_components=100)
    ratings_red = svd.fit_transform(sparse_ratings.T)
    item_similarity = cosine_similarity(ratings_red)
    item_similarity = pd.DataFrame(item_similarity, index=titles, columns=titles)

    # Enregistrer le modèle avec NumPy
    np.save(f"{data_directory}/item_similarity.npy", item_similarity)
    print(
        f"Matrice de similarité enregistrée avec NumPy sous {data_directory}/item_similarity.npy."
    )

    # Enregistrer le modèle avec MLflow
    # mlflow.sklearn.log_model(svd, "matrix_factorization_model")

    end_time = datetime.now()

    duration = end_time - start_time
    # mlflow.log_param("training_duration", duration.total_seconds())

    # mlflow.end_run()  # Finir l'exécution de l'expérience MLflow
    print(f"Durée de l'entraînement : {duration}")


if __name__ == "__main__":
    print("########## TRAIN MODELS ##########")
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/silver")
    data_directory = os.path.join(my_project_directory, "data/models")
    # authenticate_mlflow()
    movies = load_data(raw_data_relative_path, "processed_movies.csv")
    ratings = load_data(raw_data_relative_path, "processed_ratings.csv")
    df = pd.merge(ratings, movies, on="movieid", how="left")
    df = filterred_data(df)
    df = df.sample(frac=0.02, random_state=42).reset_index(drop=True)
    print("Entrainement du modèle TF-IDF")
    train_TFIDF_model(movies, data_directory)
    print("Entrainement du modèle de factorisation matricielle")
    train_matrix_factorization_model(df, data_directory)
