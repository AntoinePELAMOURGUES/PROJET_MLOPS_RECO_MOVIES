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
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import mlflow

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
                usecols=["userId", "movieId", "rating"],  # Sélectionner les colonnes
                dtype={"rating": "float32", "userId": str, "movieId": str},
            )
            return df
        elif "movies" in filename:
            df = pd.read_csv(
                f"{raw_data_relative_path}/{filename}",
                usecols=["movieId", "title", "genres"],  # Sélectionner les colonnes
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
    Filtrer les données pour ne conserver que les films ayant reçu au moins 10 évaluations
    et les utilsateurs ayant évalués au moin 10 films.
    """
    user_counts = df["userid"].value_counts()
    users_with_more_than_10_ratings = user_counts[user_counts > 10].index

    # Étape 2 : Compter le nombre de notes par film
    movie_counts = df["movieid"].value_counts()
    movies_with_at_least_2_ratings = movie_counts[movie_counts >= 10].index

    # Étape 3 : Filtrer le DataFrame
    df = df[
        (df["userid"].isin(users_with_more_than_10_ratings))
        & (df["movieid"].isin(movies_with_at_least_2_ratings))
    ]

    return df


def authenticate_mlflow():
    """Authentifie MLflow en utilisant les variables d'environnement."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Movie Recommendation Models")
    mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_username and mlflow_password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password


def train_SVD_model(df, data_directory) -> tuple:
    """Entraîne un modèle SVD de recommandation et sauvegarde le modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes userId, movieId et rating.
    """

    # Démarrer une nouvelle expérience MLflow
    mlflow.start_run()

    start_time = datetime.now()  # Démarrer la mesure du temps

    # Préparer les données pour Surprise
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)

    # Vérifier que le DataFrame n'est pas vide
    if len(df) == 0:
        raise ValueError(
            "Le DataFrame est vide et ne peut pas être utilisé pour l'entraînement."
        )

    # Diviser les données en ensembles d'entraînement et de test
    trainset, testset = train_test_split(data, test_size=0.25)

    # Créer et entraîner le modèle SVD
    model = SVD(n_factors=150, n_epochs=30, lr_all=0.01, reg_all=0.05)
    model.fit(trainset)

    # Tester le modèle sur l'ensemble de test et calculer RMSE
    predictions = model.test(testset)
    acc = accuracy.rmse(predictions)

    # Arrondir à 2 chiffres après la virgule
    acc_rounded = round(acc, 2)

    print("Valeur de l'écart quadratique moyen (RMSE) :", acc_rounded)

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement du modèle avec pickle
    with open(f"{data_directory}/model_svd.pkl", "wb") as f:
        pickle.dump(model, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_svd.pkl.")

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
    print("Colonnes du DataFrame :", movies.columns)
    print("Aperçu du DataFrame :")
    print(movies.head())
    start_time = datetime.now()  # Démarrer la mesure du temps
    # Supprimer les espaces dans les genres
    if "genres" in movies.columns:
        # Supprimer les espaces autour des virgules
        movies["genres"] = movies["genres"].str.replace(
            " ", ""
        )  # Cela supprime tous les espaces
        # Nettoyer les genres en supprimant les espaces au début et à la fin
        movies["genres"] = movies["genres"].str.strip()

        # Créer des variables indicatrices pour les genres
        genres = movies["genres"].str.get_dummies(sep=",")

        genres = genres.drop(columns=["(nogenreslisted)"])

        # Afficher la matrice des genres
        print("\nMatrice des genres :")

        print(genres)

        # Calculer la similarité cosinus
        cosine_sim = cosine_similarity(genres, genres)
        print(f"Dimensions de notre matrice de similarité cosinus : {cosine_sim.shape}")
        end_time = datetime.now()

        duration = end_time - start_time
    else:
        print("La colonne 'genres' n'existe pas dans le DataFrame.")

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


if __name__ == "__main__":
    print("########## TRAIN MODELS ##########")
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/silver")
    data_directory = os.path.join(my_project_directory, "data/models")
    authenticate_mlflow()
    movies = load_data(raw_data_relative_path, "preprocess_movies.csv")
    ratings = load_data(raw_data_relative_path, "preprocess_ratings.csv")
    df = pd.merge(ratings, movies, on="movieid")
    df = filterred_data(df)
    print("Entrainement du modèle SVD")
    train_SVD_model(df, data_directory)
    print("Création de la matrice de similarité cosinus en fonction des genres")
    train_cosine_similarity(movies, data_directory)
