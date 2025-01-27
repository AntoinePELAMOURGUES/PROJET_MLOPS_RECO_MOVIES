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
from pathlib import Path


def get_most_recent_files(directory, file_prefixes):
    """Récupère les fichiers avec les préfixes spécifiés et retourne le plus récent pour chacun."""
    path = Path(directory)
    recent_files = {
        prefix: None for prefix in file_prefixes
    }  # Dictionnaire pour stocker les fichiers récents
    recent_times = {
        prefix: 0 for prefix in file_prefixes
    }  # Dictionnaire pour stocker les timestamps

    # Parcourir tous les fichiers dans le répertoire
    for file in path.glob("*.csv"):  # Vous pouvez changer l'extension si nécessaire
        for prefix in file_prefixes:
            if file.stem.startswith(
                prefix
            ):  # Vérifie si le nom du fichier commence par le préfixe
                file_time = os.path.getmtime(file)
                if (
                    file_time > recent_times[prefix]
                ):  # Si c'est plus récent que ce qu'on a enregistré
                    recent_times[prefix] = file_time
                    recent_files[prefix] = file

    return recent_files


if __name__ == "__main__":
    directory = "/root/mountfile/raw/silver"  # Remplacez par votre répertoire
    prefixes = [
        "processed_movies",
        "processed_ratings",
    ]  # Préfixes des fichiers à rechercher
    recent_files = get_most_recent_files(directory, prefixes)

    for prefix in prefixes:
        if recent_files[prefix]:
            print(
                f"Le fichier le plus récent pour '{prefix}' est : {recent_files[prefix]}"
            )
        else:
            print(f"Aucun fichier trouvé pour '{prefix}'.")


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
        movies["genres"] = movies["genres"].str.replace(" ", "")
        # Nettoyer les genres en supprimant les espaces au début et à la fin
        movies["genres"] = movies["genres"].str.strip()

        # Créer des variables indicatrices pour les genres
        genres = movies["genres"].str.get_dummies(sep=",")

        # Calculer la similarité cosinus
        cosine_sim = cosine_similarity(genres, genres)

        end_time = datetime.now()

        duration = end_time - start_time
        print(f"Durée de l'entraînement : {duration}")
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
    data_directory = "/root/mount_file/models"
    authenticate_mlflow()
    ratings = fetch_ratings("ratings")
    movies = fetch_movies("movies")
    print("Entrainement du modèle SVD")
    train_SVD_model(ratings, data_directory)
    print("Création de notre matrice cosinus")
    train_cosine_similarity(movies, data_directory)
