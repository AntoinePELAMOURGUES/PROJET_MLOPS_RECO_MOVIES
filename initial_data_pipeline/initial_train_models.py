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

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

my_project_directory = os.getenv("MY_PROJECT_DIRECTORY")
print(f"my_project_directory: {my_project_directory}")


def load_data(raw_data_relative_path):
    """
    Charge les données des fichiers CSV dans des DataFrames pandas.

    Args:
        raw_data_relative_path (str): Chemin vers le répertoire contenant les fichiers CSV.

    Returns:
        tuple: DataFrames pour les évaluations, les films et les liens.
    """
    try:
        df_ratings = pd.read_csv(
            f"{raw_data_relative_path}/processed_ratings.csv",
            usecols=["userid", "movieid", "rating"],  # Sélectionner les colonnes
            dtype={"rating": "float32"},  # Convertir 'rating' en float32)
        )
        df_ratings = df_ratings.sample(frac=0.8)  # Utiliser 80 % des donnnées
        print("Fichier processed_ratings chargé avec succès.")
        df_movies = pd.read_csv(f"{raw_data_relative_path}/processed_movies.csv")
        print("Fichier processed_movies chargé avec succès.")
        return df_ratings, df_movies
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"No data: {e}")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")


def train_SVD_model(df, data_directory) -> tuple:
    """Entraîne un modèle SVD de recommandation et sauvegarde le modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes userId, movieId et rating.
    """

    start_time = datetime.now()  # Démarrer la mesure du temps

    # Préparer les données pour Surprise
    reader = Reader(rating_scale=(0.5, 5))
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
    with open(f"{data_directory}/model_SVD.pkl", "wb") as f:
        pickle.dump(model, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_SVD.pkl.")

    end_time = datetime.now()

    duration = end_time - start_time
    print(f"Durée de l'entraînement : {duration}")


def train_cosine_similarity(movies, data_directory):
    """
    Calcule la similarité cosinus entre les films en utilisant les genres des films dans le cadre d'un démarrage à froid.
    """
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

        # Afficher la matrice des genres
        print("\nMatrice des genres :")
        print(genres)

        # Calculer la similarité cosinus
        cosine_sim = cosine_similarity(genres, genres)
        print(f"Dimensions de notre matrice de similarité cosinus : {cosine_sim.shape}")
    else:
        print("La colonne 'genres' n'existe pas dans le DataFrame.")

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement du modèle avec NumPy
    np.save(f"{data_directory}/cosine_similarity_matrix.npy", cosine_sim)
    print(
        f"Matrice de similarité enregistrée avec NumPy sous {data_directory}/cosine_similarity_matrix.npy."
    )


if __name__ == "__main__":
    print("########## TRAIN MODELS ##########")
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/silver")
    data_directory = os.path.join(my_project_directory, "data/models")
    movies = pd.read_csv(f"{raw_data_relative_path}/processed_movies.csv")
    ratings, movies = load_data(raw_data_relative_path)
    print("Entrainement du modèle SVD")
    train_SVD_model(ratings, data_directory)
    print("Création de la matrice de similarité cosinus en fonction des genres")
    train_cosine_similarity(movies, data_directory)
