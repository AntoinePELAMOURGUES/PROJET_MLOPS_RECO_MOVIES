import os
import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pickle
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from dotenv import load_dotenv

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
        return df_ratings
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


def create_X(df):
    """Crée une matrice creuse et les dictionnaires de correspondance.

    Args:
        df (pd.DataFrame): DataFrame avec colonnes userId, movieId, rating.

    Returns:
        tuple: (matrice_creuse, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper)
    """
    M = df["userid"].nunique()
    N = df["movieid"].nunique()

    user_mapper = dict(zip(np.unique(df["userid"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieid"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userid"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieid"])))

    user_index = [user_mapper[i] for i in df["userid"]]
    item_index = [movie_mapper[i] for i in df["movieid"]]

    X = csr_matrix((df["rating"], (user_index, item_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def train_matrix_model(df, data_directory, k=10, metric="cosine"):
    """Entraîne et sauvegarde un modèle KNN basé sur une matrice creuse.

    Args:
        df (pd.DataFrame): DataFrame avec les données d'évaluation.
        k (int): Nombre de voisins à considérer.
        metric (str): Métrique de distance pour KNN.
    """

    # Démarrer la mesure du temps
    start_time = datetime.now()
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(df)
    # Transposer la matrice X pour que les films soient en lignes et les utilisateurs en colonnes
    X = X.T
    # Initialiser NearestNeighbors avec k+1 car nous voulons inclure le film lui-même dans les voisins
    kNN = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)

    kNN.fit(X)

    end_time = datetime.now()

    duration = end_time - start_time
    print(f"Durée de l'entraînement : {duration}")

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement du modèle avec pickle
    with open(f"{data_directory}/model_KNN.pkl", "wb") as f:
        pickle.dump(kNN, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_KNN.pkl.")


if __name__ == "__main__":
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/silver")
    data_directory = os.path.join(my_project_directory, "data/models")
    ratings = load_data(raw_data_relative_path)
    print("Entrainement du modèle SVD")
    train_SVD_model(ratings, data_directory)
    print("Entrainement du modèle CSR Matrix")
    train_matrix_model(ratings, data_directory)
