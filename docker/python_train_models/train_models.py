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
import psycopg2
import mlflow



def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        'host': os.getenv('POSTGRES_HOST'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

def connect(config):
    """Connecte au serveur PostgreSQL et retourne la connexion."""
    try:
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Connection error: {error}")
        return None


def fetch_ratings(table):
    """Récupère les données de la table ratings et retourne un DataFrame."""
    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            # Exécutez une requête SQL pour récupérer les données de la table ratings
            query = f"SELECT userid, movieid, rating FROM {table};"
            df = pd.read_sql_query(query, conn)
            print("Data fetched successfully.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        finally:
            conn.close()  # Assurez-vous de fermer la connexion
    else:
        print("Failed to connect to the database.")
        return None


def train_SVD_model(df, data_directory) -> tuple:
    """Entraîne un modèle SVD de recommandation et sauvegarde le modèle.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes userId, movieId et rating.
    """


    # Démarrer une nouvelle expérience MLflow
    mlflow.start_run()

    start_time = datetime.now()  # Démarrer la mesure du temps

    # Préparer les données pour Surprise
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userid', 'movieid', 'rating']], reader=reader)

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
    with open(f'{data_directory}/model_SVD.pkl', 'wb') as f:
        pickle.dump(model, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_SVD.pkl.")

    # Enregistrer les métriques dans MLflow
    mlflow.log_metric("RMSE", acc_rounded)

    # Enregistrer le modèle avec MLflow
    mlflow.sklearn.log_model(model, "model_SVD")

    end_time = datetime.now()

    duration = end_time - start_time
    print(f"Durée de l'entraînement : {duration}")

    # Finir l'exécution de l'expérience MLflow
    mlflow.end_run()



def create_X(df):
    """Crée une matrice creuse et les dictionnaires de correspondance.

    Args:
        df (pd.DataFrame): DataFrame avec colonnes userId, movieId, rating.

    Returns:
        tuple: (matrice_creuse, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper)
    """
    M = df['userid'].nunique()
    N = df['movieid'].nunique()

    user_mapper = dict(zip(np.unique(df["userid"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieid"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userid"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieid"])))

    user_index = [user_mapper[i] for i in df['userid']]
    item_index = [movie_mapper[i] for i in df['movieid']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper


def train_matrix_model(df, data_directory, k = 10, metric='cosine'):
    """Entraîne et sauvegarde un modèle KNN basé sur une matrice creuse.

    Args:
        df (pd.DataFrame): DataFrame avec les données d'évaluation.
        k (int): Nombre de voisins à considérer.
        metric (str): Métrique de distance pour KNN.
    """
    # Démarrer une nouvelle expérience MLflow
    mlflow.start_run()
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
    print(f'Durée de l\'entraînement : {duration}')

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    # Enregistrement du modèle avec pickle
    with open(f'{data_directory}/model_KNN.pkl', 'wb') as f:
        pickle.dump(kNN, f)
        print(f"Modèle SVD enregistré avec pickle sous {data_directory}/model_KNN.pkl.")

    # Enregistrer les informations du modèle dans MLflow (par exemple la durée d'entraînement)
    mlflow.log_param("k_neighbors", k)
    mlflow.log_param("metric", metric)
    mlflow.log_param("training_duration", duration.total_seconds())
    # Enregistrer le modèle avec MLflow
    mlflow.sklearn.log_model(kNN, "model_KNN")

    mlflow.end_run()  # Finir l'exécution de l'expérience MLflow

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
    data_directory = '/root/mount_file/'
    authenticate_mlflow()
    ratings = fetch_ratings('ratings')
    print('Entrainement du modèle SVD')
    train_SVD_model(ratings, data_directory)
    print('Entrainement du modèle CSR Matrix')
    train_matrix_model(ratings, data_directory)