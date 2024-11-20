import pandas as pd
import os
import json
import pickle
from surprise.prediction_algorithms.matrix_factorization import SVD
from scipy.sparse import csr_matrix
import numpy as np
from rapidfuzz import process
from fastapi import APIRouter, HTTPException
from database import get_db_connection
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel
from scipy.sparse import csr_matrix
import psycopg2
from dotenv import load_dotenv
import requests
import logging

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

tmdb_token = os.getenv("TMDB_API_TOKEN")

# Configuration du logger pour afficher les informations de débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ROUTEUR POUR GERER LES ROUTES PREDICT

router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)


# ENSEMBLE DES FONCTIONS UTILISEES


# Chargement des datasets
def read_ratings(ratings_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie ratings."""
    data = pd.read_csv(os.path.join(data_dir, ratings_csv))
    print("Dataset ratings loaded")
    return data



def read_movies(movies_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie information."""
    df = pd.read_csv(os.path.join(data_dir, movies_csv))
    print("Dataset movies loaded")
    return df

def read_links(links_csv: str, data_dir: str = "/app/raw") -> pd.DataFrame:
    """Reads the CSV file containing movie information."""
    df = pd.read_csv(os.path.join(data_dir, links_csv))
    print("Dataset links loaded")
    return df


# Chargement du dernier modèle
def load_model(pkl_files, directory = "/app/model") :
    """Charge le modèle à partir d'un répertoire."""
    # Vérifier si le répertoire existe
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
    # Charger le modèle
    filepath = os.path.join(directory, pkl_files)
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {filepath}')
    return model

def create_X(df):
    """
    Génère une matrice creuse avec quatre dictionnaires de mappage
    - user_mapper: mappe l'ID utilisateur à l'index utilisateur
    - movie_mapper: mappe l'ID du film à l'index du film
    - user_inv_mapper: mappe l'index utilisateur à l'ID utilisateur
    - movie_inv_mapper: mappe l'index du film à l'ID du film
    Args:
        df: pandas dataframe contenant 3 colonnes (userId, movieId, rating)

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    M = df['userId'].nunique()
    N = df['movieId'].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    item_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (user_index,item_index)), shape=(M,N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Fonction pour obtenir des recommandations pour un utilisateur donné
def get_user_recommendations(user_id: int, model: SVD, ratings_df: pd.DataFrame, n_recommendations: int = 10):
    """Obtenir des recommandations pour un utilisateur donné."""
    # Créer un DataFrame contenant tous les films
    all_movies = ratings_df['movieId'].unique()

    # Obtenir les films déjà évalués par l'utilisateur
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()

    # Trouver les films non évalués par l'utilisateur
    unseen_movies = [movie for movie in all_movies if movie not in rated_movies]

    # Préparer les prédictions pour les films non évalués
    predictions = []
    for movie_id in unseen_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))  # Ajouter l'ID du film et la note prédite

    # Trier les prédictions par note prédite (descendant) et prendre les meilleures n_recommendations
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
    top_n = [i[0] for i in top_n]

    return top_n  # Retourner les meilleures recommandations


def get_movie_title_recommendations(model, movie_id, X, movie_mapper, movie_inv_mapper, k):
    """
    Trouve les k voisins les plus proches pour un ID de film donné.

    Args:
        movie_id: ID du film d'intérêt
        X: matrice d'utilité utilisateur-article (matrice creuse)
        k: nombre de films similaires à récupérer
        metric: métrique de distance pour les calculs kNN

    Output: retourne une liste des k ID de films similaires
    """
    # Transposer la matrice X pour que les films soient en lignes et les utilisateurs en colonnes
    X = X.T

    neighbour_ids = []  # Liste pour stocker les ID des films similaires

    # Obtenir l'index du film à partir du mapper
    movie_ind = movie_mapper[movie_id]

    # Extraire le vecteur correspondant au film spécifié
    movie_vec = X[movie_ind]

    # Vérifier si movie_vec est un tableau NumPy et le remodeler en 2D si nécessaire
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)  # Reshape pour avoir une forme (1, n_features)

    # Trouver les k+1 voisins les plus proches (y compris le film d'intérêt)
    neighbour = model.kneighbors(movie_vec, return_distance=False)

    # Collecter les ID des films parmi les voisins trouvés
    for i in range(0, k):  # Boucler jusqu'à k pour obtenir seulement les films similaires
        n = neighbour.item(i)  # Obtenir l'index du voisin
        neighbour_ids.append(movie_inv_mapper[n])  # Mapper l'index à l'ID du film

    neighbour_ids.pop(0)  # Retirer le premier élément qui est l'ID du film original

    return neighbour_ids  # Retourner la liste des ID de films similaires

def format_movie_id(movie_id):
    """Formate l'ID du film pour qu'il ait 7 chiffres."""
    return str(movie_id).zfill(7)

def api_tmdb_request(movie_ids):
    """Effectue des requêtes à l'API TMDB pour récupérer les informations des films."""
    results = {}

    for index, movie_id in enumerate(movie_ids):
        formatted_id = format_movie_id(movie_id)
        url = f"https://api.themoviedb.org/3/find/tt{formatted_id}?external_source=imdb_id"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {tmdb_token}"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data["movie_results"]:
                # On suppose que nous voulons le premier résultat
                movie_info = data["movie_results"][0]
                results[str(index)] = {
                    "title": movie_info["title"],
                    "vote_average": movie_info['vote_average'],
                    "poster_path": f"http://image.tmdb.org/t/p/w185{movie_info['poster_path']}"
                }
            else:
                results[str(index)] = {"error": "No movie results found"}
        else:
            results[str(index)] = {"error": f"Request failed with status code {response.status_code}"}

    return results

# Recherche un titre proche de la requete
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

# ---------------------------------------------------------------

# METRICS PROMETHEUS

collector = CollectorRegistry()
# Nbre de requête
nb_of_requests_counter = Counter(
    name='predict_nb_of_requests',
    documentation='number of requests per method or per endpoint',
    labelnames=['method', 'endpoint'],
    registry=collector)
# codes de statut des réponses
status_code_counter = Counter(
    name='predict_response_status_codes',
    documentation='Number of HTTP responses by status code',
    labelnames=['status_code'],
    registry=collector)
# Taille des réponses
response_size_histogram = Histogram(
    name='http_response_size_bytes',
    documentation='Size of HTTP responses in bytes',
    labelnames=['method', 'endpoint'],
    registry=collector)
# Temps de traitement par utilisateur
duration_of_requests_histogram = Histogram(
    name='duration_of_requests',
    documentation='Duration of requests per method or endpoint',
    labelnames=['method', 'endpoint', 'user_id'],
    registry=collector)
# Erreurs spécifiques
error_counter = Counter(
    name='api_errors',
    documentation='Count of API errors by type',
    labelnames=['error_type'],
    registry=collector)

# ---------------------------------------------------------------

# CHARGEMENT DES DONNEES AU DEMARRAGE DE API
print("DEBUT DES CHARGEMENTS")
# Chargement de nos dataframe depuis mongo_db
ratings = read_ratings('processed_ratings.csv')
movies = read_movies('processed_movies.csv')
links = read_links('processed_links.csv')
# Chargement d'un modèle SVD pré-entraîné pour les recommandations
model_svd = load_model('model_SVD.pkl')
# Chargement de la matrice cosinus similarity
model_Knn = load_model('model_KNN.pkl')
# Création de la matrice utilisateur-article
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(ratings)
# Création d'un dataframe pour les liens entre les films et les ID IMDB
movies_links_df = movies.merge(links, on = "movieId", how = 'left')
# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_idx = dict(zip(movies['title'], list(movies.index)))
# Création de dictionnaires pour accéder facilement aux titres et aux couvertures des films par leur ID
movie_titles = dict(zip(movies['movieId'], movies['title']))
# Créer un dictionnaire pour un accès rapide
imdb_dict = dict(zip(movies_links_df['movieId'], movies_links_df['imdbId']))
# test de connection à la base de données
with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                tables = cur.fetchall()
                # Imprimer les noms des tables
                print("Tables présentes dans la base de données :")
                for table in tables:
                    print(table[0])
print("FIN DES CHARGEMENTS")
# ---------------------------------------------------------------

# REQUETES API

# Modèle Pydantic pour la récupération de l'user_id lié aux films
class UserRequest(BaseModel):
    userId: Optional[int]  # Forcer le type int explicitement
    movie_title: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "userId": 1,
                "movie_title": "Inception"
            }
        }

@router.post("/best_user_movies")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """
    Route API pour récupérer les 3 films les mieux notés de l'utilisateur.
    Args: user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur, y compris l'ID utilisateur et le titre du film.
    Returns:
        Dict[str, Any]: Un dictionnaire contenant le choix de l'utilisateur et les recommandations de films.
    """
    logger.info(f"Requête reçue pour l'utilisateur identifié: {user_request}")
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/predict/best_user_movies').inc()
    # Récupération de l'email utilisateur de la session
    userId = user_request.userId    # récupéartion de l'userId dans la base de données
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    try:
        # Forcer la conversion en int
        user_id = int(user_request.userId)
        df_user = ratings[ratings['userId'] == user_id]
        df_user = df_user.sort_values(by='rating', ascending=False)
        best_movies = df_user.head(3)
        imdb_list = [imdb_dict[movie_id] for movie_id in best_movies['movieId'] if movie_id in imdb_dict]
        results = api_tmdb_request(imdb_list)
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(method='POST', endpoint='/predict/best_user_movies', user_id=str(user_id)).observe(duration)  # Enregistrer la durée de la requête
        response_size_histogram.labels(method='POST', endpoint='/predict/best_user_movies').observe(response_size)  # Enregistrer la taille de la réponse
        # Utiliser le logger pour voir les résultats
        logger.info(f"Api response: {results}")
        return results
    except ValueError as e:
        status_code_counter.labels(status_code="400").inc()  # Compter les réponses échouées
        error_counter.labels(error_type="ValueError").inc()  # Enregistrer l'erreur spécifique
        raise HTTPException(status_code=400, detail="L'ID utilisateur doit être un nombre entier")



# Route API concernant les utilisateurs déjà identifiés avec titre de films
@router.post("/identified_user")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.
    Args: user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur, y compris l'ID utilisateur et le titre du film.
    Returns:
        Dict[str, Any]: Un dictionnaire contenant le choix de l'utilisateur et les recommandations de films.
    """
    logger.info(f"Requête reçue pour l'utilisateur identifié: {user_request}")
    # Debug du type et de la valeur de userId
    logger.info(f"Type de userId reçu: {type(user_request.userId)}")
    logger.info(f"Valeur de userId reçue: {user_request.userId}")
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/predict/identified_user').inc()
    # Récupération de l'email utilisateur de la session
    userId = user_request.userId    # récupéartion de l'userId dans la base de données
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    try:
        # Forcer la conversion en int
        user_id = int(user_request.userId)
        recommendations = get_user_recommendations(user_id, model_svd, ratings, n_recommendations = 8)
        logger.info(f"Recommandations pour l'utilisateur {userId}: {recommendations}")
        imdb_list = [imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict]
        results = api_tmdb_request(imdb_list)


        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(method='POST', endpoint='/predict/identified_user', user_id=str(user_id)).observe(duration)  # Enregistrer la durée de la requête
        response_size_histogram.labels(method='POST', endpoint='/predict/identified_user').observe(response_size)  # Enregistrer la taille de la réponse
        # Utiliser le logger pour voir les résultats
        logger.info(f"Api response: {results}")
        return results

    except ValueError as e:
        status_code_counter.labels(status_code="400").inc()  # Compter les réponses échouées
        error_counter.labels(error_type="ValueError").inc()  # Enregistrer l'erreur spécifique
        raise HTTPException(status_code=400, detail="L'ID utilisateur doit être un nombre entier")


# Route Api recommandation par rapport à un autre film
@router.post("/similar_movies")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """
    Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.
    Args: user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur, y compris l'ID utilisateur et le titre du film.
    Returns:
        Dict[str, Any]: Un dictionnaire contenant le choix de l'utilisateur et les recommandations de films.
    """
    logger.info(f"Requête reçue pour similar_movies: {user_request}")
    # Démarrer le chronomètre pour mesurer la durée de la requête
    start_time = time.time()
    # Incrémenter le compteur de requêtes pour prometheus
    nb_of_requests_counter.labels(method='POST', endpoint='/predict/similar_movies').inc()
    # Récupération des données Streamlit
    movie_title = movie_finder(user_request.movie_title)  # Trouver le titre du film correspondant
    movie_id = int(movies['movieId'][movies['title'] == movie_title].iloc[0])
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    recommendations = get_movie_title_recommendations(model_Knn, movie_id, X, movie_mapper, movie_inv_mapper, 9)
    imdb_list = [imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict]
    results = api_tmdb_request(imdb_list)

    # Mesurer la taille de la réponse et l'enregistrer
    response_size = len(json.dumps(results))
    # Calculer la durée et enregistrer dans l'histogramme
    duration = time.time() - start_time
    # Enregistrement des métriques pour Prometheus
    status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
    duration_of_requests_histogram.labels(method='POST', endpoint='/predict/similar_movies', user_id="N/A").observe(duration)  # Enregistrer la durée de la requête
    response_size_histogram.labels(method='POST', endpoint='/predict/similar_movies').observe(response_size)  # Enregistrer la taille de la réponse
    logger.info(f"Api response: {results}")
    response_size_histogram.labels(method='POST', endpoint='/identified_user').observe(response_size)  # Enregistrer la taille de la réponse
    return results

