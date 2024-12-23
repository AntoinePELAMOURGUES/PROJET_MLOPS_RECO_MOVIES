import os
import json
import pickle
from surprise.prediction_algorithms.matrix_factorization import SVD
from scipy.sparse import csr_matrix
import numpy as np
from rapidfuzz import process
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel
import psycopg2
from dotenv import load_dotenv
import requests
import logging
from kubernetes import client, config

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

tmdb_token = os.getenv("TMDB_API_TOKEN")

# Configuration du logger pour afficher les informations de débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration Kubernetes
config.load_incluster_config()

# Définir le volume et le montage du volume
volume = client.V1Volume(
    name="model-storage",
    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name="model-storage-pvc")
)

volume_mount = client.V1VolumeMount(
    name="model-storage",
    mount_path="/models"
)

# ROUTEUR POUR GERER LES ROUTES PREDICT

router = APIRouter(
    prefix='/predict',  # Préfixe pour toutes les routes dans ce routeur
    tags=['predict']    # Tag pour la documentation
)

# ENSEMBLE DES FONCTIONS UTILISEES

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

# def fetch_ratings(chunk_size=100000) -> pd.DataFrame:
#     """Récupère enregistrements de la table ratings par morceaux et les transforme en DataFrame."""
#     query = """
#     SELECT userId, movieId, rating
#     FROM ratings
#     """
#     config = load_config()
#     conn = connect(config)

#     if conn is not None:
#         try:
#             df_list = []  # Liste pour stocker les morceaux
#             with conn.cursor() as cur:
#                 cur.execute(query)
#                 while True:
#                     chunk = cur.fetchmany(chunk_size)  # Récupérer un morceau
#                     if not chunk:
#                         break  # Sortir si aucun enregistrement
#                     df_chunk = pd.DataFrame(chunk, columns=['userid', 'movieid', 'rating'])
#                     df_list.append(df_chunk)  # Ajouter le morceau à la liste

#             df = pd.concat(df_list, ignore_index=True)  # Concaténer tous les morceaux
#             print("Enregistrements table ratings récupérés")
#             return df

#         except Exception as e:
#             print(f"Erreur lors de la récupération des enregistrements: {e}")
#             raise


# def fetch_movies() -> pd.DataFrame:
#     """Récupère enregistrements de la table movies et les transforme en DataFrame."""
#     query = """
#     SELECT movieid, title, genres
#     FROM movies
#     """
#     config = load_config()
#     conn = connect(config)

#     if conn is not None:
#         try:
#             with conn.cursor() as cur:
#                 cur.execute(query)
#                 df = pd.DataFrame(cur.fetchall(), columns=['movieid', 'title', 'genres'])
#                 print("Enregistrements table movies récupérés")
#                 return df
#         except Exception as e:
#             print(f"Erreur lors de la récupération des enregistrements: {e}")
#             raise

# def fetch_links() -> pd.DataFrame:
#     """Récupère enregistrements de la table movies et les transforme en DataFrame."""
#     query = """
#     SELECT id, movieid, imdbid, tmdbid
#     FROM links
#     """
#     config = load_config()
#     conn = connect(config)

#     if conn is not None:
#         try:
#             with conn.cursor() as cur:
#                 cur.execute(query)
#                 df = pd.DataFrame(cur.fetchall(), columns=['id', 'movieid', 'imdbid', 'tmdbid'])
#                 print("Enregistrements table links récupérés")
#                 return df
#         except Exception as e:
#             print(f"Erreur lors de la récupération des enregistrements: {e}")
#             raise

# Chargement du dernier modèle
def load_model(model_name):
    """Charge le modèle à partir du répertoire monté."""
    model_path = f"/models/{model_name}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_name} n'existe pas dans {model_path}.")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        print(f'Modèle chargé depuis {model_path}')
    return model

def create_X(chunk_size=100000):
    """
    Génère une matrice creuse avec quatre dictionnaires de mappage à partir d'une base de données,
    sans charger toutes les données en mémoire, en utilisant le chargement par morceaux.

    Args:
        conn: connexion à la base de données.
        chunk_size: nombre de lignes à traiter par morceau.

    Returns:
        X: sparse matrix
        user_mapper: dict that maps user id's to user indices
        user_inv_mapper: dict that maps user indices to user id's
        movie_mapper: dict that maps movie id's to movie indices
        movie_inv_mapper: dict that maps movie indices to movie id's
    """
    conn = connect(load_config())
    cursor = conn.cursor()

    # Récupérer les IDs uniques des utilisateurs et des films
    user_query = "SELECT DISTINCT userid FROM ratings"
    movie_query = "SELECT DISTINCT movieid FROM ratings"

    # Récupérer les IDs uniques
    users = [row[0] for row in cursor.execute(user_query)]
    movies = [row[0] for row in cursor.execute(movie_query)]

    M = len(users)
    N = len(movies)

    # Créer des mappers
    user_mapper = {user_id: index for index, user_id in enumerate(users)}
    movie_mapper = {movie_id: index for index, movie_id in enumerate(movies)}

    user_inv_mapper = {index: user_id for index, user_id in enumerate(users)}
    movie_inv_mapper = {index: movie_id for index, movie_id in enumerate(movies)}

    # Initialiser des listes pour stocker les indices et les notes
    user_index = []
    item_index = []
    ratings = []

    # Exécuter la requête et traiter par morceaux
    ratings_query = "SELECT userid, movieid, rating FROM ratings"

    # Utiliser un curseur pour récupérer les données par morceaux
    rating_chunk = cursor.execute(ratings_query)

    while True:
        chunk = rating_chunk.fetchmany(chunk_size)  # Récupérer un morceau de données
        if not chunk:
            break  # Sortir si aucun enregistrement

        for row in chunk:
            user_id, movie_id, rating = row
            if user_id in user_mapper and movie_id in movie_mapper:
                user_index.append(user_mapper[user_id])
                item_index.append(movie_mapper[movie_id])
                ratings.append(rating)

    # Création de la matrice creuse
    X = csr_matrix((ratings, (user_index, item_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# Fonction pour obtenir des recommandations pour un utilisateur donné
def get_user_recommendations(user_id: int, model: SVD, conn, n_recommendations: int = 10):
    """Obtenir des recommandations pour un utilisateur donné à partir de la base de données."""
    conn = connect(load_config())
    cursor = conn.cursor()
    # Récupérer tous les films
    all_movies_query = "SELECT DISTINCT movieid FROM ratings"
    all_movies = [row[0] for row in cursor.execute(all_movies_query)]

    # Obtenir les films déjà évalués par l'utilisateur
    rated_movies_query = f"SELECT movieid FROM ratings WHERE userid = {user_id}"
    rated_movies = [row[0] for row in cursor.execute(rated_movies_query)]

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
    # # Transposer la matrice X pour que les films soient en lignes et les utilisateurs en colonnes
    # X = X.T

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
                movie_info = data["movie_results"][0]
                results[str(index)] = {
                    "title": movie_info["title"],
                    "vote_average": movie_info['vote_average'],
                    "poster_path": f"http://image.tmdb.org/t/p/w185{movie_info['poster_path']}"
                }

        else:
            results[str(index)] = {"error": f"Request failed with status code {response.status_code}"}

    return results

# Recherche un titre proche de la requete
def movie_finder(all_titles, title):
    closest_match = process.extractOne(title,all_titles)
    return closest_match[0]

def get_movie_id_by_title(user_title):
    conn = connect(load_config())
    cursor = conn.cursor()
    query = f"""
    SELECT movieid, title
    FROM movies
    WHERE title = '{user_title}';
    """
    movie_id = [row[0] for row in cursor.execute(query)]
    return movie_id


def get_best_movies_for_user(user_id, n=3):
    """Obtenir les meilleurs films notés par un utilisateur donné."""
    conn = connect(load_config())
    cursor = conn.cursor()
    query = f"""
    SELECT movieId, rating
    FROM ratings
    WHERE userid = {user_id}
    ORDER BY rating DESC
    LIMIT {n};
    """

    best_movies = [(row[0], row[1]) for row in cursor.execute(query)]
    return best_movies  # Retourner une liste de tuples (movieId, rating)

def get_imdb_ids_for_best_movies(best_movies):
    """Récupère les IMDb IDs pour les meilleurs films notés."""
    conn = connect(load_config())
    cursor = conn.cursor()

    movie_ids = [movie_id for movie_id, _ in best_movies]  # Extraire les movieId des meilleurs films

    # Créer une chaîne de caractères avec les IDs pour la clause IN
    movie_ids_str = ', '.join(map(str, movie_ids))

    # Requête SQL pour récupérer les IMDb IDs
    query = f"""
    SELECT movieid, imdbid
    FROM links
    WHERE movieid IN ({movie_ids_str});
    """

    # Exécuter la requête et créer un dictionnaire pour un accès facile
    imdb_dict = {row[0]: row[1] for row in cursor.execute(query)}

    return imdb_dict  # Retourner un dictionnaire {movieid: imdbid}

def get_imdb_ids_for_recommands_movies(movies_list):
    conn = connect(load_config())
    cursor = conn.cursor()

    # Créer une chaîne de caractères avec les IDs pour la clause IN
    movie_ids_str = ', '.join(map(str, movies_list))

    # Requête SQL pour récupérer les IMDb IDs
    query = f"""
    SELECT movieid, imdbid
    FROM links
    WHERE movieid IN ({movie_ids_str});
    """

    # Exécuter la requête et créer un dictionnaire pour un accès facile
    imdb_dict = {row[0]: row[1] for row in cursor.execute(query)}

    return imdb_dict  # Retourner un dictionnaire {movieid: imdbid}


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
# Nombre de films recommandés
recommendations_counter = Counter(
    name='number_of_recommendations',
    documentation='Number of movie recommendations made',
    labelnames=['endpoint'],
    registry=collector)
# Temps de traitement des requêtes TMDB
tmdb_request_duration_histogram = Histogram(
    name='tmdb_request_duration_seconds',
    documentation='Duration of TMDB API requests',
    labelnames=['endpoint'],
    registry=collector)

# ---------------------------------------------------------------

# CHARGEMENT DES DONNEES AU DEMARRAGE DE API
print("############ DEBUT DES CHARGEMENTS ############")
# # Chargement de nos dataframe
# ratings = fetch_ratings()
# movies = fetch_movies()
# links = fetch_links()
conn = connect(load_config())
cursor = conn.cursor()
# Chargement d'un modèle SVD pré-entraîné pour les recommandations
model_svd = load_model('model_SVD.pkl')
# Chargement de la matrice cosinus similarity
model_Knn = load_model('model_KNN.pkl')
# Création de la matrice utilisateur-article
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_X(chunk_size=100000)
# # Création d'un dataframe pour les liens entre les films et les ID IMDB
# movies_links_df = movies.merge(links, on = "movieid", how = 'left')
# # Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
# movie_idx = dict(zip(movies['title'], list(movies.index)))
# # Création de dictionnaires pour accéder facilement aux titres et aux couvertures des films par leur ID
# movie_titles = dict(zip(movies['movieid'], movies['title']))
# Créer un dictionnaire pour un accès rapide
# imdb_dict = dict(zip(movies_links_df['movieid'], movies_links_df['imdbid']))
query = "SELECT title FROM movies;"
all_titles = [row[0] for row in cursor.execute(query)]
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
        user_id = int(userId)
        best_movies = get_best_movies_for_user(user_id, n=3)
        imdb_dict = get_imdb_ids_for_best_movies(best_movies)
        # Créer la liste des IMDb IDs
        imdb_list = [imdb_dict[movie_id] for movie_id, _ in best_movies if movie_id in imdb_dict]
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(endpoint='/predict/best_user_movies').observe(tmdb_duration)
        recommendations_counter.labels(endpoint='/predict/best_user_movies').inc(len(results))
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
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results
    except ValueError as e:
        status_code_counter.labels(status_code="400").inc()  # Compter les réponses échouées
        error_counter.labels(error_type="ValueError").inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur de conversion de l'ID utilisateur: {e}")
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
        recommendations = get_user_recommendations(user_id, model_svd, conn, n_recommendations = 12)
        logger.info(f"Recommandations pour l'utilisateur {userId}: {recommendations}")
        imdb_dict =get_imdb_ids_for_recommands_movies(recommendations)
        imdb_list = [imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict]
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(endpoint='/predict/identified_user').observe(tmdb_duration)
        recommendations_counter.labels(endpoint='/predict/identified_user').inc(len(results))
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
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results

    except ValueError as e:
        status_code_counter.labels(status_code="400").inc()  # Compter les réponses échouées
        error_counter.labels(error_type="ValueError").inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur de conversion de l'ID utilisateur: {e}")
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
    try:
        # Récupération des données Streamlit
        movie_title = movie_finder(all_titles,  user_request.movie_title)  # Trouver le titre du film correspondant
        movie_id = get_movie_id_by_title( movie_title)
        # Récupérer les ID des films recommandés en utilisant la fonction de similarité
        recommendations = get_movie_title_recommendations(model_Knn, movie_id, X, movie_mapper, movie_inv_mapper, 9)
        imdb_dict =get_imdb_ids_for_recommands_movies(recommendations)
        imdb_list = [imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict]
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(endpoint='/predict/similar_movies').observe(tmdb_duration)
        recommendations_counter.labels(endpoint='/predict/similar_movies').inc(len(results))
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(status_code="200").inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(method='POST', endpoint='/predict/similar_movies', user_id="N/A").observe(duration)  # Enregistrer la durée de la requête
        response_size_histogram.labels(method='POST', endpoint='/predict/similar_movies').observe(response_size)  # Enregistrer la taille de la réponse
        logger.info(f"Api response: {results}")
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results
    except Exception as e:
        status_code_counter.labels(status_code="500").inc()  # Compter les réponses échouées
        error_counter.labels(error_type="Exception").inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

