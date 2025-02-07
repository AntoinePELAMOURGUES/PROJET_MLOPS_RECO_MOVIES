import os
import pandas as pd
import json
import pickle
from surprise.prediction_algorithms.matrix_factorization import SVD
import numpy as np
from rapidfuzz import process
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
from pydantic import BaseModel
import requests
import logging
from kubernetes import client, config
from surprise import Reader
from surprise import Dataset


tmdb_token = os.getenv("TMDB_API_TOKEN")

# Configuration du logger pour afficher les informations de débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration Kubernetes
config.load_incluster_config()

# Définir le volume et le montage du volume
volume1 = client.V1Volume(
    name="model-storage",
    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
        claim_name="model-storage-pvc"
    ),
)

volume2 = client.V1Volume(
    name="raw-storage",
    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
        claim_name="raw-storage-pvc"
    ),
)

volume1_mount = client.V1VolumeMount(name="model-storage", mount_path="/models")

volume2_mount = client.V1VolumeMount(name="raw-storage", mount_path="/data")

# ROUTEUR POUR GERER LES ROUTES PREDICT

router = APIRouter(
    prefix="/predict",  # Préfixe pour toutes les routes dans ce routeur
    tags=["predict"],  # Tag pour la documentation
)


# ENSEMBLE DES FONCTIONS UTILISEES


def load_csv_to_df(file_path: str, chunk_size: int = 2000) -> pd.DataFrame:
    """Charge un fichier CSV en DataFrame par morceaux."""
    try:
        # Initialiser une liste pour stocker les morceaux
        chunks = []

        # Lire le fichier CSV par morceaux
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Traitez chaque morceau ici si nécessaire
            chunks.append(chunk)

        # Concaténer tous les morceaux en un seul DataFrame
        df = pd.concat(chunks, ignore_index=True)
        print(f"Chargement du fichier {file_path} réussi.")
        return df

    except FileNotFoundError:
        print(f"Le fichier {file_path} est introuvable.")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur


# Chargement du dernier modèle
def load_model(model_name):
    """Charge le modèle à partir du répertoire monté."""
    model_path = f"/models/{model_name}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle {model_name} n'existe pas dans {model_path}."
        )
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        print(f"Modèle chargé depuis {model_path}")
    return model


# Fonction pour obtenir des recommandations pour un utilisateur donné
def get_user_recommendations(
    ratings_df, user_id: int, model: SVD, n_recommendations: int = 24
):
    """Obtenir des recommandations pour un utilisateur donné."""
    # Initialiser une liste vide pour stocker les paires (utilisateur, movie) pour le jeu "anti-testset"
    anti_testset = []
    # Convertir l'ID de l'utilisateur externe en l'ID interne utilisé par Surprise
    targetUser = train_set.to_inner_uid(user_id)
    # Obtenir la valeur de remplissage à utiliser (moyenne globale des notes du jeu d'entraînement)
    moyenne = train_set.global_mean
    # Obtenir les évaluations de l'utilisateur cible pour les movies
    user_note = train_set.ur[targetUser]
    # Extraire la liste des movies notés par l'utilisateur
    user_movie = [item for (item,_) in (user_note)]
    # Obtenir toutes les notations du jeu d'entraînement
    ratings = train_set.all_ratings()
    # Boucle sur tous les items du jeu d'entraînement
    for movie in train_set.all_items():
    # Si l'item n'a pas été noté par l'utilisateur
        if movie not in user_movie:
            # Ajouter la paire (utilisateur, movie, valeur de remplissage) à la liste "anti-testset"
            anti_testset.append((user_id, train_set.to_raw_iid(movie), moyenne))
    predictionsSVD = model.test(anti_testset)
    # Convertir les prédictions en un DataFrame pandas
    predictionsSVD = pd.DataFrame(predictionsSVD)
    # Trier les prédictions par la colonne 'est' (estimation) en ordre décroissant
    predictionsSVD.sort_values(by=['est'], inplace=True, ascending=False)
    # Afficher les 10 meilleures prédictions
    return predictionsSVD["iid"].values[:n_recommendations]


# Recherche un titre proche de la requête
def movie_finder(all_titles, title):
    """
    Trouve le titre de film le plus proche d'une requête donnée.
    Args:
        all_titles (list): Liste de tous les titres de films disponibles.
        title (str): Titre du film à rechercher.
    Returns:
        str: Le titre du film le plus proche trouvé.
    """
    closest_match = process.extractOne(title, all_titles)
    return (
        closest_match[0] if closest_match else None
    )  # Retourne None si aucun match n'est trouvé


# Focntion qui regroupe les recommandations
def get_content_based_recommendations(
    all_titles, title_string, cosine_sim, n_recommendations=15
):
    title = movie_finder(all_titles, title_string)
    idx = movie_idx[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : (n_recommendations + 1)]
    similar_movies = [i[0] for i in sim_scores]
    return similar_movies


def format_movie_id(movie_id):
    """Transforme en ImdbId et  Formate l'ID du film pour qu'il ait 7 chiffres."""
    imdbid_format = str(movie_id).zfill(7)  # Formate l'ID pour qu'il ait 7 chiffres
    return imdbid_format


def api_tmdb_request(movie_ids):
    """Effectue des requêtes à l'API TMDB pour récupérer les informations des films."""
    results = {}

    for movie_id in movie_ids:
        formatted_id = format_movie_id(movie_id)
        url = f"https://api.themoviedb.org/3/find/tt{formatted_id}?external_source=imdb_id"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {tmdb_token}",
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data["movie_results"]:
                index = len(results)
                movie_info = data["movie_results"][0]
                results[str(index)] = {
                    "title": movie_info["title"],
                    "vote_average": movie_info["vote_average"],
                    "poster_path": f"http://image.tmdb.org/t/p/w185{movie_info['poster_path']}",
                }

        # Vérifie si nous avons atteint 12 résultats
        if len(results) > 11:
            break

    return results


#


# ---------------------------------------------------------------

# METRICS PROMETHEUS

collector = CollectorRegistry()
# Nbre de requête
nb_of_requests_counter = Counter(
    name="predict_nb_of_requests",
    documentation="number of requests per method or per endpoint",
    labelnames=["method", "endpoint"],
    registry=collector,
)
# codes de statut des réponses
status_code_counter = Counter(
    name="predict_response_status_codes",
    documentation="Number of HTTP responses by status code",
    labelnames=["status_code"],
    registry=collector,
)
# Taille des réponses
response_size_histogram = Histogram(
    name="http_response_size_bytes",
    documentation="Size of HTTP responses in bytes",
    labelnames=["method", "endpoint"],
    registry=collector,
)
# Temps de traitement par utilisateur
duration_of_requests_histogram = Histogram(
    name="duration_of_requests",
    documentation="Duration of requests per method or endpoint",
    labelnames=["method", "endpoint", "user_id"],
    registry=collector,
)
# Erreurs spécifiques
error_counter = Counter(
    name="api_errors",
    documentation="Count of API errors by type",
    labelnames=["error_type"],
    registry=collector,
)
# Nombre de films recommandés
recommendations_counter = Counter(
    name="number_of_recommendations",
    documentation="Number of movie recommendations made",
    labelnames=["endpoint"],
    registry=collector,
)
# Temps de traitement des requêtes TMDB
tmdb_request_duration_histogram = Histogram(
    name="tmdb_request_duration_seconds",
    documentation="Duration of TMDB API requests",
    labelnames=["endpoint"],
    registry=collector,
)

# ---------------------------------------------------------------

# CHARGEMENT DES DONNEES AU DEMARRAGE DE API
print("############ DEBUT DES CHARGEMENTS ############")
# Chargement de nos dataframe
ratings = load_csv_to_df("/data/processed_ratings.csv")
movies = load_csv_to_df("/data/processed_movies.csv")
links = load_csv_to_df("/data/processed_links.csv")
# ratings = fetch_ratings()
# movies = fetch_movies()
# links = fetch_links()
# Chargement d'un modèle SVD pré-entraîné pour les recommandations
model_svd = load_model("model_SVD.pkl")
# Chargement de la matrice cosinus similarity
similarity_cosinus = np.load("/models/cosine_similarity_matrix.npy")
# Création d'un dataframe pour les liens entre les films et les ID IMDB
movies_links_df = movies.merge(links, on="movieid", how="left")
# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_idx = dict(zip(movies["title"], list(movies.index)))
# Création de dictionnaires pour accéder facilement aux titres et aux couvertures des films par leur ID
movie_titles = dict(zip(movies["movieid"], movies["title"]))
# Créer un dictionnaire pour un accès rapide
imdb_dict = dict(zip(movies_links_df["movieid"], movies_links_df["imdbid"]))
# Créer une liste de tous les titres de films
all_titles = movies["title"].tolist()
# Créer un dataset surprise pour les recommandations
reader = Reader(rating_scale = (0, 5))
ratings_surprise = Dataset.load_from_df(ratings[["userid", "movieid", "rating"]])
# Construire le jeu d'entraînement complet à partir du DataFrame df_surprise
train_set = ratings_surprise.build_full_trainset()
print("############ FIN DES CHARGEMENTS ############")
# ---------------------------------------------------------------

# REQUETES API


# Modèle Pydantic pour la récupération de l'user_id lié aux films
class UserRequest(BaseModel):
    userId: Optional[int]  # Forcer le type int explicitement
    movie_title: Optional[str] = None

    class Config:
        json_schema_extra = {"example": {"userId": 1, "movie_title": "Inception"}}


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
    nb_of_requests_counter.labels(
        method="POST", endpoint="/predict/best_user_movies"
    ).inc()
    # Récupération de l'email utilisateur de la session
    userId = user_request.userId  # récupéartion de l'userId dans la base de données
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    try:
        # Forcer la conversion en int
        user_id = int(userId)
        df_user = ratings[ratings["userid"] == user_id]
        df_user = df_user.sort_values(by="rating", ascending=False)
        best_movies = df_user.head(3)
        imdb_list = [
            imdb_dict[movie_id]
            for movie_id in best_movies["movieid"]
            if movie_id in imdb_dict
        ]
        logger.info(f"Meilleurs films pour l'utilisateur {userId}: {imdb_list}")
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(
            endpoint="/predict/best_user_movies"
        ).observe(tmdb_duration)
        recommendations_counter.labels(endpoint="/predict/best_user_movies").inc(
            len(results)
        )
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(
            status_code="200"
        ).inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(
            method="POST", endpoint="/predict/best_user_movies", user_id=str(user_id)
        ).observe(
            duration
        )  # Enregistrer la durée de la requête
        response_size_histogram.labels(
            method="POST", endpoint="/predict/best_user_movies"
        ).observe(
            response_size
        )  # Enregistrer la taille de la réponse
        # Utiliser le logger pour voir les résultats
        logger.info(f"Api response: {results}")
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results
    except ValueError as e:
        status_code_counter.labels(
            status_code="400"
        ).inc()  # Compter les réponses échouées
        error_counter.labels(
            error_type="ValueError"
        ).inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur de conversion de l'ID utilisateur: {e}")
        raise HTTPException(
            status_code=400, detail="L'ID utilisateur doit être un nombre entier"
        )


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
    nb_of_requests_counter.labels(
        method="POST", endpoint="/predict/identified_user"
    ).inc()
    # Récupération de l'email utilisateur de la session
    userId = user_request.userId  # récupéartion de l'userId dans la base de données
    # Récupérer les ID des films recommandés en utilisant la fonction de similarité
    try:
        # Forcer la conversion en int
        user_id = int(user_request.userId)
        recommendations = get_user_recommendations(
            ratings, user_id, model_svd, n_recommendations=24
        )
        logger.info(f"Recommandations pour l'utilisateur {userId}: {recommendations}")
        imdb_list = [
            imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict
        ]
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(
            endpoint="/predict/identified_user"
        ).observe(tmdb_duration)
        recommendations_counter.labels(endpoint="/predict/identified_user").inc(
            len(results)
        )
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(
            status_code="200"
        ).inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(
            method="POST", endpoint="/predict/identified_user", user_id=str(user_id)
        ).observe(
            duration
        )  # Enregistrer la durée de la requête
        response_size_histogram.labels(
            method="POST", endpoint="/predict/identified_user"
        ).observe(
            response_size
        )  # Enregistrer la taille de la réponse
        # Utiliser le logger pour voir les résultats
        logger.info(f"Api response: {results}")
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results

    except ValueError as e:
        status_code_counter.labels(
            status_code="400"
        ).inc()  # Compter les réponses échouées
        error_counter.labels(
            error_type="ValueError"
        ).inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur de conversion de l'ID utilisateur: {e}")
        raise HTTPException(
            status_code=400, detail="L'ID utilisateur doit être un nombre entier"
        )


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
    nb_of_requests_counter.labels(
        method="POST", endpoint="/predict/similar_movies"
    ).inc()
    movie_title = user_request.movie_title
    try:
        # Récupérer les ID des films recommandés en utilisant la fonction de similarité
        recommendations = get_content_based_recommendations(
            movie_titles, movie_title, similarity_cosinus, n_recommendations=24
        )
        movies_id = [movies["movieid"].iloc[i] for i in recommendations]
        imdb_list = [
            imdb_dict[movie_id] for movie_id in movies_id if movie_id in imdb_dict
        ]
        start_tmdb_time = time.time()
        results = api_tmdb_request(imdb_list)
        tmdb_duration = time.time() - start_tmdb_time
        tmdb_request_duration_histogram.labels(
            endpoint="/predict/similar_movies"
        ).observe(tmdb_duration)
        recommendations_counter.labels(endpoint="/predict/similar_movies").inc(
            len(results)
        )
        # Mesurer la taille de la réponse et l'enregistrer
        response_size = len(json.dumps(results))
        # Calculer la durée et enregistrer dans l'histogramme
        duration = time.time() - start_time
        # Enregistrement des métriques pour Prometheus
        status_code_counter.labels(
            status_code="200"
        ).inc()  # Compter les réponses réussies
        duration_of_requests_histogram.labels(
            method="POST", endpoint="/predict/similar_movies", user_id="N/A"
        ).observe(
            duration
        )  # Enregistrer la durée de la requête
        response_size_histogram.labels(
            method="POST", endpoint="/predict/similar_movies"
        ).observe(
            response_size
        )  # Enregistrer la taille de la réponse
        logger.info(f"Api response: {results}")
        logger.info(f"Durée de la requête: {duration} secondes")
        logger.info(f"Taille de la réponse: {response_size} octets")
        return results
    except Exception as e:
        status_code_counter.labels(
            status_code="500"
        ).inc()  # Compter les réponses échouées
        error_counter.labels(
            error_type="Exception"
        ).inc()  # Enregistrer l'erreur spécifique
        logger.error(f"Erreur lors du traitement de la requête: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
