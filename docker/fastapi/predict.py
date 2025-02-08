import os
import pandas as pd
import json
import pickle
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
import joblib


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


# Chargement des données matrix_factorization_model
def load_model_artifacts(data_directory):
    """Charge les artefacts du modèle sauvegardés."""
    svd = joblib.load(os.path.join(data_directory, "svd_model.joblib"))
    with open(os.path.join(data_directory, "titles.pkl"), "rb") as f:
        titles = pickle.load(f)
    item_similarity = joblib.load(
        os.path.join(data_directory, "item_similarity.joblib")
    )
    mat_ratings = joblib.load(os.path.join(data_directory, "mat_ratings.joblib"))
    return mat_ratings, item_similarity, titles, svd


# Fonction pour obtenir des recommandations pour un utilisateur donné
def pred_item(mat_ratings, item_similarity, k, user_id):
    # Sélectionner dans mat_ratings les films qui n'ont pas été encore lu par le user
    to_predict = mat_ratings.loc[user_id][mat_ratings.loc[user_id] == 0]
    # Itérer sur tous ces films
    for i in to_predict.index:
        # Trouver les k films les plus similaires en excluant le film lui-même
        similar_items = item_similarity.loc[i].sort_values(ascending=False)[1 : k + 1]
        # Calcul de la norme du vecteur similar_items
        norm = np.sum(np.abs(similar_items))
        # Récupérer les notes données par l'utilisateur aux k plus proches voisins
        ratings = mat_ratings[similar_items.index].loc[user_id]
        # Calculer le produit scalaire entre ratings et similar_items
        scalar_prod = np.dot(ratings, similar_items)
        # Calculer la note prédite pour le film i
        pred = scalar_prod / norm
        # Remplacer par la prédiction
        to_predict[i] = pred
    return to_predict


# Fonction pour obtenir les recommandations sans notion d'user_id
def load_tfidf_model_artifacts(data_directory):
    """Charge les artefacts du modèle TF-IDF sauvegardés."""
    tfidf = joblib.load(os.path.join(data_directory, "tfidf_model.joblib"))
    sim_cosinus = joblib.load(os.path.join(data_directory, "sim_cosinus.joblib"))
    indices = pd.read_pickle(os.path.join(data_directory, "indices.pkl"))
    return tfidf, sim_cosinus, indices


def recommandations(
    titre, sim_cosinus, indices, title_to_movieid, num_recommandations=10
):
    """Fonction qui à partir des indices trouvés, renvoie les movie_id des films les plus similaires."""
    # récupérer dans idx l'indice associé au titre depuis la série indices
    idx = indices[titre]
    # garder dans une liste les scores de similarité correspondants à l'index du film cible
    score_sim = list(enumerate(sim_cosinus[idx]))
    #  trier les scores de similarité, trouver les plus similaires et récupérer ses indices
    score_sim = sorted(score_sim, key=lambda x: x[1], reverse=True)
    # Obtenir les scores des 10 films les plus similaires
    top_similair = score_sim[1 : num_recommandations + 1]
    # Obtenir les indices des films
    res = [(indices.index[idx], score) for idx, score in top_similair]
    result = []
    for i in range(len(res)):
        result.append(title_to_movieid[res[i][0]])
    return result


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
# Charger les artefacts du modèle
mat_ratings, item_similarity, titles, svd = load_model_artifacts("/models/")
# Charger les artefacts du modèle
tfidf, sim_cosinus, indices = load_tfidf_model_artifacts("/models/")
# Création d'un dataframe pour les liens entre les films et les ID IMDB
movies_links_df = movies.merge(links, on="movieid", how="left")
# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_idx = dict(zip(movies["title"], list(movies.index)))
# Création de dictionnaires pour accéder facilement aux titres et aux couvertures des films par leur ID
movie_titles = dict(zip(movies["movieid"], movies["title"]))
# Création d'un dictionnaire pour liier title et movieid
title_to_movieid = dict(zip(movies["title"], movies["movieid"]))
# Créer un dictionnaire pour un accès rapide
imdb_dict = dict(zip(movies_links_df["movieid"], movies_links_df["imdbid"]))
# Créer une liste de tous les titres de films
all_titles = movies["title"].tolist()
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
        recommendations = pred_item(mat_ratings, item_similarity, 12, user_id)
        recommandations = recommandations.sort_values(ascending=False).head(12)
        titles = recommandations.index.tolist()
        logger.info(f"Recommandations pour l'utilisateur {userId}: {recommendations}")
        reco = [title_to_movieid[title] for title in titles]
        imdb_list = [imdb_dict[movie_id] for movie_id in reco if movie_id in imdb_dict]
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
    movie_title = movie_finder(all_titles, movie_title)
    try:
        # Récupérer les ID des films recommandés en utilisant la fonction de similarité
        recommendations = recommandations(
            movie_title, sim_cosinus, indices, num_recommandations=12
        )
        imdb_list = [
            imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict
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
