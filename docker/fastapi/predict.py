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
from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD

# Récupérer le token TMDB API depuis les variables d'environnement
tmdb_token = os.getenv("TMDB_API_TOKEN")

# Configuration du logger pour afficher les informations de débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration Kubernetes
config.load_incluster_config()

# Définir les volumes pour le stockage des modèles et des données brutes
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

# Définir les points de montage des volumes
volume1_mount = client.V1VolumeMount(name="model-storage", mount_path="/models")
volume2_mount = client.V1VolumeMount(name="raw-storage", mount_path="/data")

# Routeur pour gérer les routes de prédiction
router = APIRouter(
    prefix="/predict",  # Préfixe pour toutes les routes dans ce routeur
    tags=["predict"],  # Tag pour la documentation
)


# Fonction pour charger un fichier CSV en DataFrame par morceaux
def load_csv_to_df(file_path: str, chunk_size: int = 2000) -> pd.DataFrame:
    """Charge un fichier CSV en DataFrame par morceaux pour gérer les grands fichiers.

    Args:
        file_path (str): Chemin vers le fichier CSV.
        chunk_size (int, optional): Taille des morceaux pour la lecture. Defaults to 2000.

    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier CSV. Retourne un DataFrame vide en cas d'erreur.
    """
    try:
        # Initialiser une liste pour stocker les morceaux
        chunks = []

        # Lire le fichier CSV par morceaux
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)

        # Concaténer tous les morceaux en un seul DataFrame
        df = pd.concat(chunks, ignore_index=True)
        print(f"Chargement du fichier {file_path} réussi.")
        return df

    except FileNotFoundError:
        print(f"Le fichier {file_path} est introuvable.")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur


# Fonction pour charger le modèle
def load_model(model_name: str):
    """Charge le modèle à partir du répertoire monté.

    Args:
        model_name (str): Nom du fichier du modèle.

    Returns:
        tuple: Un tuple contenant le modèle et le lecteur associé.

    Raises:
        FileNotFoundError: Si le modèle n'existe pas dans le répertoire spécifié.
    """
    model_path = f"/models/{model_name}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle {model_name} n'existe pas dans {model_path}."
        )
    with open(model_path, "rb") as file:
        model_data = pickle.load(file)
        model = model_data["model"]
        reader = model_data["reader"]
        print(f"Modèle chargé depuis {model_path}")
    return model, reader


# Fonction pour recommander des films à un utilisateur identifié
def recommend_movies(
    model: SVD, user_id: int, df: pd.DataFrame, top_n: int = 15
) -> list:
    """Recommande des films à un utilisateur en fonction de ses évaluations prédites.

    Args:
        model (SVD): Le modèle SVD entraîné.
        user_id (int): L'ID de l'utilisateur pour lequel on veut des recommandations.
        df (pd.DataFrame): DataFrame contenant les données des films et les évaluations.
        top_n (int, optional): Le nombre de films à recommander. Defaults to 15.

    Returns:
        list: Une liste des top_n films recommandés.
    """
    # Obtenir la liste des films que l'utilisateur a déjà évalués
    movies_already_rated = df[df["userid"] == user_id]["movieid"].unique()

    # Créer une liste de tous les films possibles
    all_movie_ids = df["movieid"].unique()

    # Filtrer les films que l'utilisateur n'a pas encore évalués
    movies_to_predict = [
        movie_id for movie_id in all_movie_ids if movie_id not in movies_already_rated
    ]

    # Faire des prédictions pour chaque film non évalué
    predictions = []
    for movie_id in movies_to_predict:
        predictions.append((movie_id, model.predict(user_id, movie_id).est))

    # Trier les prédictions par ordre décroissant d'évaluation prédite
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Retourner les top_n films
    top_recommendations = [movie_id for movie_id, _ in predictions[:top_n]]
    return top_recommendations


# Fonction pour charger les artefacts du modèle TF-IDF
def load_tfidf_model_artifacts(data_directory: str):
    """Charge les artefacts du modèle TF-IDF sauvegardés.

    Args:
        data_directory (str): Le chemin vers le répertoire contenant les artefacts du modèle.

    Returns:
        tuple: Un tuple contenant le modèle TF-IDF, la matrice de similarité cosinus et les indices.
    """
    tfidf = joblib.load(os.path.join(data_directory, "tfidf_model.joblib"))
    sim_cosinus = joblib.load(os.path.join(data_directory, "sim_cosinus.joblib"))
    indices = pd.read_pickle(os.path.join(data_directory, "indices.pkl"))
    return tfidf, sim_cosinus, indices


# Fonction pour obtenir les recommandations sans notion d'user_id
def recommandations(
    titre: str,
    sim_cosinus: np.ndarray,
    indices: pd.Series,
    title_to_movieid: Dict[str, int],
    num_recommandations: int = 10,
) -> list:
    """À partir des indices trouvés, renvoie les movie_id des films les plus similaires.

    Args:
        titre (str): Titre du film de référence.
        sim_cosinus (np.ndarray): Matrice de similarité cosinus.
        indices (pd.Series): Série contenant les indices des films.
        title_to_movieid (Dict[str, int]): Dictionnaire associant les titres de films à leurs IDs.
        num_recommandations (int, optional): Nombre de recommandations à retourner. Defaults to 10.

    Returns:
        list: Une liste des movie_id des films les plus similaires.
    """
    # Récupérer dans idx l'indice associé au titre depuis la série indices
    idx = indices[titre]
    # Garder dans une liste les scores de similarité correspondants à l'index du film cible
    score_sim = list(enumerate(sim_cosinus[idx]))
    # Trier les scores de similarité, trouver les plus similaires et récupérer ses indices
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
def movie_finder(all_titles: list, title: str) -> Optional[str]:
    """Trouve le titre de film le plus proche d'une requête donnée.

    Args:
        all_titles (list): Liste de tous les titres de films disponibles.
        title (str): Titre du film à rechercher.

    Returns:
        str: Le titre du film le plus proche trouvé. Retourne None si aucun match n'est trouvé.
    """
    closest_match = process.extractOne(title, all_titles)
    return (
        closest_match[0] if closest_match else None
    )  # Retourne None si aucun match n'est trouvé


# Formater l'ID du film pour la requête TMDB
def format_movie_id(movie_id: int) -> str:
    """Transforme en ImdbId et Formate l'ID du film pour qu'il ait 7 chiffres.

    Args:
        movie_id (int): L'ID du film à formater.

    Returns:
        str: L'ID du film formaté.
    """
    imdbid_format = str(movie_id).zfill(7)  # Formate l'ID pour qu'il ait 7 chiffres
    return imdbid_format


# Effectuer une requête à l'API TMDB
def api_tmdb_request(movie_ids: list) -> Dict[str, Any]:
    """Effectue des requêtes à l'API TMDB pour récupérer les informations des films.

    Args:
        movie_ids (list): Une liste d'IDs de films (IMDB IDs).

    Returns:
        Dict[str, Any]: Un dictionnaire contenant les informations des films récupérées depuis TMDB.
    """
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


# METRICS PROMETHEUS
collector = CollectorRegistry()

# Nombre de requêtes
nb_of_requests_counter = Counter(
    name="predict_nb_of_requests",
    documentation="nombre de requêtes par méthode ou par endpoint",
    labelnames=["method", "endpoint"],
    registry=collector,
)

# Codes de statut des réponses
status_code_counter = Counter(
    name="predict_response_status_codes",
    documentation="Nombre de réponses HTTP par code de statut",
    labelnames=["status_code"],
    registry=collector,
)

# Taille des réponses
response_size_histogram = Histogram(
    name="http_response_size_bytes",
    documentation="Taille des réponses HTTP en octets",
    labelnames=["method", "endpoint"],
    registry=collector,
)

# Temps de traitement par utilisateur
duration_of_requests_histogram = Histogram(
    name="duration_of_requests",
    documentation="Durée des requêtes par méthode ou endpoint",
    labelnames=["method", "endpoint", "user_id"],
    registry=collector,
)

# Erreurs spécifiques
error_counter = Counter(
    name="api_errors",
    documentation="Nombre d'erreurs API par type",
    labelnames=["error_type"],
    registry=collector,
)

# Nombre de films recommandés
recommendations_counter = Counter(
    name="number_of_recommendations",
    documentation="Nombre de recommandations de films effectuées",
    labelnames=["endpoint"],
    registry=collector,
)

# Temps de traitement des requêtes TMDB
tmdb_request_duration_histogram = Histogram(
    name="tmdb_request_duration_seconds",
    documentation="Durée des requêtes TMDB API",
    labelnames=["endpoint"],
    registry=collector,
)

# CHARGEMENT DES DONNEES AU DEMARRAGE DE API
print("############ DEBUT DES CHARGEMENTS ############")

# Chargement des dataframes
ratings = load_csv_to_df("/data/processed_ratings.csv")
movies = load_csv_to_df("/data/processed_movies.csv")
df = pd.merge(ratings, movies, on="movieid", how="left")
links = load_csv_to_df("/data/processed_links.csv")

# Charger les artefacts du modèle SVD
model, reader = load_model("svd_model_v1.pkl")
df_svd = Dataset.load_from_df(df[["userid", "movieid", "rating"]], reader=reader)

# Charger les artefacts du modèle TF-IDF
tfidf, sim_cosinus, indices = load_tfidf_model_artifacts("/models/")

# Création d'un dataframe pour les liens entre les films et les ID IMDB
movies_links_df = movies.merge(links, on="movieid", how="left")

# Création de dictionnaires pour faciliter l'accès aux titres et aux couvertures des films par leur ID
movie_titles = dict(zip(movies["movieid"], movies["title"]))

# Création d'un dictionnaire pour lier title et movieid
title_to_movieid = dict(zip(movies["title"], movies["movieid"]))

# Créer un dictionnaire pour un accès rapide
imdb_dict = dict(zip(movies_links_df["movieid"], movies_links_df["imdbid"]))

# Créer une liste de tous les titres de films
all_titles = movies["title"].tolist()

print("############ FIN DES CHARGEMENTS ############")


# REQUETES API
# Modèle Pydantic pour la récupération de l'user_id lié aux films
class UserRequest(BaseModel):
    userId: Optional[int]  # Forcer le type int explicitement
    movie_title: Optional[str] = None

    class Config:
        json_schema_extra = {"example": {"userId": 1, "movie_title": "Inception"}}


@router.post("/best_user_movies")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """Route API pour récupérer les 3 films les mieux notés de l'utilisateur.

    Args:
        user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur,y compris l'ID utilisateur et le titre du film.

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
    userId = user_request.userId  # récupération de l'userId dans la base de données

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
        ) from e
    except Exception as e:
        status_code_counter.labels(
            status_code="500"
        ).inc()  # Compter les erreurs du serveur
        error_counter.labels(
            error_type="InternalServerError"
        ).inc()  # Enregistrer l'erreur
        logger.error(f"Erreur interne du serveur: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur") from e


@router.post("/identified_user")
async def predict(user_request: UserRequest) -> Dict[str, Any]:
    """Route API pour obtenir des recommandations de films basées sur l'ID utilisateur.

    Args:
        user_request (UserRequest): Un objet contenant les détails de la requête de l'utilisateur,y compris l'ID utilisateur.

    Returns:
        Dict[str, Any]: Un dictionnaire contenant des recommandations de films.
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

    try:
        # Forcer la conversion en int
        user_id = int(user_request.userId)
        recommendations = recommend_movies(model, user_id, df_svd, top_n=15)
        titles = [
            movie_titles[movie_id]
            for movie_id in recommendations
            if movie_id in movie_titles
        ]

        start_tmdb_time = time.time()
        imdb_list = [
            imdb_dict[movie_id] for movie_id in recommendations if movie_id in imdb_dict
        ]

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
        ) from e
    except Exception as e:
        status_code_counter.labels(
            status_code="500"
        ).inc()  # Compter les erreurs du serveur
        error_counter.labels(
            error_type="InternalServerError"
        ).inc()  # Enregistrer l'erreur
        logger.error(f"Erreur interne du serveur: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur") from e
