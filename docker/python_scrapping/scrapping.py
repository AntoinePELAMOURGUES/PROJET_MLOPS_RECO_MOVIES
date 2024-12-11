import os
import time
import requests
from bs4 import BeautifulSoup as bs
import psycopg2
from psycopg2 import OperationalError

def create_connection():
    """Crée une connexion à la base de données PostgreSQL."""
    try:
        conn = psycopg2.connect(
            database=os.getenv("DATABASE"),
            host=os.getenv("AIRFLOW_POSTGRESQL_SERVICE_HOST"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            port=os.getenv("AIRFLOW_POSTGRESQL_SERVICE_PORT")
        )
        print("Connexion à la base de données réussie.")
        return conn
    except OperationalError as e:
        print(f"Erreur lors de la connexion à la base de données: {e}")
        return None

# Récupération du token TMDB depuis les variables d'environnement
tmdb_token = os.getenv("TMDB_TOKEN")

def scrape_imdb_first_page():
    """Scrape les données des films depuis IMDb et les renvoie sous forme de listes."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Récupérer la page des box-offices d'IMDb
        page = requests.get("https://www.imdb.com/chart/boxoffice", headers=headers)
        page.raise_for_status()  # Vérifier que la requête a réussi
        soup = bs(page.content, 'lxml')  # Extraire les liens et titres des films

        links = [a['href'] for a in soup.find_all('a', class_='ipc-title-link-wrapper')]
        cleaned_links = [link.split('/')[2].split('?')[0].replace('tt', '') for link in links]

        return cleaned_links
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération de la page IMDb : {e}")
        return []

def genres_request():
    """Effectue des requêtes à l'API TMDB pour récupérer les informations des genres de films."""
    url = "https://api.themoviedb.org/3/genre/movie/list?language=en"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {tmdb_token}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Vérifier que la requête a réussi

        data = response.json()
        genres = {str(genre["id"]): genre["name"] for genre in data["genres"]}
        return genres
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des genres : {e}")
        return {}

def api_tmdb_request():
    """Effectue des requêtes à l'API TMDB pour récupérer les informations des films."""
    results = {}
    cleaned_links = scrape_imdb_first_page()

    if not cleaned_links:  # Vérifier si le scraping a échoué
        return results

    genres = genres_request()

    for index, movie_id in enumerate(cleaned_links):
        url = f"https://api.themoviedb.org/3/find/tt{movie_id}?external_source=imdb_id"

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {tmdb_token}"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Vérifier que la requête a réussi

            data = response.json()

            if data["movie_results"]:
                movie_info = data["movie_results"][0]
                release_date = movie_info["release_date"]
                release_year = release_date.split("-")[0]  # Extraire l'année

                results[str(index)] = {
                    "tmdb_id": movie_info["id"],
                    "title": movie_info["title"],
                    "genre_ids": movie_info['genre_ids'],
                    "imbd_id": movie_id,
                    "date": release_date,
                    "year": release_year,
                    "genres": [genres[str(genre_id)] for genre_id in movie_info['genre_ids']]
                }
            else:
                results[str(index)] = {"error": f"Aucun résultat trouvé pour l'ID IMDb {movie_id}"}

        except requests.RequestException as e:
            results[str(index)] = {"error": f"Erreur lors de la requête TMDB : {e}"}

    return results

def insert_movies_and_links(scraped_data):
    """Insère les films et les liens dans la base de données."""
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(movieid) FROM movies")
    max_movie_id = cursor.fetchone()[0]
    try:
        for movie_key, movie_info in scraped_data.items():
            if 'error' in movie_info:
                print(f"Ignoré: {movie_info['error']}")
                continue

            required_fields = ['title', 'year', 'genres', 'imbd_id', 'tmdb_id']
            if not all(field in movie_info for field in required_fields):
                print(f"Champs manquants pour l'entrée {movie_key}: {movie_info}")
                continue
            cursor.execute("SELECT MAX(movieid) FROM movies")
            max_movie_id = cursor.fetchone()[0]
            movieid = max_movie_id + 1
            title = movie_info["title"]
            year = int(movie_info["year"])
            genres_str = ','.join(movie_info["genres"])
            imdb_id = int(movie_info["imbd_id"])
            tmdb_id = int(movie_info["tmdb_id"])

            # Vérifier si le film existe déjà
            cursor.execute("SELECT COUNT(*) FROM movies WHERE title = %s AND year = %s", (title, year))
            exists = cursor.fetchone()[0] > 0

            if exists:
                print(f"Le film '{title}' de l'année {year} existe déjà. Ignorer l'insertion.")
                continue

            # Insérer le nouveau film
            cursor.execute(
                "INSERT INTO movies (movieid, title, genres, year) VALUES (%s, %s, %s, %s)",
                (movieid, title, genres_str, year)
            )

            # Insérer les liens associés
            cursor.execute(
                "INSERT INTO links (movieid, imdbid, tmdbid) VALUES (%s, %s, %s)",
                (movieid, imdb_id, tmdb_id)
            )

        conn.commit()  # Valider les modifications
    except Exception as e:
        print(f"Erreur lors de l'insertion: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    results = api_tmdb_request()
    print(results)
    insert_movies_and_links(results)