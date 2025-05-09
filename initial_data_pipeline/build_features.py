import pandas as pd
import os
from passlib.context import CryptContext
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Configuration du contexte pour le hachage des mots de passe
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

my_project_directory = os.getenv("MY_PROJECT_DIRECTORY")


def load_data(raw_data_relative_path):
    """
    Charge les données des fichiers CSV dans des DataFrames pandas.

    Args:
        raw_data_relative_path (str): Chemin vers le répertoire contenant les fichiers CSV.

    Returns:
        tuple: DataFrames pour les évaluations, les films et les liens.
    """
    try:
        df_ratings = pd.read_csv(f"{raw_data_relative_path}/ratings.csv")
        df_movies = pd.read_csv(f"{raw_data_relative_path}/movies.csv")
        df_links = pd.read_csv(f"{raw_data_relative_path}/links.csv")
        print(
            f" Ratings, movies and links loaded from {raw_data_relative_path} directory"
        )
        return df_ratings, df_movies, df_links
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"No data: {e}")
    except Exception as e:
        print(f"An error occurred while loading data: {e}")


def preprocessing_ratings(df_ratings) -> pd.DataFrame:
    """
    Change les noms des colonnes et nettoie le fichier CSV des évaluations.
    """
    print("Début preprocessing ratings")
    df_ratings["userId"] = df_ratings["userId"].astype(int)
    # Filtrer les évaluations pour les utilisateurs ayant un ID inférieur à 20000 pour réduire la taille du dataset
    df_ratings = df_ratings[df_ratings["userId"] < 20000]
    # Renommer les colonnes
    df_ratings = df_ratings.rename(columns={"userId": "userid", "movieId": "movieid"})
    print("Preprocessing ratings OK")
    return df_ratings


def preprocessing_movies(df_movies) -> pd.DataFrame:
    """
    Traite le fichier CSV des films et extrait les informations nécessaires.

    Args:
        df_movies (pd.DataFrame): DataFrame contenant les films.

    Returns:
        pd.DataFrame: DataFrame contenant les films traités.
    """
    print(" Début preprocessing movies")
    print("Création d'une colonne year et passage des genres en liste de genres")

    # Séparer les genres sur les pipes et les joindre par des virgules
    df_movies["genres"] = df_movies["genres"].apply(lambda x: ", ".join(x.split("|")))

    # Extraction de l'année et mise à jour du titre
    df_movies["year"] = df_movies["title"].str.extract(r"\((\d{4})\)")[0]

    # Nettoyer le titre en retirant l'année
    df_movies["title"] = df_movies["title"].str.replace(r" \(\d{4}\)", "", regex=True)

    # Remplir les valeurs manquantes avec la méthode forward fill
    df_movies.ffill(inplace=True)

    df_movies = df_movies.rename(columns={"movieId": "movieid"})
    print(" Preprocessing movies OK")
    return df_movies


def preprocessing_links(df_links) -> pd.DataFrame:
    """
    Modifie le type de tmdbId dans le dataset des liens.

    Args:
        df_links (pd.DataFrame): DataFrame contenant les liens.

    Returns:
        pd.DataFrame: DataFrame contenant les liens traités.

    """
    print(" Début preprocessing links")
    print("Modification du type de la colonne tmdbId en int")
    # Remplacer les valeurs manquantes par 0 et convertir en entier
    df_links["tmdbId"] = df_links.tmdbId.fillna(0).astype(int)

    # Renommer les colonnes
    df_links = df_links.rename(
        columns={"tmdbId": "tmdbid", "imdbId": "imdbid", "movieId": "movieid"}
    )
    print(" Preprocessing links OK")
    return df_links


def create_users() -> pd.DataFrame:
    """
    Crée un DataFrame d'utilisateurs fictifs avec mots de passe hachés.

    Returns:
        pd.DataFrame: DataFrame contenant les utilisateurs créés.
    """

    print(" Création des utilisateurs _ fichier users.csv")
    username = []
    email = []
    password = []

    for i in range(1, 501):
        username.append("user" + str(i))
        email.append("user" + str(i) + "@example.com")
        password.append("password" + str(i))

    hached_password = [bcrypt_context.hash(i) for i in password]

    # Créer un DataFrame
    df_users = pd.DataFrame(
        {"username": username, "email": email, "hached_password": hached_password}
    )
    print("Création users OK")
    return df_users


def save_data(df_ratings, df_movies, df_links, df_users, data_directory):
    """
    Enregistre les DataFrames traités dans des fichiers CSV.

    Args:
        df_ratings (pd.DataFrame): DataFrame contenant les évaluations traitées.
        df_movies (pd.DataFrame): DataFrame contenant les films traités.
        df_links (pd.DataFrame): DataFrame contenant les liens traités.
        df_users (pd.DataFrame): DataFrame contenant les utilisateurs créés.
        data_directory (str): Répertoire où enregistrer les fichiers CSV.
    """

    os.makedirs(data_directory, exist_ok=True)  # Crée le répertoire si nécessaire

    try:
        # Enregistrement des fichiers CSV
        df_ratings.to_csv(f"{data_directory}/processed_ratings.csv", index=False)
        df_movies.to_csv(f"{data_directory}/processed_movies.csv", index=False)
        df_links.to_csv(f"{data_directory}/processed_links.csv", index=False)
        df_users.to_csv(f"{data_directory}/users.csv", index=False)

        print(f"Processed ratings, movies, links and users loaded in {data_directory}")

    except IOError as e:
        print(f"Error saving files: {e}")


if __name__ == "__main__":
    print("########## PREPROCESSING DATA ##########")
    raw_data_relative_path = os.path.join(my_project_directory, "data/raw/bronze")
    data_directory = os.path.join(my_project_directory, "data/raw/silver")
    # Chargement des données à partir du chemin spécifié
    try:
        df_ratings, df_movies, df_links = load_data(raw_data_relative_path)

        # Prétraitement des données
        if not any(df is None for df in [df_ratings, df_movies, df_links]):
            df_ratings = preprocessing_ratings(df_ratings)
            df_movies = preprocessing_movies(df_movies)
            df_links = preprocessing_links(df_links)

            # Création d'utilisateurs fictifs
            df_users = create_users()

            # Sauvegarde des données traitées dans le répertoire spécifié
            save_data(df_ratings, df_movies, df_links, df_users, data_directory)
        else:
            print("Une ou plusieurs DataFrames n'ont pas pu être chargées.")

    except Exception as e:
        print(f"An error occurred in the main execution flow: {e}")
