import os
import requests

def download_and_save_file(url, raw_data_relative_path):
    """
    Télécharge les fichiers CSV depuis l'URL donnée et les enregistre dans le chemin spécifié.

    Args:
        url (str): L'URL de base pour télécharger les fichiers.
        raw_data_relative_path (str): Chemin relatif où les fichiers seront enregistrés.
    """
    filenames = ['links.csv', 'movies.csv', 'ratings.csv']
    os.makedirs(raw_data_relative_path, exist_ok=True)  # Crée le répertoire si nécessaire

    for filename in filenames:
        data_url = os.path.join(url, filename)
        try:
            response = requests.get(data_url)
            response.raise_for_status()  # Assure que la requête a réussi
            print(f"Downloading {filename} from {data_url}")
            file_path = os.path.join(raw_data_relative_path, filename)
            with open(file_path, 'wb') as file:
                file.write(response.content)  # Écrit le contenu dans le fichier
            print(f"File saved to {file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
        except IOError as e:
            print(f"Error saving {filename}: {e}")

# ...existing code...
if __name__ == "__main__":
    raw_data_relative_path="/root/mount_file/bronze"
    bucket_folder_url="https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"
    download_and_save_file(url=bucket_folder_url, raw_data_relative_path=raw_data_relative_path)