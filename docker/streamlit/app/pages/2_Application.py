import streamlit as st
import requests
from utils import display_movies_grid
import json

# Charger le CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Vérification plus robuste de l'authentification
if not st.session_state.get("is_logged_in", False):
    st.warning("Veuillez vous connecter pour accéder à cette page.")
    st.switch_page("pages/4_Authentification.py")
    st.stop()

# Utilisation des headers avec le token pour l'authentification
headers = {"Authorization": f"Bearer {st.session_state.token}"}

# Récupérer le token depuis la session
token = st.session_state.get("token")

response = requests.get("http://fastapi/", json={"token": token}, headers=headers)
result = response.json()
user_id = result["User"]["id"]
username = result["User"]["username"]
username = username.capitalize()

st.header(f"Bienvenue {username} !")

st.markdown("---")

# Récupérer les 3 films les mieux notés pour l'utilisateur
try:
    payload = {"userId": user_id}
    response = requests.post(
        "http://fastapi/predict/best_user_movies", json=payload, headers=headers
    )
    if response.status_code == 200:
        result_1 = response.json()
        len_result_1 = result_1["len"]
        imdb_dict_1 = result_1["data"]
        if imdb_dict_1:
            st.write(f"Voici vos {len_result_1} films les mieux notés :")
            display_movies_grid(result_1, 8)
        else:
            st.warning("Aucun film trouvé.")
    else:
        st.error(
            f"Erreur lors de la requête : {response.status_code} - {response.text}"
        )
except Exception as e:
    st.error(f"Erreur de requête: {str(e)}")

# Ajouter une ligne horizontale
st.markdown("---")

st.write("Voici une recommandation de films au regard de vos notations :")

# Récupérer les recommandations pour l'utilisateur
try:
    payload = {"userId": user_id}
    response = requests.post(
        "http://fastapi/predict/identified_user", json=payload, headers=headers
    )

    if response.status_code == 200:
        result_2 = response.json()
        if result_2:  # Vérifier que le résultat n'est pas vide
            display_movies_grid(result_2, 12)
        else:
            st.warning("Aucun film trouvé.")
    else:
        st.error(
            f"Erreur lors de la requête : {response.status_code} - {response.text}"
        )
except ValueError as e:
    st.error("Erreur de conversion de l'ID utilisateur")
    st.stop()
except Exception as e:
    st.error(f"Erreur de requête: {str(e)}")

# Ajouter une ligne horizontale
st.markdown("---")

st.markdown("---")

st.write(
    "Nous pouvons aussi vous faire des recommandations en relation avec un film. Entrez le nom d'un film que vous avez aimé et nous vous recommanderons des films similaires."
)

st.markdown("---")

# Demander à l'utilisateur de saisir le nom d'un film
movie_name = st.text_input("Entrez le nom d'un film que vous avez aimé", "Toy story")

# Dans la partie recherche de films similaires
if st.button("Rechercher"):
    payload = {"userId": user_id, "movie_title": movie_name}
    response = requests.post(
        "http://fastapi/predict/similar_movies", json=payload, headers=headers
    )

    if response.status_code == 200:
        api_result_3 = response.json()
        movie_find = api_result_3["movie_find"]
        if movie_find:
            st.write(
                f"Voici le film que nous avons trouvé au sein de notre base de donnée :"
            )
            single_movie_info = api_result_3["single_movie_info"]
            if single_movie_info:
                display_movies_grid(single_movie_info, 1)
                st.markdown("---")
        st.write("Voici nos recommandations de films similaires :")
        predict_movies_infos = api_result_3["predict_movies"]
        if predict_movies_infos:
            display_movies_grid(predict_movies_infos, 12)
        else:
            st.warning("Aucun film trouvé.")
    else:
        st.error(
            f"Erreur lors de la requête : {response.status_code} - {response.text}"
        )
