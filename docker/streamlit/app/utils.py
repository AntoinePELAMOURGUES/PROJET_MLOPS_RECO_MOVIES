import streamlit as st


def display_movies_grid(movies_info, len_movies):
    if len_movies == 1:
        rows = [st.columns(1) for _ in range(1)]
    if len_movies < 9:
        # Créer deux lignes principales et 4 colonnes pour chaque ligne
        rows = [st.columns(4) for _ in range(2)]
    else:
        # Créer trois lignes principales et 4 colonnes pour chaque ligne
        rows = [st.columns(4) for _ in range(3)]

    # Diviser les films entre les lignes et colonnes
    for idx, movie_info in movies_info.items():
        idx = int(idx)
        row_idx = idx // 4  # Déterminer la ligne (0, 1 ou 2)
        col_idx = idx % 4  # Déterminer la colonne (0 à 3)

        # Vérifier que nous ne dépassons pas le nombre de lignes
        if row_idx < len(rows):
            with rows[row_idx][col_idx]:
                html_content = f"""
                <div class="movie-container">
                    <div class="movie-tile">
                        <img src="{movie_info['poster_path']}" alt="{movie_info['title']}">
                    </div>
                    <div class="overlay">
                        <h5 style="color: white;">{movie_info['title']}</h5>
                        <p class="rating-badge">{round(movie_info['vote_average'], 1)} ⭐️</p>
                    </div>
                </div>
                """
                st.markdown(html_content, unsafe_allow_html=True)
