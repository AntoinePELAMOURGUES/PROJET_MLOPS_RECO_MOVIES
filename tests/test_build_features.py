import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from initial_data_pipeline.build_features import (
    load_data,
    preprocessing_ratings,
    preprocessing_movies,
    preprocessing_links,
    create_users,
)


@pytest.fixture
def mock_csv_files(tmp_path):
    """Fixture to create mock CSV files for testing."""
    df_ratings = pd.DataFrame(
        {"userId": [1, 2, 3], "movieId": [101, 102, 103], "rating": [5, 4, 3]}
    )
    df_movies = pd.DataFrame(
        {
            "movieId": [101, 102, 103],
            "title": ["Movie A (2020)", "Movie B (2021)", "Movie C (2022)"],
            "genres": ["Action|Drama", "Comedy|Romance", "Thriller"],
        }
    )
    df_links = pd.DataFrame(
        {
            "movieId": [101, 102, 103],
            "imdbId": ["tt1234567", "tt2345678", "tt3456789"],
            "tmdbId": [1001, 1002, 1003],
        }
    )

    # Save the DataFrames to CSV files in the temporary directory
    df_ratings.to_csv(tmp_path / "ratings.csv", index=False)
    df_movies.to_csv(tmp_path / "movies.csv", index=False)
    df_links.to_csv(tmp_path / "links.csv", index=False)

    return str(tmp_path)


def test_load_data(mock_csv_files):
    df_ratings, df_movies, df_links = load_data(mock_csv_files)

    assert not df_ratings.empty
    assert not df_movies.empty
    assert not df_links.empty


def test_preprocessing_ratings(mock_csv_files):
    df_ratings = pd.read_csv(os.path.join(mock_csv_files, "ratings.csv"))

    processed_ratings = preprocessing_ratings(df_ratings)

    assert "bayesian_mean" in processed_ratings.columns


def test_preprocessing_movies(mock_csv_files):
    df_movies = pd.read_csv(os.path.join(mock_csv_files, "movies.csv"))

    processed_movies = preprocessing_movies(df_movies)

    assert "year" in processed_movies.columns
    assert all(isinstance(x, str) for x in processed_movies["genres"])


def test_preprocessing_links(mock_csv_files):
    df_links = pd.read_csv(os.path.join(mock_csv_files, "links.csv"))

    processed_links = preprocessing_links(df_links)

    assert "tmdbid" in processed_links.columns
    assert "imdbid" in processed_links.columns


def test_create_users():
    users_df = create_users()

    assert len(users_df) == 500
    assert all(users_df["username"].str.startswith("user"))
