import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from .main import app  # Assurez-vous que votre application FastAPI est importée correctement

client = TestClient(app)

# Test pour la route /predict/best_user_movies
@pytest.mark.asyncio
async def test_best_user_movies():
    user_data = {"userId": 1}

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict/best_user_movies", json=user_data)

    assert response.status_code == 200
    assert isinstance(response.json(), dict)

# Test pour la route /predict/identified_user
@pytest.mark.asyncio
async def test_identified_user():
    user_data = {"userId": 1}

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict/identified_user", json=user_data)

    assert response.status_code == 200
    assert isinstance(response.json(), dict)

# Test pour la route /predict/similar_movies
@pytest.mark.asyncio
async def test_similar_movies():
    movie_data = {"movie_title": "Inception"}

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/predict/similar_movies", json=movie_data)

    assert response.status_code == 200
    assert isinstance(response.json(), dict)
