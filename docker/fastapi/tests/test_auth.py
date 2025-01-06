import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from .main import app  # Assurez-vous que votre application FastAPI est importée correctement

client = TestClient(app)

# Test pour la création d'utilisateur
@pytest.mark.asyncio
async def test_create_user():
    # Données de test pour la création d'utilisateur
    user_data = {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "TestPassword123!"
    }

    # Envoyer une requête POST pour créer un utilisateur
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/auth/", json=user_data)

    # Vérifier que la réponse a un statut 201 CREATED
    assert response.status_code == 201

# Test pour l'authentification et l'obtention d'un token
@pytest.mark.asyncio
async def test_login_for_access_token():
    # Données de test pour l'authentification
    login_data = {
        "username": "testuser",
        "password": "TestPassword123!"
    }

    # Envoyer une requête POST pour obtenir un token
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/auth/token", data=login_data)

    # Vérifier que la réponse a un statut 200 OK
    assert response.status_code == 200
    # Vérifier que le token est présent dans la réponse
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

# Test pour obtenir l'utilisateur actuel à partir du token
@pytest.mark.asyncio
async def test_get_current_user():
    # Données de test pour l'authentification
    login_data = {
        "username": "testuser",
        "password": "TestPassword123!"
    }

    # Obtenir un token d'accès
    async with AsyncClient(app=app, base_url="http://test") as ac:
        login_response = await ac.post("/auth/token", data=login_data)
        token = login_response.json()["access_token"]

    # Envoyer une requête GET pour obtenir l'utilisateur actuel
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/auth/users/me", headers=headers)

    # Vérifier que la réponse a un statut 200 OK
    assert response.status_code == 200
    # Vérifier que les informations de l'utilisateur sont correctes
    assert response.json()["email"] == "testuser@example.com"
    assert response.json()["username"] == "testuser"
