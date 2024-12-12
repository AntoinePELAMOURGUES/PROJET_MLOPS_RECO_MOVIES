from datetime import timedelta, datetime
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette import status
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
import os
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, CollectorRegistry
import time
import re
import psycopg2
import logging
from kubernetes import config

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration Kubernetes
config.load_incluster_config()

# Création d'un routeur pour gérer les routes d'authentification
router = APIRouter(
    prefix='/auth',  # Préfixe pour toutes les routes dans ce routeur
    tags=['auth']    # Tag pour la documentation
)
# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Clé secrète pour le JWT (à remplacer par une variable d'environnement en production)
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')  # Algorithme utilisé pour encoder le JWT

# Contexte de hachage pour le mot de passe
bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

# Définition du schéma de sécurité pour le token
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

# Modèle Pydantic pour la création d'un utilisateur
class CreateUserRequest(BaseModel):
    username: str  # Nom d'utilisateur
    email: str  # Adresse e-mail
    password: str  # Mot de passe

# Modèle Pydantic pour le token d'accès
class Token(BaseModel):
    access_token: str  # Le token d'accès
    token_type: str    # Type de token (généralement "bearer")

# Compteurs et histogrammes
collector = CollectorRegistry()

# Compteurs et histogrammes pour les métriques
user_creation_counter = Counter(
    name='user_creation_requests_total',
    documentation='Total number of user creation requests',
    labelnames=['status_code'],
    registry=collector
)

login_requests_counter = Counter(
    name='login_requests_total',
    documentation='Total number of login requests',
    labelnames=['status_code'],
    registry=collector
)

user_creation_duration_histogram = Histogram(
    name='user_creation_duration_seconds',
    documentation='Duration of user creation requests in seconds',
    labelnames=['status_code'],
    registry=collector
)

login_duration_histogram = Histogram(
    name='login_duration_seconds',
    documentation='Duration of login requests in seconds',
    labelnames=['status_code'],
    registry=collector
)

error_counter = Counter(
    name='user_creation_errors_total',
    documentation='Total number of user creation errors',
    labelnames=['error_type'],
    registry=collector
)

password_validation_counter = Counter(
    name='password_validation_errors_total',
    documentation='Total number of password validation errors',
    labelnames=['error_type'],
    registry=collector
)

email_validation_counter = Counter(
    name='email_validation_errors_total',
    documentation='Total number of email validation errors',
    labelnames=['error_type'],
    registry=collector
)

username_validation_counter = Counter(
    name='username_validation_errors_total',
    documentation='Total number of username validation errors',
    labelnames=['error_type'],
    registry=collector
)

def load_config():
    """Charge la configuration de la base de données à partir des variables d'environnement."""
    return {
        'host': os.getenv('POSTGRES_HOST'),
        'database': os.getenv('POSTGRES_DB'),
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD')
    }

def connect(config):
    """Connecte au serveur PostgreSQL et retourne la connexion."""
    try:
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Connection error: {error}")
        return None


# Fonction pour valider le nom d'utilisateur
def validate_username(username):
    if re.match("^[A-Za-z0-9_]+$", username) is None:
        return "Le nom d'utilisateur ne doit contenir que des lettres, chiffres et underscores."
    return None

def validate_email(email):
    # Expression régulière pour valider une adresse e-mail
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

    if re.match(email_regex, email) is None:
        return "L'adresse e-mail n'est pas valide."
    return None

# Fonction pour valider le mot de passe
def validate_password(password):
    if len(password) < 12:
        return "Le mot de passe doit contenir au moins 12 caractères."
    if not re.search(r"\d", password):
        return "Le mot de passe doit contenir au moins un chiffre."
    if not re.search(r"[A-Z]", password):
        return "Le mot de passe doit contenir au moins une lettre majuscule."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Le mot de passe doit contenir au moins un caractère spécial."
    return None

# Route pour créer un nouvel utilisateur
@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(create_user_request: CreateUserRequest):
    logger.info(f"Requête reçue création utilisateur: {create_user_request}")
    start_time = time.time()

    # Validation checks
    username_error = validate_username(create_user_request.username)
    if username_error:
        username_validation_counter.labels(error_type='invalid_username').inc()
        error_counter.labels(error_type='invalid_username').inc()
        logger.error(f"Erreur validation nom d'utilisateur: {username_error}")
        raise HTTPException(status_code=400, detail=username_error)

    mail_error = validate_email(create_user_request.email)
    if mail_error:
        email_validation_counter.labels(error_type='invalid_mail').inc()
        error_counter.labels(error_type='invalid_mail').inc()
        logger.error(f"Erreur validation email: {mail_error}")
        raise HTTPException(status_code=400, detail=mail_error)

    password_error = validate_password(create_user_request.password)
    if password_error:
        password_validation_counter.labels(error_type='invalid_password').inc()
        error_counter.labels(error_type='invalid_password').inc()
        logger.error(f"Erreur validation mot de passe: {password_error}")
        raise HTTPException(status_code=400, detail=password_error)

    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT email FROM users WHERE email = %s", (create_user_request.email,))
                if cur.fetchone() is not None:
                    error_counter.labels(error_type='username_already_registered').inc()
                    user_creation_counter.labels(status_code='400').inc()
                    logger.error("Email déjà enregistré")
                    raise HTTPException(status_code=400, detail="Email already registered")

                # Créer le nouvel utilisateur
                hached_password = bcrypt_context.hash(create_user_request.password)
                cur.execute("INSERT INTO users (username, email, hached_password) VALUES (%s, %s, %s)",
                            (create_user_request.username, create_user_request.email, hached_password,))
                conn.commit()
                logger.info("Utilisateur créé avec succès")

        except psycopg2.Error as e:
            error_counter.labels(error_type='database_error').inc()
            logger.error(f"Erreur base de données: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        finally:
            conn.close()  # Assurez-vous de fermer la connexion après utilisation

    else:
        logger.error("Échec de la connexion à la base de données.")
        raise HTTPException(status_code=500, detail="Database connection failed.")

    duration = time.time() - start_time
    user_creation_duration_histogram.labels(status_code='201').observe(duration)
    user_creation_counter.labels(status_code='201').inc()
    logger.info(f"Durée de création utilisateur: {duration} secondes")

# Route pour obtenir un token d'accès
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    logger.info(f"Requête reçue identification: {form_data}")
    start_time = time.time()
    user = await authenticate_user(form_data.username, form_data.password)
    logger.info(f"Retour fonction authenticate_user: {user}")
    if not user:
        login_requests_counter.labels(status_code='401').inc()
        logger.error("Échec de l'authentification: utilisateur non valide")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user.')

    token = create_access_token(user['email'], user['userId'], timedelta(minutes=30))
    duration = time.time() - start_time
    login_duration_histogram.labels(status_code='200').observe(duration)
    login_requests_counter.labels(status_code='200').inc()
    logger.info(f"Durée de connexion: {duration} secondes")

    # Modification de la structure de réponse pour garantir la cohérence
    response = {
        "access_token": token,
        "token_type": "bearer",
        "userId": int(user['userId']),  # Assurer que c'est un entier
        "username": user['username']
    }
    # Logging pour debug
    logger.info(f"Retour de la requête: {response}")
    return response

async def authenticate_user(email: str, password: str):
    config = load_config()
    conn = connect(config)

    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT userid, username, email, hached_password FROM users WHERE email = %s",
                    (email,)
                )
                user = cur.fetchone()

                if not user:
                    login_requests_counter.labels(status_code='404').inc()
                    logger.error("Utilisateur non trouvé")
                    raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

                if not bcrypt_context.verify(password, user[3]):
                    login_requests_counter.labels(status_code='401').inc()
                    logger.error("Mot de passe incorrect")
                    raise HTTPException(status_code=401, detail="Mot de passe incorrect")

                # Modification de la structure retournée pour garantir la cohérence
                user_data = {
                    'userId': int(user[0]),  # Conversion explicite en int
                    'username': user[1],
                    'email': user[2]
                }
                logger.info(f"Type de userId dans authenticate_user: {type(user_data['userId'])}")
                logger.info(f"Valeur de userId dans authenticate_user: {user_data['userId']}")
                return user_data

        except psycopg2.Error as e:
            error_counter.labels(error_type='database_error').inc()
            logger.error(f"Erreur base de données: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        finally:
            conn.close()  # Assurez-vous de fermer la connexion après utilisation
    else:
        logger.error("Échec de la connexion à la base de données.")
        raise HTTPException(status_code=500, detail="Database connection failed.")

# Fonction pour créer un token d'accès
def create_access_token(email: str, user_id: int, expires_delta: timedelta):
    encode = {'sub': email, 'id': user_id}  # Charge utile du token
    expires = datetime.utcnow() + expires_delta  # Définit la date d'expiration
    encode.update({'exp': expires})  # Ajoute la date d'expiration à la charge utile
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)  # Encode le token

# Fonction pour obtenir l'utilisateur actuel à partir du token
async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # Décode le token
        email: str = payload.get('sub')  # Récupère le nom d'utilisateur
        user_id: int = payload.get('id')  # Récupère l'ID de l'utilisateur
        config = load_config()
        conn = connect(config)

        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT userid, username FROM users WHERE userid = %s",
                    (user_id,)
                )
                user = cur.fetchone()
                username = user[1]
        if email is None or user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user.')  # Erreur si les donnees sont manquantes
        response = {'email': email, 'id': user_id, 'username': username}  # Retourne les données de l'utilisateur
        logger.info(f"Utilisateur actuel: {response}")
        return response # Retourne les données de l'utilisateur
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate user')  # Erreur si le token est invalide