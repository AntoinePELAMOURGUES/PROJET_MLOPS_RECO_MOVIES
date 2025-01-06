# Utiliser une image de base Python
FROM python:3.12

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && \
    apt-get install -y python3-distutils gcc g++ && \
    apt-get clean

# Copier le fichier requirements.txt
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copier le reste de l'application
COPY . .


# Commande par défaut pour exécuter l'application (ajustez selon vos besoins)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]