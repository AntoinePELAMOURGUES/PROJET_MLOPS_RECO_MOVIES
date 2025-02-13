<h1>RECOMMANDATION DE FILMS</h1>

![Project Logo](/img/reco_movies_project.jpeg)

<h2>:diamond_shape_with_a_dot_inside:Introduction</h2>
Dans le cadre de notre formation en Machine Learning Operations (MLOps) dispensée par Datascientest, nous avons l'opportunité d'explorer la mise en production et le monitoring d'une API dédiée à la recommandation de films. Ce projet vise à allier théorie et pratique en développant une solution robuste qui non seulement prédit les films susceptibles d'intéresser les utilisateurs, mais assure également un suivi constant de la performance de l'API.

<h2>:diamond_shape_with_a_dot_inside:Objectifs du Projet</h2>

- **Développement d'une API Performante** : Créer une API capable de fournir des recommandations de films basées sur des données utilisateurs et des modèles d'apprentissage automatique.
- **Mise en Production** : Déployer l'API dans un environnement de production pour garantir son accessibilité et sa fiabilité.
- **Monitoring Efficace** : Mettre en place des outils de surveillance pour suivre la disponibilité, les performances et les éventuelles erreurs de l'API, assurant ainsi une expérience utilisateur optimale.

<h2>:diamond_shape_with_a_dot_inside:Pourquoi ce Projet ?</h2>

Avec l'essor des services de streaming et la demande croissante pour des recommandations personnalisées, ce projet s'inscrit dans une tendance actuelle du marché. En intégrant des pratiques de MLOps, nous visons à garantir que notre solution soit non seulement fonctionnelle, mais aussi scalable et maintenable. Le monitoring joue un rôle crucial dans cette démarche, permettant d'anticiper les problèmes avant qu'ils n'affectent l'utilisateur final.

Ce repository est donc un témoignage de notre apprentissage et de notre capacité à développer des solutions innovantes dans le domaine du Machine Learning et des API. Nous vous invitons à explorer le code, à tester l'API et à contribuer à son amélioration.

<h2>:diamond_shape_with_a_dot_inside:Guide de démarrage rapide</h2>

Vous souhaitez déployer notre application de recommandation de films.

Suivez ces étapes simples pour lancer l'application en local.

## 🛠️ Prérequis

Assurez-vous d'avoir installé les éléments suivants sur votre machine :

1. [Python](https://www.python.org/) >= 3.9
2. [Docker Desktop](https://docs.docker.com/desktop/)
3. [Minikube](https://minikube.sigs.k8s.io/docs/start/)
4. [Helm](https://helm.sh/)

## :computer: Installation

### Clonage du repository

Clonnez ce repository sur votre machine locale :

```bash
git clone https://github.com/AntoinePELAMOURGUES/PROJET_MLOPS_RECO_MOVIES
```

## :wrench: Téléchargement et preprocessing des données initiales

### Instructions :

Avant de plonger dans l'aventure, nous vous invitons à créer un environnement isolé avec env ou conda. Cela vous permettra de travailler en toute sérénité et d'explorer sans limites !

1. Naviguez jusqu'à la racine du répertoire cloné :

```bash
cd {REPOSITORY}
```

2. Commencez par télécharger les bibliothèques nécessaires, puis personnalisez votre fichier .env :

```bash
make install-initial-data
```

3. Lancez le téléchargement des données initiales qui seront pré-traitées puis entrainez les modèles :

```bash
make preprocess-data
```

## 🚀 Déploiement via Kubernetes et cluster local Minikube

1. Lancement de minikube

Avant de lancer notre cluster local, rendez-vous dans le fichier Makefile afin de modifier la ligne 7 : PROJECT_DIRECTORY = "Chemin vers votre projet". Sauvegardez les modifications puis lancez minikube:

```bash
make start-minikube
```

2. Lancement de kubernetes & airflow & mlflow

```bash
make start-airflow
```

```bash
make start-mlflow
```

Une fois les déploiements, volumes, services lancés, entrez le code suivant afin de vous rendre sur l'interface d'Airflow (user et mdp = admin)

```bash
minikube service airflow-webserver -n airflow
```

Vous devriez voir apparaître un DAG nommé 'unique_dag_preprocess_data_to_db'. Ce DAG sera essentiel pour créer nos tables movies, ratings, links et users, et pour les alimenter avec les données prétraitées. Préparez-vous à donner vie à votre base de données !

Notez que vous pouvez accéder à Pgadmin afin de visualiser votre base de données:

```bash
minikube service pgadmin-service -n airflow
```

Configuration initiale de pgadmin:

- **user** : admin@pgadmin.org
- **password**: admin

Rendez-vous dans "Add New Server", entrez dans Général le nom que vous souhaitez puis dans Connection, remplissez les champs suivants :

- **Host name/address** : airflow-postgresql
- **Port** : 5432
- **Username** : postgres
- **Password** : postgres

## 🧠 Entrainement de nos modèles de Machine Learning

Vous pouvez maintenant lancer le DAG training_models pour entraîner un modèle TF-IDF afin d'extraire des caractéristiques des genres de films, ainsi qu'un modèle SVD pour la réduction de dimensionnalité et l'analyse des données.

3. Lancement de Fastapi, Streamlit

```bash
make start-api
```

## 👓 Visualisation de notre API

1. Visualisation de notre Api de recommandation:

```bash
minikube service streamlit -n api
```

2. Visualisation de fastapi:

```bash
minikube service fastapi -n api
```

## 👀 Monitoring des logs

```bash
make start-prometheus
```

## :skull: Arrêt de l'application

Rendez-vous dans le terminal puis tapez:

```bash
minikube stop
```
