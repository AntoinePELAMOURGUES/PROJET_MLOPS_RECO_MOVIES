# Définir les namespaces pour Kubernetes
NAMESPACE1 = api
NAMESPACE2 = airflow


# Répertoire du projet
PROJECT_DIRECTORY = /home/antoine/PROJET_MLOPS_RECO_MOVIES


# Déclarer les phony targets qui ne correspondent pas à des fichiers
.PHONY: help start-all start-minikube install-helm start-airflow start-mlflow start-api delete-pv-airflow check-kube change-namespace-api change-namespace-airflow change-namespace-mlflow clean-kube-api clean-kube-airflow clean-kube-mlflow clean-kube-all install-initial-data preprocess-data start-prometheus

# Commande d'aide pour lister toutes les targets disponibles
help:
	@echo "Utilisation: make [cible]"
	@echo "Cibles:"
	@echo "  start-all     - Démarrer tous les services"
	@echo "  start-minikube - Démarrer Minikube avec les ressources spécifiées"
	@echo "  install-helm   - Installer le gestionnaire de paquets Helm"
	@echo "  start-airflow  - Déployer Airflow en utilisant Helm"
	@echo "  start-mlflow   - Déployer MLflow en utilisant Helm"
	@echo "  start-api      - Déployer les services API"
	@echo "  delete-pv-airflow - Supprimer les volumes persistants pour Airflow"
	@echo "  check-kube     - Vérifier que kubectl est connecté à un cluster"
	@echo "  change-namespace-* - Changer le contexte d'espace de noms actuel pour Kubernetes"
	@echo "  clean-kube-*   - Nettoyer les espaces de noms spécifiques"

##### MAKEFILE INITIAL DATA PIPELINE ######

# Installer les dépendances Python - télécharger et prétraiter les données
install-initial-data:
	sudo apt-get update
	sudo apt-get install build-essential python3-dev
	cd initial_data_pipeline && pip install -r requirements.txt
	cd initial_data_pipeline && cp .env.example .env
	echo "Veuillez remplir le fichier .env avec votre project_directory avant de continuer"

preprocess-data:
	cd initial_data_pipeline && python recover_data.py
	cd initial_data_pipeline && python build_features.py
	cd initial_data_pipeline && python initial_train_models.py

###### MAKEFILE KUBERNETES ######

# Démarrer tous les services
start-all: start-minikube start-airflow start-mlflow start-api

# Démarrer Minikube avec les ressources spécifiées
start-minikube:
	minikube start --driver=docker --memory=18000 --cpus=4 --mount --mount-string="$(PROJECT_DIRECTORY):/host"

# Installer le gestionnaire de paquets Helm
install-helm:
	curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
	chmod +x get_helm.sh
	./get_helm.sh

# Déployer Airflow en utilisant Helm
start-airflow:
	sudo apt-get update
	helm repo add apache-airflow https://airflow.apache.org
	helm upgrade --install airflow apache-airflow/airflow --namespace $(NAMESPACE2) --create-namespace -f kubernetes/airflow/my_airflow_values.yml

	# Appliquer les volumes persistants et les revendications (claims) pour Airflow
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-dags-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-logs-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-data-init-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/my-db-backup-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-dags-folder-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-logs-folder-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-data-init-folder-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/my-db-backup-pvc.yml
	kubectl apply -f kubernetes/configmaps/airflow-configmaps.yml
	kubectl apply -f kubernetes/deployments/pgadmin-deployment.yml
	kubectl apply -f kubernetes/services/pgadmin-service.yml
	kubectl apply -f kubernetes/secrets/airflow-secrets.yaml

# Déployer MLflow en utilisant Helm
start-mlflow:
	helm repo add bitnami https://charts.bitnami.com/bitnami
	helm repo update
	@echo "Veuillez entrer votre identifiant :"
	@read USERNAME; \
	echo "Veuillez entrer votre mot de passe :"; \
	read -s PASSWORD; \
	echo "Création du secret Kubernetes..."; \
	kubectl create secret generic mlf-ts-mlflow-tracking --namespace airflow \
		--from-literal=admin-user=$$USERNAME \
		--from-literal=admin-password=$$PASSWORD; \
	echo "Secret créé avec succès."
	helm install mlf-ts bitnami/mlflow --namespace $(NAMESPACE2) --create-namespace -f kubernetes/ml_flow/values.yaml
	kubectl apply -f kubernetes/secrets/mlflow-secrets.yaml

# Déployer les services API (FastAPI et Streamlit)
start-api:
	kubectl create namespace $(NAMESPACE1) || true
	kubectl apply -f kubernetes/secrets/api-secrets.yaml
	kubectl apply -f kubernetes/deployments/fastapi-deployment.yml
	kubectl apply -f kubernetes/deployments/streamlit-deployment.yml
	kubectl apply -f kubernetes/services/api-service.yml

# Supprimer les volumes persistants pour Airflow (s'ils existent)
delete-pv-airflow:
	kubectl delete pv airflow-local-dags-folder-pv || true
	kubectl delete pv airflow-local-logs-folder-pv || true

# Vérifier si kubectl est connecté à un cluster Kubernetes
check-kube:
	@kubectl cluster-info > /dev/null 2>&1 || { echo "kubectl n'est pas connecté à un cluster"; exit 1; }

# Changer le contexte d'espace de noms actuel pour les commandes Kubernetes (API)
change-namespace-api:
	kubectl config set-context --current --namespace=$(NAMESPACE1)

# Changer le contexte d'espace de noms actuel pour les commandes Kubernetes (Airflow)
change-namespace-airflow:
	kubectl config set-context --current --namespace=$(NAMESPACE2)

# Nettoyer les espaces de noms spécifiques dans Kubernetes (API)
clean-kube-api: check-kube
	kubectl delete namespace $(NAMESPACE1) || true

# Nettoyer les espaces de noms spécifiques dans Kubernetes (Airflow)
clean-kube-airflow: check-kube
	kubectl delete namespace $(NAMESPACE2) || true

# Nettoyer tous les espaces de noms spécifiés dans Kubernetes
clean-kube-all: check-kube
	kubectl delete namespace $(NAMESPACE1) || true
	kubectl delete namespace $(NAMESPACE2) || true

start-service:
	minikube service airflow-webserver -n airflow
	minikube service mlflow -n airflow
	minikube service fastapi -n api
	minikube service streamlit -n api
	minikube service pgadmin -n airflow
