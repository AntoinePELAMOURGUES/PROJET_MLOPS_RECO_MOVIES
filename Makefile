SHELL := /bin/bash

# Définir les namespaces pour Kubernetes

NAMESPACE = airflow


# Répertoire du projet
PROJECT_DIRECTORY = /home/antoine/PROJET_MLOPS_RECO_MOVIES


# Déclarer les phony targets qui ne correspondent pas à des fichiers
.PHONY: help start-all start-minikube install-helm start-airflow start-mlflow start-api delete-pv-airflow check-kube change-namespace-api change-namespace-airflow change-namespace-mlflow clean-kube-api clean-kube-airflow clean-kube-mlflow clean-kube-all install-initial-data preprocess-data start-monitoring

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
	@echo "  start-monitoring - Déployer les services de monitoring"
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

###### MAKEFILE KUBERNETES ######

# Démarrer tous les services
start-all: start-minikube start-airflow start-mlflow start-api

# Démarrer Minikube avec les ressources spécifiées
start-minikube:
	minikube start --driver=docker --memory=9000 --cpus=4 --mount --mount-string="$(PROJECT_DIRECTORY):/host" --bootstrapper=kubeadm --extra-config=kubelet.authentication-token-webhook=true --extra-config=kubelet.authorization-mode=Webhook --extra-config=scheduler.bind-address=0.0.0.0 --extra-config=controller-manager.bind-address=0.0.0.0
	minikube addons disable metrics-server

# Installer le gestionnaire de paquets Helm
install-helm:
	curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
	chmod +x get_helm.sh
	./get_helm.sh

# Déployer Airflow en utilisant Helm
start-airflow:
	sudo apt-get update
	helm repo add apache-airflow https://airflow.apache.org
	helm upgrade --install airflow apache-airflow/airflow --namespace $(NAMESPACE) --create-namespace -f kubernetes/airflow/my_airflow_values.yml

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
	kubectl apply -f kubernetes/services/airflow-service.yml
	kubectl apply -f kubernetes/services/pgadmin-service.yml
	kubectl apply -f kubernetes/secrets/airflow-secrets.yaml


# Déployer les services API (FastAPI et Streamlit)
start-api:
	kubectl apply -f kubernetes/deployments/fastapi-deployment.yml
	kubectl apply -f kubernetes/deployments/streamlit-deployment.yml
	kubectl apply -f kubernetes/services/api-service.yml

start-monitoring:
	helm install prometheus prometheus-community/kube-prometheus-stack --namespace airflow --create-namespace --set grafana.service.type=NodePort --set promotheus.service.type=NodePort



# Supprimer les volumes persistants pour Airflow (s'ils existent)
delete-pv-airflow:
	kubectl delete pv airflow-local-dags-folder-pv || true
	kubectl delete pv airflow-local-logs-folder-pv || true

# Vérifier si kubectl est connecté à un cluster Kubernetes
check-kube:
	@kubectl cluster-info > /dev/null 2>&1 || { echo "kubectl n'est pas connecté à un cluster"; exit 1; }

# Changer le contexte d'espace de noms actuel pour les commandes Kubernetes (Airflow)
change-namespace-airflow:
	kubectl config set-context --current --namespace=$(NAMESPACE)


# Nettoyer les espaces de noms spécifiques dans Kubernetes (Airflow)
clean-kube-airflow: check-kube
	kubectl delete namespace $(NAMESPACE) || true

