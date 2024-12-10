# Define namespaces for Kubernetes
NAMESPACE1 = api
NAMESPACE2 = airflow
NAMESPACE3 = mlflow

# Declare phony targets that do not correspond to files
.PHONY: help start-all start-minikube install-helm start-airflow start-mlflow start-api delete-pv-airflow check-kube change-namespace-api change-namespace-airflow change-namespace-mlflow clean-kube-api clean-kube-airflow clean-kube-mlflow clean-kube-all

# Help command to list all available targets
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  start-all     - Start all services"
	@echo "  start-minikube - Start Minikube with specified resources"
	@echo "  install-helm   - Install Helm package manager"
	@echo "  start-airflow  - Deploy Airflow using Helm"
	@echo "  start-mlflow   - Deploy MLflow using Helm"
	@echo "  start-api      - Deploy API services"
	@echo "  delete-pv-airflow - Delete persistent volumes for Airflow"
	@echo "  check-kube     - Verify kubectl is connected to a cluster"
	@echo "  change-namespace-* - Change current namespace context for Kubernetes"
	@echo "  clean-kube-*   - Clean up specific namespaces"

###### MAKEFILE KUBERNETES ######

# Start all services
start-all: start-airflow start-mlflow start-api

# Start Minikube with specified resources
start-minikube:
	minikube start --driver=docker --memory=8192 --cpus=4 --mount --mount-string="/home/antoine/jul24_cmlops_reco_film/:/host"

# Install Helm package manager
install-helm:
	curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
	chmod +x get_helm.sh
	./get_helm.sh

# Deploy Airflow using Helm
start-airflow:
	sudo apt-get update
	helm repo add apache-airflow https://airflow.apache.org
	helm upgrade --install airflow apache-airflow/airflow --namespace $(NAMESPACE2) --create-namespace -f kubernetes/airflow/my_airflow_values.yml

	# Apply persistent volumes and claims for Airflow
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-dags-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-logs-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/mlfow-storage-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-raw-init-folder-pv.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-dags-folder-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-logs-folder-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/mlfow-storage-pvc.yml
	kubectl apply -f kubernetes/persistent-volumes/airflow-local-raw-init-folder-pvc.yml
	kubectl apply -f kubernetes/secrets/airflow-secrets.yaml
	kubectl apply -f kubernetes/secrets/mlflow-secrets.yaml
	kubectl apply -f kubernetes/configmaps/airflow-configmaps.yml
	# Deploy pgAdmin service for managing PostgreSQL databases
	kubectl apply -f kubernetes/deployments/pgadmin-deployment.yml
	kubectl apply -f kubernetes/services/pgadmin-service.yml

# Deploy MLflow using Helm
start-mlflow:
	helm repo add bitnami https://charts.bitnami.com/bitnami
	helm repo update
	helm install mlf-ts bitnami/mlflow --namespace $(NAMESPACE3) --create-namespace

	kubectl apply -f kubernetes/services/mlflow-service.yml


# Deploy API services (FastAPI and Streamlit)
start-api:
	kubectl create namespace $(NAMESPACE1) || true # Avoid error if namespace already exists
	kubectl apply -f kubernetes/deployments/fastapi-deployment.yml
	kubectl apply -f kubernetes/deployments/streamlit-deployment.yml
	kubectl apply -f kubernetes/secrets/api-secrets.yaml
	kubectl apply -f kubernetes/services/api-service.yml

# Delete persistent volumes for Airflow (if they exist)
delete-pv-airflow:
	kubectl delete pv airflow-local-dags-folder-pv || true
	kubectl delete pv airflow-local-logs-folder-pv || true

# Check if kubectl is connected to a Kubernetes cluster
check-kube:
	@kubectl cluster-info > /dev/null 2>&1 || { echo "kubectl is not connected to a cluster"; exit 1; }

# Change the current namespace context for Kubernetes commands (API)
change-namespace-api:
	kubectl config set-context --current --namespace=$(NAMESPACE1)

# Change the current namespace context for Kubernetes commands (Airflow)
change-namespace-airflow:
	kubectl config set-context --current --namespace=$(NAMESPACE2)

# Change the current namespace context for Kubernetes commands (MLFlow)
change-namespace-mlflow:
	kubectl config set-context --current --namespace=$(NAMESPACE3)

# Clean up specific namespaces in Kubernetes (API)
clean-kube-api: check-kube
	kubectl delete namespace $(NAMESPACE1) || true

# Clean up specific namespaces in Kubernetes (Airflow)
clean-kube-airflow: check-kube
	kubectl delete namespace $(NAMESPACE2) || true

# Clean up specific namespaces in Kubernetes (MLFlow)
clean-kube-mlflow: check-kube
	kubectl delete namespace $(NAMESPACE3) || true

# Clean up all specified namespaces in Kubernetes
clean-kube-all: check-kube
	kubectl delete namespace $(NAMESPACE1) || true
	kubectl delete namespace $(NAMESPACE2) || true
	kubectl delete namespace $(NAMESPACE3) || true
