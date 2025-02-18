from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s
import datetime

# Définition des secrets

# Définition des secrets
secret_password_airflow = Secret(
    deploy_type="env", deploy_target="POSTGRES_PASSWORD", secret="sql-conn"
)

# Définition des arguments par défaut
default_args = {
    "owner": "antoine",
    "start_date": datetime.datetime(2025, 2, 1),  # Date de début au 1er février 2025
    "retries": 1,
}

# Création du DAG principal
with DAG(
    dag_id="training_models",
    description="training models with MlFlow",
    tags=["antoine"],
    default_args=default_args,
    schedule_interval="0 0 1 * *",  # Exécution le 1er de chaque mois à minuit
    catchup=False,
) as dag:

    # Tâche pour exécuter le script de transformation dans un pod Kubernetes
    dag_training_models = KubernetesPodOperator(
        task_id="training_models",
        image="antoinepela/projet_reco_movies:train_models-latest",
        cmds=["python3", "train_models.py"],
        namespace="airflow",
        env_vars={
            "POSTGRES_HOST": "airflow-postgresql.airflow.svc.cluster.local",
            "POSTGRES_DB": "postgres",
            "POSTGRES_USER": "postgres",
        },
        secrets=[secret_password_airflow],
        volumes=[
            k8s.V1Volume(
                name="airflow-local-data-folder",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-local-data-folder"
                ),
            )
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="airflow-local-data-folder", mount_path="/root/mount_file"
            )
        ],
        is_delete_operator_pod=True,  # Supprimez le pod après exécution
        get_logs=True,  # Récupérer les logs du pod
        image_pull_policy="Always",  # Forcer le rechargement de l'image
    )

dag_training_models
