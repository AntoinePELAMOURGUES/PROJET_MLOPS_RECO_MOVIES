from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s

secret_password = Secret(
    deploy_type="env", deploy_target="POSTGRES_PASSWORD", secret="sql-conn"
)

mlflow_password = Secret(
    deploy_type="env", deploy_target="MLFLOW_TRACKING_PASSWORD", secret="mlflow-secrets"
)

with DAG(
    dag_id="first_train_models",
    tags=["antoine"],
    default_args={
        "owner": "airflow",
        "start_date": days_ago(0, minute=1),
    },
    schedule_interval=None,  # Pas de planification automatique
    catchup=False,
) as dag:

    python_train_models = KubernetesPodOperator(
        task_id="first_train_models",
        name="first-train-models-pod",
        image="antoinepela/projet_reco_movies:first_train_models-latest",
        cmds=["python3", "first_train_models.py"],
        namespace="airflow",
        env_vars={
            "POSTGRES_HOST": "airflow-postgresql",
            "POSTGRES_DB": "postgres",
            "POSTGRES_USER": "postgres",
            "MLFLOW_TRACKING_URI": "http://mlf-ts-mlflow-tracking.mlflow.svc.cluster.local",
            "MLFLOW_TRACKING_USERNAME": "user",
            "GIT_PYTHON_REFRESH": "quiet",
        },
        secrets=[secret_password, mlflow_password],
        volumes=[
            k8s.V1Volume(
                name="models-mlflow-pv",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="models-mlflow-pvc"
                ),
            ),
            k8s.V1Volume(
                name="airflow-local-raw-folder",
                persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                    claim_name="airflow-local-raw-folder"
                ),
            ),
        ],
        volume_mounts=[
            k8s.V1VolumeMount(
                name="models-mlflow-pv", mount_path="/root/mount_file/models"
            ),
            k8s.V1VolumeMount(
                name="airflow-local-raw-folder", mount_path="/root/mount_file/raw"
            ),
        ],  # Chemin où les modèles seront sauvegardés.
        is_delete_operator_pod=True,  # Supprimez le pod après exécution
        get_logs=True,  # Récupérer les logs du pod
        image_pull_policy="Always",  # Forcer le rechargement de l'image
        container_resources=k8s.V1ResourceRequirements(
            limits={"memory": "25Gi", "cpu": "4"},  # Limites de ressources
            requests={"memory": "12Gi", "cpu": "2"},  # Demandes de ressources
        ),
    )


python_train_models
