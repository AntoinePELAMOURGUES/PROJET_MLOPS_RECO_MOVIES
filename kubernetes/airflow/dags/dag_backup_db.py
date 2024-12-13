from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s

secret_password = Secret(
    deploy_type="env",
    deploy_target="POSTGRES_PASSWORD",
    secret="my-api-postgres-secrets"
)

with DAG(
  dag_id='backup_database',
  tags=['antoine'],
  default_args={
    'owner': 'airflow',
    'start_date': days_ago(0, minute=1),
    },
  schedule_interval='0 0 * * 1',  # Exécution tous les lundis à 00:00
  catchup=False
) as dag:

    backup_db = KubernetesPodOperator(
    task_id="backup_database",
    image="antoinepela/projet_reco_movies:python-backup-db-latest",
    cmds=["python3", "backup_db.py"],
    namespace= "airflow",
    env_vars={
            'POSTGRES_HOST': "my-api-postgres.airflow.svc.cluster.local",
            'POSTGRES_DB': 'my-api-database',
            'POSTGRES_USER': 'antoine',
        },
    secrets= [secret_password],
    volumes=[
        k8s.V1Volume(
            name="my-api-postgres-pv",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="my-api-postgres-pvc")
        )
    ],
    volume_mounts=[
        k8s.V1VolumeMount(
            name="my-api-postgres-pv",
            mount_path="/root/mount_file/backup_db"
        )
    ],  # Chemin où les modèles seront sauvegardés.
    is_delete_operator_pod=True,  # Supprimez le pod après exécution
    get_logs=True,          # Récupérer les logs du pod
    image_pull_policy='Always',  # Forcer le rechargement de l'image
)


backup_db