from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s


# Définition des secrets
secret_password = Secret(
    deploy_type="env",
    deploy_target="POSTGRES_PASSWORD",
    secret="sql-conn"
)


# Définition des arguments par défaut
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),  # Commencer à s'exécuter à partir d'un jour en arrière
}

# Création du DAG principal
with DAG(
    dag_id='db_to_csv',
    description='création des fichiers csv à partir de la base de données',
    tags=['antoine'],
    default_args=default_args,
    schedule_interval='0 11 */6 * *',  # Exécution tout les 3 jours à 6:00
    catchup=False,
) as dag:

    # Tâche pour exécuter le script de transformation dans un pod Kubernetes
    dag_db_to_csv = KubernetesPodOperator(
        task_id="database_to_csv",
        image="antoinepela/projet_reco_movies:python-db-to-csv-latest",
        cmds=["python3", "db_to_csv.py"],
        namespace="airflow",
        env_vars={
            'POSTGRES_HOST': "airflow-postgresql.airflow.svc.cluster.local",
            'POSTGRES_DB': 'postgres',
            'POSTGRES_USER': 'postgres',
        },
        secrets=[secret_password],  # Ajout des deux secrets
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
        get_logs=True,          # Récupérer les logs du pod
        image_pull_policy='Always',  # Forcer le rechargement de l'image
    )

dag_db_to_csv