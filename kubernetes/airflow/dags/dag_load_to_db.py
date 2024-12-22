from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s

# Définition du secret pour la connexion à la base de données
secret_password = Secret(
    deploy_type="env", deploy_target="POSTGRES_PASSWORD", secret="sql-conn"
)

# Définition du DAG
with DAG(
    dag_id="unique_dag_preprocess_data_to_db",
    tags=["antoine"],
    default_args={
        "owner": "airflow",
        "start_date": days_ago(1),
    },
    schedule_interval=None,  # Pas de planification automatique
    catchup=False,  # Ne pas exécuter les exécutions passées
) as dag:

    python_init_db = KubernetesPodOperator(
        task_id="init_db",
        image="antoinepela/projet_reco_movies:python-init-db-latest",
        cmds=["python3", "init_db.py"],
        namespace="airflow",
        env_vars={
            "POSTGRES_HOST": "airflow-postgresql.airflow.svc.cluster.local",
            "POSTGRES_DB": "postgres",
            "POSTGRES_USER": "postgres",
        },
        secrets=[secret_password],
        is_delete_operator_pod=True,  # Supprimez le pod après exécution
        get_logs=True,  # Récupérer les logs du pod
        image_pull_policy="Always",  # Forcer le rechargement de l'image
    )

    # Tâche pour charger les données prétraitées dans la base de données
    python_load_to_db = KubernetesPodOperator(
        task_id="load_to_db",
        image="antoinepela/projet_reco_movies:python-load-data-latest",
        cmds=["python3", "data_to_db.py"],
        namespace="airflow",
        env_vars={
            "POSTGRES_HOST": "airflow-postgresql.airflow.svc.cluster.local",
            "POSTGRES_DB": "postgres",
            "POSTGRES_USER": "postgres",
        },
        secrets=[secret_password],
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
        image_pull_policy="Always",
    )
    python_load_to_db.doc_md = """
    ### Chargement des Données dans la Base de Données
    Cette tâche exécute un script Python (`data_to_db.py`) qui charge les caractéristiques
    prétraitées dans la base de données PostgreSQL.
    """

# Définition de l'ordre d'exécution des tâches
python_init_db >> python_load_to_db
