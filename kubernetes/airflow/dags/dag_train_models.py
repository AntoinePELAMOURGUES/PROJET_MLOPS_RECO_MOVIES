from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret
from kubernetes.client import models as k8s

secret_password  = Secret(
  deploy_type="env",
  deploy_target="PASSWORD",
  secret="sql-conn"
)

with DAG(
  dag_id='first_train_models',
  tags=['antoine'],
  default_args={
    'owner': 'airflow',
    'start_date': days_ago(0, minute=1),
    },
  schedule_interval=None,  # Pas de planification automatique
  catchup=False
) as dag:

    python_transform = KubernetesPodOperator(
    task_id="train_models",
    image="antoinepela/projet_reco_movies:train_models-latest",
    cmds=["python3", "train_models.py"],
    namespace= "airflow",
    env_vars={
            'DATABASE': 'postgres',
            'USER': 'postgres',
            'MLFLOW_TRACKING_URI': 'http://mlf-ts-mlflow-tracking.mlflow.svc.cluster.local',
            'MLFLOW_TRACKING_USERNAME': 'user',
            'MLFLOW_TRACKING_PASSWORD': 'Nds70uprI3',
        },
    secrets= [secret_password],
    volumes=[
        k8s.V1Volume(
            name="models-storage-pv",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="models-storage-pvc")
        )
    ],
    volume_mounts=[
        k8s.V1VolumeMount(
            name="models-storage-pv",
            mount_path="/root/mount_file"
        )
    ],  # Chemin où les modèles seront sauvegardés.
    is_delete_operator_pod=True,  # Supprimez le pod après exécution
    get_logs=True,          # Récupérer les logs du pod
    image_pull_policy='Always',  # Forcer le rechargement de l'image
)


python_transform