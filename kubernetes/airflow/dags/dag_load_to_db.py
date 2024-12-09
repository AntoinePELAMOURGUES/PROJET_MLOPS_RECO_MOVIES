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
  dag_id='load_to_db',
  tags=['antoine'],
  default_args={
    'owner': 'airflow',
    'start_date': days_ago(1),
    },
  catchup=False
) as dag:

    python_load = KubernetesPodOperator(
    task_id="load_to_db",
    image="antoinepela/projet_reco_movies:python-load-data-latest",
    cmds=["python3", "data_to_db.py"],
    namespace= "airflow",
    env_vars={
            'DATABASE': 'postgres',
            'USER': 'postgres',
        },
    secrets= [secret_password],
    volumes=[
        k8s.V1Volume(
            name="airflow-local-raw-folder",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="airflow-local-raw-folder")
        )
    ],
    volume_mounts=[
        k8s.V1VolumeMount(
            name="airflow-local-raw-folder",
            mount_path="/raw"
        )
    ],
    ),

python_load