from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.kubernetes.secret import Secret


# Définition des secrets
secret_password = Secret(
    deploy_type="env",
    deploy_target="PASSWORD",
    secret="sql-conn"
)

secret_token = Secret(
    deploy_type="env",
    deploy_target="TMDB_TOKEN",
    secret="sql-conn"
)

# Définition des arguments par défaut
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),  # Commencer à s'exécuter à partir d'un jour en arrière
}

# Création du DAG principal
with DAG(
    dag_id='scrapping_TMDB_data',
    description='Scraping TMDB data and updating database',
    tags=['antoine'],
    default_args=default_args,
    schedule_interval='@daily',  # Exécution quotidienne à minuit
    catchup=False,
) as dag:

    # Tâche pour exécuter le script de transformation dans un pod Kubernetes
    dag_scraping_and_inserting_movies = KubernetesPodOperator(
        task_id="python_scrapping",
        image="antoinepela/projet_reco_movies:python-scrapping-latest",
        cmds=["python3", "scrapping.py"],
        namespace="airflow",
        env_vars={
            'DATABASE': 'postgres',
            'USER': 'postgres',
        },
        secrets=[secret_password, secret_token],  # Ajout des deux secrets
    )

dag_scraping_and_inserting_movies