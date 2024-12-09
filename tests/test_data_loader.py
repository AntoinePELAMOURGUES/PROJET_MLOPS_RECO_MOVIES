import unittest
import pandas as pd
import os
import importlib.util
import sys



# Définir le chemin du fichier à tester
module_file_path = './kubernetes/airflow/order/docker/prod/python_transform/data_loader.py'

# Charger le module
spec = importlib.util.spec_from_file_location("data_loader", module_file_path)
data_loader = importlib.util.module_from_spec(spec)
sys.modules["data_loader"] = data_loader
spec.loader.exec_module(data_loader)


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        self.config = data_loader.load_config()  # Chargez votre configuration ici

    def test_load_config(self):
        self.assertIsNotNone(self.config['host'])
        self.assertIsNotNone(self.config['database'])
        self.assertIsNotNone(self.config['user'])
        self.assertIsNotNone(self.config['password'])

    def test_execute_query(self):
        query = "SELECT 1;"  # Exemple simple qui devrait toujours réussir
        result = data_loader.execute_query_psql(query, self.config)
        self.assertEqual(result, 1)  #

    def test_upsert_to_psql(self):
        # Créez un DataFrame d'exemple pour tester l'upsert
        df_test = pd.DataFrame({
            'id': [1],
            'userId': [1],
            'movieId': [1],
            'rating': [5],
            'timestamp': [1234567890],
            'bayesian_mean': [4.5]
        })
        data_loader.upsert_to_psql(data_loader.table_ratings, df_test)  #

if __name__ == "__main__":
    unittest.main()