from surprise import Dataset, Reader
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle


def train_SVD(ratings_df, movies_df):

    df = pd.merge(
        ratings_df, movies_df[["movieId", "genres"]], on="movieId", how="left"
    )

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    mlb = MultiLabelBinarizer()

    df["userId"] = user_encoder.fit_transform(df["userId"])
    df["movieId"] = movie_encoder.fit_transform(df["movieId"])

    df = df.join(
        pd.DataFrame(
            mlb.fit_transform(df.pop("genres").str.split("|")),
            columns=mlb.classes_,
            index=df.index,
        )
    )

    df.drop(columns="(no genres listed)", inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.5)

    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(train_df[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()
    model_svd = SVD()
    model_svd.fit(trainset)
    predictions_svd = model_svd.test(trainset.build_anti_testset())
    accuracy.rmse(predictions_svd)
    with open(
        "/home/antoine/PROJET_MLOPS_RECO_MOVIES/data/models/model_SVD_1.pkl", "wb"
    ) as f:
        pickle.dump(model_svd, f)
    return model_svd


if __name__ == "__main__":
    ratings_df = pd.read_csv(
        "/home/antoine/PROJET_MLOPS_RECO_MOVIES/data/raw/bronze/ratings.csv"
    )
    movies_df = pd.read_csv(
        "/home/antoine/PROJET_MLOPS_RECO_MOVIES/data/raw/bronze/movies.csv"
    )
    train_SVD(ratings_df, movies_df)
