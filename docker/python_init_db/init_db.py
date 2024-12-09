import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey

# Define the database connection parameters
conn_keys = ['conn_id', 'conn_type', 'host', 'login', 'password', 'schema']

def get_postgres_conn_conf():
    postgres_conn_conf = {}
    postgres_conn_conf['host'] = os.getenv("AIRFLOW_POSTGRESQL_SERVICE_HOST")
    postgres_conn_conf['port'] = os.getenv("AIRFLOW_POSTGRESQL_SERVICE_PORT")

    if postgres_conn_conf['host'] is None:
        raise TypeError("The AIRFLOW_POSTGRESQL_SERVICE_HOST isn't defined")
    elif postgres_conn_conf['port'] is None:
        raise TypeError("The AIRFLOW_POSTGRESQL_SERVICE_PORT isn't defined")

    postgres_conn_conf['conn_id'] = 'postgres'
    postgres_conn_conf['conn_type'] = 'postgres'
    postgres_conn_conf['login'] = 'postgres'
    postgres_conn_conf['password'] = 'postgres'
    postgres_conn_conf['schema'] = 'postgres'

    return postgres_conn_conf

# Define SQLAlchemy base and models
Base = declarative_base()

class Movie(Base):
    __tablename__ = 'movies'

    movieid = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    genres = Column(String)
    year = Column(Integer)

class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True)
    userid = Column(Integer)
    movieid = Column(Integer, ForeignKey('movies.movieid'))
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer)
    bayesian_mean = Column(Float, nullable=False)

class Link(Base):
    __tablename__ = 'links'

    id = Column(Integer, primary_key=True)
    movieid = Column(Integer, ForeignKey('movies.movieid'))
    imdbid = Column(Integer)
    tmdbid = Column(Integer)

class User(Base):
    __tablename__ = 'users'

    userid = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(300), nullable=False)

def create_tables(engine):
    Base.metadata.create_all(engine)
    print("Tables created")

def main():
    # Get PostgreSQL connection configuration
    postgres_conn_conf = get_postgres_conn_conf()

    # Create database URL
    db_url = f"postgresql://{postgres_conn_conf['login']}:{postgres_conn_conf['password']}@" \
             f"{postgres_conn_conf['host']}:{postgres_conn_conf['port']}/{postgres_conn_conf['schema']}"

    # Create a new SQLAlchemy engine
    engine = create_engine(db_url)

    # Create a new session
    Session = sessionmaker(bind=engine)

    # Create tables in the database
    create_tables(engine)

if __name__ == "__main__":
    main()