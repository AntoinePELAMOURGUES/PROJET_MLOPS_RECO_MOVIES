
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from docker.python_init_db.init_db import Base, Movie, Rating, Link, User, create_tables

@pytest.fixture(scope='module')
def engine():
    # Utiliser une base de données en mémoire pour les tests
    engine = create_engine('sqlite:///:memory:')
    create_tables(engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope='module')
def session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_tables(engine):
    # Vérifier que les tables ont été créées
    inspector = engine.inspect()
    assert 'movies' in inspector.get_table_names()
    assert 'ratings' in inspector.get_table_names()
    assert 'links' in inspector.get_table_names()
    assert 'users' in inspector.get_table_names()

def test_insert_movie(session):
    new_movie = Movie(movieid=1, title="Test Movie", genres="Action", year=2021)
    session.add(new_movie)
    session.commit()
    movie = session.query(Movie).filter_by(movieid=1).first()
    assert movie is not None
    assert movie.title == "Test Movie"

def test_insert_rating(session):
    new_rating = Rating(id=1, userid=1, movieid=1, rating=5.0, timestamp=1234567890, bayesian_mean=4.5)
    session.add(new_rating)
    session.commit()
    rating = session.query(Rating).filter_by(id=1).first()
    assert rating is not None
    assert rating.rating == 5.0

def test_insert_link(session):
    new_link = Link(id=1, movieid=1, imdbid=123456, tmdbid=654321)
    session.add(new_link)
    session.commit()
    link = session.query(Link).filter_by(id=1).first()
    assert link is not None
    assert link.imdbid == 123456

def test_insert_user(session):
    new_user = User(userid=1, username="testuser", email="testuser@example.com", hashed_password="hashedpassword")
    session.add(new_user)
    session.commit()
    user = session.query(User).filter_by(userid=1).first()
    assert user is not None
    assert user.username == "testuser"