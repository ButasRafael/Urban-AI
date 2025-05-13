# tests/test_auth.py
import uuid
from fastapi import status
import pytest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models.user 
import app.models.revoked 

from app.main import app
from app.core.database import Base, get_db


TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def prepare_db():

    try:
        Path("./test.db").unlink()
    except FileNotFoundError:
        pass
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    Path("./test.db").unlink()

@pytest.fixture
def db_session():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def random_username():
    return f"user_{uuid.uuid4().hex[:8]}"


def test_register_success_and_defaults(client):
    username = random_username()
    password = "StrongPass1"

    r = client.post("/auth/register", json={"username": username, "password": password})
    assert r.status_code == 201
    assert r.json() == {"message": "User registered successfully"}

    login = client.post(
        "/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = login.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["role"] == "user"

@pytest.mark.parametrize("password", ["short", "allletters", "12345678"])
def test_register_weak_password(password, client):
    username = random_username()
    r = client.post("/auth/register", json={"username": username, "password": password})
    assert r.status_code == 422


def test_register_missing_fields(client):
    # missing username
    r1 = client.post("/auth/register", json={"password": "Strong1"})
    assert r1.status_code == 422
    # missing password
    r2 = client.post("/auth/register", json={"username": random_username()})
    assert r2.status_code == 422


def test_duplicate_register(client):
    username = random_username()
    pw = "StrongPass1"
    client.post("/auth/register", json={"username": username, "password": pw})
    r = client.post("/auth/register", json={"username": username, "password": pw})

    assert r.status_code == 400
    assert r.json()["detail"] == "Username already exists"



def test_login_nonexistent_user(client):
    r = client.post(
        "/auth/login",
        data={"username": "nouser", "password": "whatever"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert r.status_code == 400
    assert "Invalid username or password" in r.text


def test_login_wrong_password(client):
    username = random_username()
    pw = "StrongPass1"
    client.post("/auth/register", json={"username": username, "password": pw})
    r = client.post(
        "/auth/login",
        data={"username": username, "password": "WrongPass1"},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert r.status_code == 400


def test_access_me_without_token(client):
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_refresh_invalid_token(client):
    r = client.post("/auth/refresh", json={"refresh_token": "badtoken"})
    assert r.status_code == 401
    assert "Invalid refresh token" in r.text


def test_refresh_revoked_token(client):

    username = random_username(); pw = "StrongPass1"
    client.post("/auth/register", json={"username": username, "password": pw})
    login = client.post(
        "/auth/login",
        data={"username": username, "password": pw},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    tokens = login.json()
    original_refresh = tokens['refresh_token']

    r1 = client.post("/auth/refresh", json={"refresh_token": original_refresh})
    assert r1.status_code == 200
    new_tokens = r1.json()
    assert new_tokens['refresh_token'] != original_refresh

    r2 = client.post("/auth/refresh", json={"refresh_token": original_refresh})
    assert r2.status_code == 401
    assert "Refresh token revoked" in r2.text


def test_logout_invalid_token(client):

    r = client.post(
        "/auth/logout",
        headers={"Authorization": "Bearer badtoken"}
    )
    assert r.status_code == 401

def test_protected_admin_endpoint(client):
    username = random_username(); pw = "StrongPass1"

    client.post("/auth/register", json={"username": username, "password": pw, "role": "user"})
    login = client.post(
        "/auth/login",
        data={"username": username, "password": pw},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = login.json()["access_token"]

    r1 = client.get("/healthz", headers={"Authorization": f"Bearer {token}"})
    assert r1.status_code == 403

    admin = random_username()
    client.post("/auth/register", json={"username": admin, "password": pw, "role": "admin"})
    login2 = client.post(
        "/auth/login",
        data={"username": admin, "password": pw},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token2 = login2.json()["access_token"]
    r2 = client.get("/healthz", headers={"Authorization": f"Bearer {token2}"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "ok"
