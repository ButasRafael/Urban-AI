import io
import uuid
import pytest
import numpy as np
from unittest import mock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import Base, get_db
from app.services import inference as svc

TEST_DATABASE_URL = "sqlite:///./test_inference.db"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def prepare_db():

    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

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

def create_auth_headers(client: TestClient, role: str = "user"):
    username = f"user_{uuid.uuid4().hex[:8]}"
    password = "TestPass1"

    client.post(
        "/auth/register",
        json={"username": username, "password": password, "role": role}
    )

    r = client.post(
        "/auth/login",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    tokens = r.json()
    return {"Authorization": f"Bearer {tokens['access_token']}"}

def test_image_requires_auth(client):
    dummy = io.BytesIO(b"no creds")
    resp = client.post(
        "/infer/image",
        files={"file": ("test.txt", dummy, "text/plain")}
    )
    assert resp.status_code == 401

@mock.patch.object(svc, 'process_image')
def test_image_invalid_content_type(mock_process, client):
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"not an image")
    resp = client.post(
        "/infer/image",
        headers=headers,
        files={"file": ("file.txt", dummy, "text/plain")}
    )
    assert resp.status_code == 400
    assert "Unsupported image format: .txt" in resp.text
    mock_process.assert_not_called()

@mock.patch.object(svc, 'process_image')
@mock.patch('cv2.imread')
def test_image_processing_error(mock_imread, mock_process, client):

    mock_imread.return_value = None
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"fakejpg")
    resp = client.post(
        "/infer/image",
        headers=headers,
        files={"file": ("image.jpg", dummy, "image/jpeg")}
    )
    assert resp.status_code == 400
    assert "Could not decode image" in resp.text
    mock_process.assert_not_called()

@mock.patch.object(svc, 'process_image')
@mock.patch('cv2.imread')
def test_image_success_empty_detects(mock_imread, mock_process, client):
    img = np.zeros((5,5,3), dtype=np.uint8)
    mock_imread.return_value = img
    mock_process.return_value = (img, [])
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"jpegbytes")
    resp = client.post(
        "/infer/image?use_sam=false",
        headers=headers,
        files={"file": ("image.jpg", dummy, "image/jpeg")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "media_id" in body
    assert isinstance(body["media_id"], int)
    assert body["frames"][0]["objects"] == []

@mock.patch.object(svc, 'process_image')
@mock.patch('cv2.imread')
def test_image_success_with_detects(mock_imread, mock_process, client):
    img = np.zeros((5,5,3), dtype=np.uint8)
    det = {"track_id": 1, "class_id": 0, "class_name": "obj", "confidence": 0.9,
           "bbox": [0,0,1,1], "mask": {"rle": {}, "polygon": []}}
    mock_imread.return_value = img
    mock_process.return_value = (img, [det])
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"jpegbytes")
    resp = client.post(
        "/infer/image",
        headers=headers,
        files={"file": ("image.jpg", dummy, "image/jpeg")}
    )
    assert resp.status_code == 200
    body = resp.json()
    objs = body["frames"][0]["objects"]
    assert len(objs) == 1 and objs[0]["class_name"] == "obj"

def test_image_forbidden_role(client):
    headers = create_auth_headers(client, role="authority")
    dummy = io.BytesIO(b"jpegbytes")
    resp = client.post(
        "/infer/image",
        headers=headers,
        files={"file": ("image.jpg", dummy, "image/jpeg")}
    )
    assert resp.status_code == 403

def test_video_requires_auth(client):
    dummy = io.BytesIO(b"no creds")
    resp = client.post(
        "/infer/video",
        files={"file": ("test.mp4", dummy, "text/plain")}
    )
    assert resp.status_code == 401

@mock.patch.object(svc, 'process_video')
def test_video_invalid_content_type(mock_process, client):
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"not a video")
    resp = client.post(
        "/infer/video",
        headers=headers,
        files={"file": ("vid.txt", dummy, "text/plain")}
    )
    assert resp.status_code == 400
    assert "Unsupported video format: .txt" in resp.text
    mock_process.assert_not_called()

@mock.patch.object(svc, 'process_video')
@mock.patch('app.api.inference_routes._save_temp')
def test_video_processing_error(mock_save, mock_process, client):
    mock_save.return_value = "/tmp/fake"
    mock_process.side_effect = RuntimeError("fail")
    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"videobytes")
    resp = client.post(
        "/infer/video",
        headers=headers,
        files={"file": ("video.mp4", dummy, "video/mp4")}
    )
    assert resp.status_code == 500
    assert "Inference failed" in resp.text

@mock.patch.object(svc, 'process_video')
@mock.patch('app.api.inference_routes._save_temp')
def test_video_success(mock_save, mock_process, client):
    mock_save.return_value = "/tmp/fake.mp4"

    frames = [{"frame_index": 0, "timestamp_ms": 0.0, "objects": []}]
    tmp_out = "/tmp/out.mp4"

    open(tmp_out, "wb").close()

    mock_process.return_value = (tmp_out, frames)

    headers = create_auth_headers(client)
    dummy = io.BytesIO(b"videobytes")
    resp = client.post(
        "/infer/video?use_sam=false",
        headers=headers,
        files={"file": ("video.mp4", dummy, "video/mp4")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["frames"][0]["frame_index"] == 0
    assert data["annotated_video_url"].endswith(".mp4")


@mock.patch.object(svc, 'process_video')
def test_video_forbidden_role(mock_process, client):
    headers = create_auth_headers(client, role="authority")
    dummy = io.BytesIO(b"videobytes")
    resp = client.post(
        "/infer/video",
        headers=headers,
        files={"file": ("video.mp4", dummy, "video/mp4")},
    )
    assert resp.status_code == 403
