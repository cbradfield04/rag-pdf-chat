from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_docs():
    r = client.get("/docs")
    assert r.status_code == 200