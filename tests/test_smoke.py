import sys
from pathlib import Path

# add repo root to import path so `import main` works in CI
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200

def test_docs():
    r = client.get("/docs")
    assert r.status_code == 200