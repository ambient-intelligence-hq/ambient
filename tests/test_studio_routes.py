"""Tests for the Studio-facing engine routes:
  * GET /v1/files/{id}/content  — ranged video streaming (query-param auth too)
  * GET /v1/sessions            — server-side session list

No DB/network: a fake store is injected into app.state and the router is
exercised with Starlette's TestClient.

Run: python -m unittest tests.test_studio_routes
"""
import os
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ambient.config import settings
from ambient.server.routes import managed_agents


def _session(session_id: str) -> dict:
    now = "2026-01-01T00:00:00.000Z"
    return {
        "session_id": session_id, "status": "ready", "video_id": "v1",
        "created_at": now, "updated_at": now, "usage": {}, "metadata": {},
    }


class _FakeStore:
    def __init__(self, file_rec=None, sessions=None):
        self._file_rec = file_rec
        self._sessions = sessions or []

    async def get_file(self, file_id):
        if self._file_rec and self._file_rec.get("video_id") == file_id:
            return self._file_rec
        return None

    async def list_sessions(self, limit=50):
        return self._sessions[:limit]


def _make_client(store) -> TestClient:
    app = FastAPI()
    app.include_router(managed_agents.router)
    app.state.store = store
    return TestClient(app)


class FileContentTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self._tmp.write(b"0123456789" * 100)  # 1000 bytes
        self._tmp.close()
        self.rec = {"video_id": "v1", "local_path": self._tmp.name, "mime_type": "video/mp4"}
        self.client = _make_client(_FakeStore(file_rec=self.rec))
        self.auth = {"x-api-key": settings.api_key}

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_full_get(self):
        r = self.client.get("/v1/files/v1/content", headers=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["accept-ranges"], "bytes")
        self.assertEqual(len(r.content), 1000)

    def test_range_get(self):
        r = self.client.get("/v1/files/v1/content", headers={**self.auth, "Range": "bytes=0-99"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.headers["content-range"], "bytes 0-99/1000")
        self.assertEqual(r.headers["content-length"], "100")
        self.assertEqual(len(r.content), 100)

    def test_suffix_range(self):
        r = self.client.get("/v1/files/v1/content", headers={**self.auth, "Range": "bytes=-50"})
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r.headers["content-range"], "bytes 950-999/1000")
        self.assertEqual(len(r.content), 50)

    def test_unsatisfiable_range(self):
        r = self.client.get("/v1/files/v1/content", headers={**self.auth, "Range": "bytes=5000-6000"})
        self.assertEqual(r.status_code, 416)

    def test_query_param_auth(self):
        # A bare <video src> can't set headers; the ?api_key= form must work.
        r = self.client.get(f"/v1/files/v1/content?api_key={settings.api_key}")
        self.assertEqual(r.status_code, 200)

    def test_missing_auth_rejected(self):
        r = self.client.get("/v1/files/v1/content")
        self.assertEqual(r.status_code, 401)

    def test_unknown_file(self):
        r = self.client.get("/v1/files/nope/content", headers=self.auth)
        self.assertEqual(r.status_code, 404)

    def test_missing_local_bytes(self):
        client = _make_client(_FakeStore(file_rec={"video_id": "v1", "local_path": "/no/such.mp4"}))
        # falls back to VIDEO_FOLDER/v1.ext; that won't exist in the test env
        r = client.get("/v1/files/v1/content", headers=self.auth)
        self.assertEqual(r.status_code, 404)


class SessionListTests(unittest.TestCase):
    def test_list_sessions(self):
        store = _FakeStore(sessions=[_session("s1"), _session("s2")])
        client = _make_client(store)
        r = client.get("/v1/sessions", headers={"x-api-key": settings.api_key})
        self.assertEqual(r.status_code, 200)
        data = r.json()["data"]
        self.assertEqual([s["id"] for s in data], ["s1", "s2"])
        self.assertEqual(data[0]["type"], "session")


if __name__ == "__main__":
    unittest.main()
