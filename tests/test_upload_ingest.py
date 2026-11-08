"""POST /v1/files: when does an upload get a background description job?

  * host backend, no S3  -> yes. The ingest worker describes the local file on the
    host (IngestWorker._run_job), so S3 isn't needed. (Previously no job was
    queued and the first question waited on a live description.)
  * e2b backend, no S3   -> no. The ingest sandbox reads the source from S3.
  * e2b backend, S3 copy -> yes.

No DB/network/ffmpeg: store_video is stubbed and a fake store + ingest queue are
injected into app.state.

Run: python -m unittest tests.test_upload_ingest
"""
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ambient.config import settings
from ambient.server.routes import managed_agents


class _FakeStore:
    def __init__(self):
        self.files = {}

    async def get_file(self, file_id):
        return self.files.get(file_id)

    async def put_file(self, rec):
        self.files[rec["video_id"]] = dict(rec)
        return rec


class _FakeIngest:
    def __init__(self):
        self.enqueued = []

    async def enqueue(self, video_id):
        self.enqueued.append(video_id)


def _stored(r2_key):
    def fake_store_video(data, filename, content_type):
        return {
            "video_id": "vid_test", "filename": filename, "mime_type": "video/mp4",
            "size_bytes": len(data), "created_at": "2026-01-01T00:00:00.000Z",
            "local_path": "/tmp/vid_test.mp4", "r2_key": r2_key,
        }
    return fake_store_video


class UploadIngestTests(unittest.TestCase):
    def _upload(self, backend, r2_key):
        store, ingest = _FakeStore(), _FakeIngest()
        app = FastAPI()
        app.include_router(managed_agents.router)
        app.state.store, app.state.ingest = store, ingest
        with mock.patch.object(settings, "sandbox_backend", backend), \
             mock.patch.object(managed_agents, "store_video", _stored(r2_key)):
            resp = TestClient(app).post(
                "/v1/files",
                headers={"x-api-key": settings.api_key},
                files={"file": ("clip.mp4", b"not really a video", "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return store.files["vid_test"], ingest.enqueued

    def test_host_backend_without_s3_is_described_in_background(self):
        row, enqueued = self._upload("inprocess", r2_key=None)
        self.assertEqual(enqueued, ["vid_test"])
        self.assertEqual(row["description_status"], "pending")

    def test_e2b_backend_without_s3_is_not_enqueued(self):
        row, enqueued = self._upload("e2b", r2_key=None)
        self.assertEqual(enqueued, [])
        self.assertIsNone(row.get("description_status"))

    def test_e2b_backend_with_s3_copy_is_enqueued(self):
        row, enqueued = self._upload("e2b", r2_key="videos/vid_test.mp4")
        self.assertEqual(enqueued, ["vid_test"])
        self.assertEqual(row["description_status"], "pending")


if __name__ == "__main__":
    unittest.main()
