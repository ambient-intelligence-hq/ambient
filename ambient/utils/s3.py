"""
upload chunk to s3/minio
"""

import os
import json
import boto3
import logging
from typing import Optional, List, Dict
from botocore.config import Config
from ambient.config import settings
from functools import lru_cache

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(self):
        # Force SigV4 for S3-compatible providers (for example Cloudflare R2).
        s3_config = Config(signature_version="s3v4")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=s3_config,
        )
        self.bucket = settings.s3_bucket

    def upload_chunks_to_s3(
        self,
        chunk_paths: List[str],
        stream_id: str,
        tenant_id: Optional[str] = None,
        generate_presigned_url: bool = True,
    ) -> List[Dict[str, str]]:
        if not self.bucket:
            raise ValueError("S3_BUCKET is required")
        if tenant_id:
            s3_path = f"{tenant_id}/{stream_id}/chunks"
        else:
            s3_path = f"{stream_id}/chunks"

        urls = []
        extra_args = {
            "ContentType": "video/mp4",
            "Metadata": {"source": "stream-worker"},
        }

        for chunk_path in chunk_paths:
            chunk_filename = os.path.basename(chunk_path)
            url = self.upload_file(
                chunk_path, f"{s3_path}/{chunk_filename}", extra_args
            )
            if generate_presigned_url:
                presigned_url = self.get_presigned_url(
                    f"{s3_path}/{chunk_filename}",
                    expires_in=settings.chunk_presigned_url_ttl,
                )
                urls.append({"url": url, "presigned_url": presigned_url})
            else:
                urls.append({"url": url})

        return urls

    def read_json(self, key: str) -> Optional[dict]:
        """Read and parse a JSON object from S3, or None if it's missing."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except Exception:  # missing key / access error -> treat as absent
            return None

    def object_exists(self, key: str) -> bool:
        """True if an object exists at `key` (cheap HEAD). False on 404 or any
        error — callers use this to avoid handing out a presigned URL for a
        missing object (which a remote consumer would then 404 on)."""
        if not self.bucket:
            return False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001 - 404 / access error -> treat as absent
            return False

    def get_presigned_url(self, key: str, expires_in: int = 300) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def upload_file(
        self, local_path: str, key: str, extra_args: Optional[dict] = None
    ) -> str:
        if extra_args is None:
            extra_args = {
                "ContentType": "video/mp4",
                "Metadata": {"source": "stream-worker"},
            }
        self.s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args)
        return f"s3://{self.bucket}/{key}"


@lru_cache(maxsize=1)
def get_s3_client() -> S3Client:
    return S3Client()


s3_client = get_s3_client()


if __name__ == "__main__":
    s3_client = get_s3_client()
    print(
        s3_client.get_presigned_url("s_d30e03a329d0/chunks/chunk_20260216_055352.mp4")
    )
