"""Per-environment storage for each read's artifacts (input.jpg, the numbered debug
slice stages, 7-scorecard.csv, verdict.md, 8-feedback.csv).

Backend is chosen from config/<APP_ENV>.yaml (`results.backend`):
  local -> results/<id>/...        on local disk
  s3    -> s3://<bucket>/<id>/...   (boto3)

`/read` writes its files into a working directory, then `commit()` finalizes it
(local: nothing to do; s3: upload the directory's files, then drop the local temp).
Later requests (`/accept`, `/decline`, `/feedback`) write single files straight to
the store -- so the multi-request flow works even when prod requests land on
different App Runner instances (local disk there is ephemeral and not shared)."""
import shutil
import tempfile
from pathlib import Path


class LocalResultStore:
    """results/<id>/... on local disk (the default / dev behavior)."""

    def __init__(self, dir="results"):
        self.root = Path(dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_working_dir(self, result_id):
        d = self.root / result_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def commit(self, result_id, working_dir):
        pass  # files are already under results/<id>/

    def exists(self, result_id):
        return (self.root / result_id).is_dir()

    def put_text(self, result_id, name, text):
        d = self.root / result_id
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)

    def describe(self, result_id, name=""):
        return str(self.root / result_id / name) if name else str(self.root / result_id)


class S3ResultStore:
    """s3://<bucket>/<id>/... -- used in aws-prod (App Runner disk is ephemeral and
    not shared across instances, so results must live in shared object storage)."""

    def __init__(self, bucket, region=None):
        import boto3
        self.bucket = bucket
        self._s3 = boto3.client("s3", region_name=region)

    def _key(self, result_id, name=""):
        return f"{result_id}/{name}" if name else f"{result_id}/"

    def new_working_dir(self, result_id):
        # Stage to a temp dir so slice_sheet's cv2.imwrite(...) stays unchanged;
        # commit() uploads its contents.
        return Path(tempfile.mkdtemp(prefix=f"{result_id}-"))

    def commit(self, result_id, working_dir):
        wd = Path(working_dir)
        try:
            for p in sorted(wd.rglob("*")):
                if p.is_file():
                    key = self._key(result_id, p.relative_to(wd).as_posix())
                    self._s3.upload_file(str(p), self.bucket, key)
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def exists(self, result_id):
        resp = self._s3.list_objects_v2(
            Bucket=self.bucket, Prefix=self._key(result_id), MaxKeys=1)
        return resp.get("KeyCount", 0) > 0

    def put_text(self, result_id, name, text):
        self._s3.put_object(
            Bucket=self.bucket, Key=self._key(result_id, name),
            Body=text.encode("utf-8"), ContentType="text/plain")

    def describe(self, result_id, name=""):
        return f"s3://{self.bucket}/{self._key(result_id, name)}"


def make_store(results):
    """Build the result store from the `results:` block of the env config."""
    backend = (results or {}).get("backend", "local")
    if backend == "s3":
        return S3ResultStore(results["bucket"], region=results.get("region"))
    if backend == "local":
        return LocalResultStore(results.get("dir", "results"))
    raise SystemExit(f"Unknown results.backend {backend!r} (use 'local' or 's3').")
