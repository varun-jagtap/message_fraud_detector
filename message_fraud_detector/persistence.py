from .core import MessageFraudDetector

"""Atomic JSON model persistence and model metadata."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict



def detector_sha256(detector: MessageFraudDetector) -> str:
    payload = json.dumps(detector.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_detector(detector: MessageFraudDetector, path: str | os.PathLike[str]) -> Dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = detector.to_dict()
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "example_count": detector.index.size(),
    }


def load_detector(path: str | os.PathLike[str]) -> MessageFraudDetector:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError("model not found: {}".format(target))
    if not target.is_file():
        raise ValueError("model path is not a file: {}".format(target))
    with target.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    detector = MessageFraudDetector.from_dict(state)
    return detector


def inspect_model(path: str | os.PathLike[str]) -> Dict[str, Any]:
    detector = load_detector(path)
    raw = Path(path).read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": "message-fraud-detector",
        "format_version": 1,
        "example_count": detector.index.size(),
        "labels": sorted({example.label for example in detector.index.examples}),
        "vocabulary_size": len(detector.index.vectorizer.vocabulary),
        "config": detector.config.to_dict(),
    }

"""Optional local audit events for operational use."""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable, Mapping


class AuditLog:
    """Append-only JSONL audit logger protected against partial writes."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, event: Mapping[str, object]) -> dict:
        if not isinstance(event, Mapping):
            raise ValueError("audit event must be a mapping")
        payload = dict(event)
        payload["event_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        return payload

    def read_events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid audit JSONL at line {}".format(line_number)
                    ) from exc
                if not isinstance(event, dict):
                    raise ValueError("audit line must be an object")
                events.append(event)
        return events

    def verify_hashes(self) -> bool:
        for event in self.read_events():
            stored_hash = event.pop("event_hash", None)
            if stored_hash is None:
                return False
            calculated = hashlib.sha256(
                json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if calculated != stored_hash:
                return False
        return True

"""Local retention planning for generated reports and audit logs."""

import os
import time
from pathlib import Path
from typing import Iterable, List


def expired_files(directory: str, maximum_age_days: float) -> List[dict]:
    if maximum_age_days < 0:
        raise ValueError("maximum_age_days must not be negative")
    root = Path(directory)
    if not root.exists():
        return []
    cutoff = time.time() - maximum_age_days * 86400
    expired = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.stat().st_mtime < cutoff:
            expired.append({
                "path": str(path),
                "size": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            })
    return expired


def apply_retention(directory: str, maximum_age_days: float, dry_run: bool = True) -> List[dict]:
    candidates = expired_files(directory, maximum_age_days)
    if dry_run:
        return candidates
    deleted = []
    for candidate in candidates:
        path = Path(candidate["path"])
        if path.is_file():
            path.unlink()
            deleted.append(candidate)
    return deleted


def directory_usage(directory: str) -> dict:
    root = Path(directory)
    if not root.exists():
        return {"exists": False, "file_count": 0, "bytes": 0}
    files = [path for path in root.iterdir() if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": total,
    }

"""Safe model snapshot and rollback helpers using local files only."""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import List


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_snapshot(model_path: str, snapshot_directory: str) -> dict:
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError("model not found: {}".format(model))
    directory = Path(snapshot_directory)
    directory.mkdir(parents=True, exist_ok=True)
    sha = file_sha256(model)
    short_sha = sha[:12]
    snapshot = directory / "{}.{}.json".format(model.stem, short_sha)
    shutil.copy2(model, snapshot)
    return {
        "snapshot": str(snapshot),
        "sha256": sha,
        "source": str(model),
    }


def list_snapshots(model_path: str, snapshot_directory: str) -> List[dict]:
    model = Path(model_path)
    directory = Path(snapshot_directory)
    if not directory.exists():
        return []
    prefix = model.stem + "."
    snapshots = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name.startswith(prefix) and path.suffix == ".json":
            snapshots.append({
                "path": str(path),
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            })
    return snapshots


def restore_snapshot(snapshot_path: str, model_path: str) -> dict:
    snapshot = Path(snapshot_path)
    target = Path(model_path)
    if not snapshot.is_file():
        raise FileNotFoundError("snapshot not found: {}".format(snapshot))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".restore.tmp")
    shutil.copy2(snapshot, temporary)
    sha = file_sha256(temporary)
    os.replace(temporary, target)
    return {
        "restored": str(target),
        "sha256": sha,
    }
