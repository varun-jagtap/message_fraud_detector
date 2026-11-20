"""Small standard-library web interface for the message fraud detector."""

import json
import mimetypes
import sys
import threading
import urllib.parse
import csv
import io
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from message_fraud_detector.core import (
    DetectorConfig,
    Example,
    MessageFraudDetector,
)
from message_fraud_detector.persistence import inspect_model, load_detector, save_detector
from message_fraud_detector.sample import generate_records


WEB_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "web_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def discover_models() -> list[str]:
    """Find detector models in the project root and all subdirectories."""
    models: Dict[str, Path] = {}
    candidates = list(PROJECT_ROOT.rglob("*.json"))
    candidates.extend(DATA_ROOT.rglob("*.json"))
    for path in candidates:
        try:
            if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
                continue
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                continue
            if value.get("format") != "message-fraud-detector":
                continue
            relative = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            models[relative] = path.resolve()
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return sorted(models)


def resolve_model(reference: str) -> Path:
    """Resolve a model reference without allowing paths outside the project."""
    reference = str(reference or "detector.json").strip().replace("\\", "/")
    if "/" not in reference:
        web_candidate = (DATA_ROOT / reference).resolve()
        if web_candidate.is_file():
            return web_candidate
        root_candidate = (PROJECT_ROOT / reference).resolve()
        if root_candidate.is_file():
            return root_candidate
        if not reference.lower().endswith(".json"):
            web_candidate = (DATA_ROOT / (safe_name(reference, ".json"))).resolve()
            if web_candidate.is_file():
                return web_candidate
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("model path must be inside the project") from exc
    if not candidate.is_file():
        raise FileNotFoundError("model not found: {}".format(reference))
    return candidate


def safe_name(value: str, suffix: str) -> str:
    value = (value or "default").strip()
    value = "".join(character if character.isalnum() or character in "-_." else "-" for character in value)
    value = value.strip(".-")
    if not value:
        value = "default"
    if not value.lower().endswith(suffix):
        value += suffix
    return value


def json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def parse_records(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    records = payload.get("records")
    if isinstance(records, list):
        yield from records
        return
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("provide records or text")
    input_format = payload.get("format", "jsonl")
    if input_format == "json":
        value = json.loads(text)
        if isinstance(value, list):
            yield from value
            return
        if isinstance(value, dict):
            yield value
            return
        raise ValueError("JSON input must be an object or array")
    if input_format == "csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV input must contain a text column")
        yield from reader
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("text must contain one JSON object per line") from exc


class FraudWebHandler(BaseHTTPRequestHandler):
    server_version = "MessageFraudWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            "%s - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )

    def send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, "File not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_file(WEB_ROOT / "index.html")
            return
        if parsed.path == "/styles.css":
            self.send_file(WEB_ROOT / "styles.css")
            return
        if parsed.path == "/app.js":
            self.send_file(WEB_ROOT / "app.js")
            return
        if parsed.path == "/api/models":
            models = discover_models()
            self.send_json(HTTPStatus.OK, {"models": models})
            return
        if parsed.path == "/api/health":
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = json_body(self)
            if parsed.path == "/api/generate":
                count = int(payload.get("count", 100))
                seed = int(payload.get("seed", 7))
                if count < 1 or count > 100000:
                    raise ValueError("count must be between 1 and 100000")
                records = generate_records(count, seed)
                self.send_json(HTTPStatus.OK, {"count": len(records), "records": records})
                return
            if parsed.path == "/api/config/create":
                name = safe_name(payload.get("name", "detector"), ".json")
                path = DATA_ROOT / name
                config = DetectorConfig(**payload.get("config", {}))
                config.validate()
                path.write_text(
                    json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.send_json(HTTPStatus.OK, {"path": name, "config": config.to_dict()})
                return
            if parsed.path == "/api/config/validate":
                config = DetectorConfig(**payload.get("config", {}))
                config.validate()
                self.send_json(HTTPStatus.OK, {"valid": True, "config": config.to_dict()})
                return
            if parsed.path == "/api/train":
                model_name = safe_name(payload.get("model", "detector"), ".json")
                records = list(parse_records(payload))
                examples = []
                for index, record in enumerate(records, start=1):
                    if not isinstance(record, dict):
                        raise ValueError("record {} must be an object".format(index))
                    text = record.get("text")
                    label = record.get("label")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("record {} needs text".format(index))
                    if not isinstance(label, str):
                        raise ValueError("record {} needs label".format(index))
                    examples.append(
                        Example(
                            text=text.strip(),
                            label=label.strip().lower(),
                            source=str(record.get("source", "web")),
                            message_id=str(record.get("message_id", "")),
                            metadata={
                                key: value
                                for key, value in record.items()
                                if key not in ("text", "label", "source", "message_id")
                            },
                        )
                    )
                config = DetectorConfig(**payload.get("config", {}))
                detector = MessageFraudDetector.train(examples, config)
                metadata = save_detector(detector, DATA_ROOT / model_name)
                metadata["path"] = model_name
                self.send_json(HTTPStatus.OK, metadata)
                return
            if parsed.path == "/api/score":
                model_name = str(payload.get("model", "detector.json"))
                text = payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text is required")
                detector = load_detector(resolve_model(model_name))
                result = detector.score(text, str(payload.get("message_id", "")))
                self.send_json(HTTPStatus.OK, result.to_dict())
                return
            if parsed.path == "/api/batch":
                model_name = str(payload.get("model", "detector.json"))
                detector = load_detector(resolve_model(model_name))
                results = []
                for index, record in enumerate(parse_records(payload), start=1):
                    if not isinstance(record, dict):
                        raise ValueError("record {} must be an object".format(index))
                    text = record.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("record {} needs text".format(index))
                    result = detector.score(
                        text,
                        str(record.get("message_id", "")),
                    )
                    results.append(result.to_dict())
                self.send_json(HTTPStatus.OK, {"count": len(results), "results": results})
                return
            if parsed.path == "/api/inspect":
                model_name = str(payload.get("model", "detector.json"))
                self.send_json(
                    HTTPStatus.OK,
                    inspect_model(resolve_model(model_name)),
                )
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
        except (ValueError, KeyError, OSError, json.JSONDecodeError, FileNotFoundError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def make_server(host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), FraudWebHandler)


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the message fraud detector web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print("Message Fraud Detector: http://{}:{}/".format(args.host, args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
