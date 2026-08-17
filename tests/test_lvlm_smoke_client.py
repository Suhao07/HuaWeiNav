"""Protocol tests for the cross-host LVLM smoke client."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

from deployment.lvlm_server import smoke_client


class _Response:
    """Minimal context-managed HTTP response used by the protocol test."""

    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> "_Response":
        """Return this response to the urllib context manager."""

        return self

    def __exit__(self, *_args) -> None:
        """Close the synthetic response without side effects."""

    def read(self) -> bytes:
        """Return the configured response body."""

        return self.payload


def test_smoke_client_exercises_health_models_and_chat(monkeypatch, capsys) -> None:
    """Verify health, model discovery, chat payload, and bearer authentication."""

    requests: list[dict] = []

    def fake_urlopen(request, *, timeout):
        """Return deterministic responses for the required API endpoints."""

        path = urlsplit(request.full_url).path
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        requests.append(
            {
                "method": request.get_method(),
                "path": path,
                "authorization": request.get_header("Authorization"),
                "payload": payload,
                "timeout": timeout,
            }
        )
        if path == "/health":
            return _Response(b"")
        if path == "/v1/models":
            return _Response(json.dumps({"data": [{"id": "strive-qwen2.5-vl-7b"}]}).encode("utf-8"))
        if path == "/v1/chat/completions":
            return _Response(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        '{"ok":true,"image_received":false,'
                                        '"summary":"ready"}'
                                    )
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )
        raise AssertionError(f"unexpected endpoint: {path}")

    monkeypatch.setattr(smoke_client.urllib.request, "urlopen", fake_urlopen)

    result = smoke_client.main(
        [
            "--base-url",
            "http://lvlm.example:8000/v1",
            "--model",
            "strive-qwen2.5-vl-7b",
            "--api-key",
            "test-token",
            "--timeout",
            "2",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["health_status"] == 200
    assert output["model"] == "strive-qwen2.5-vl-7b"
    assert [record["path"] for record in requests] == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
    ]
    assert {record["authorization"] for record in requests} == {"Bearer test-token"}
    assert requests[-1]["payload"]["model"] == "strive-qwen2.5-vl-7b"
