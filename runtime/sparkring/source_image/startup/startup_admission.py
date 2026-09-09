"""Keep public inference outside the scheduler until startup warmup completes."""

import hmac
import os
from pathlib import Path

HEADER = b"x-sparkring-startup-token"


class StartupAdmission:
    def __init__(self, app):
        self.app = app
        self.token = os.environ.get("SPARKRING_STARTUP_TOKEN", "").encode()
        if not self.token:
            raise RuntimeError("The serving wrapper must provide a startup token")
        self.ready = Path(os.environ.get(
            "SPARKRING_READY_PATH", "/tmp/sparkring-engine-ready"
        ))

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", ())
        tokens = [value for name, value in headers if name.lower() == HEADER]
        internal = len(tokens) == 1 and hmac.compare_digest(tokens[0], self.token)
        # Do not leak the bypass credential into downstream request logging.
        scope = dict(scope, headers=[(name, value) for name, value in headers
                                     if name.lower() != HEADER])
        probe = scope.get("method") in ("GET", "HEAD") and scope["path"] in (
            "/health", "/v1/models", "/metrics"
        )
        if not (internal or probe or self.ready.is_file()):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
                return
            body = b'{"error":{"message":"Model warmup is in progress","type":"service_unavailable"}}'
            await send({"type": "http.response.start", "status": 503, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", b"5"),
            ]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
