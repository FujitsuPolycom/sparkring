"""Authenticated, on-demand helper calls. No background work or model API access."""
import hmac
import json
import secrets
import threading
import time

from runtime.host import action_commands, helper

TTL = 300
MAX_PENDING = 32


class Manager:
    def __init__(self, token, *, invoke=None, now=time.monotonic):
        if not isinstance(token, str) or len(token) < 32 or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("Manage token must be at least 32 ASCII characters without whitespace")
        self.token, self.csrf = token, secrets.token_urlsafe(32)
        self.invoke = invoke or (lambda argv: helper.request(argv, capture=True))
        self.now, self.pending = now, {}
        self.lock = threading.Lock()  # protects simultaneous HTTP confirmations; starts no thread

    def authenticated(self, headers):
        supplied = headers.get("Authorization", "")
        return supplied.isascii() and hmac.compare_digest(supplied, "Bearer " + self.token)

    def session(self, headers):
        if not self.authenticated(headers):
            raise PermissionError("Authentication required")
        return {"csrf": self.csrf, "expires_in": TTL}

    def dispatch(self, path, headers, body):
        self.session(headers)
        if headers.get("Origin") not in ("http://" + headers.get("Host", ""), "https://" + headers.get("Host", "")):
            raise PermissionError("Same-origin request required")
        csrf = headers.get("X-CSRF-Token", "")
        if not csrf.isascii() or not hmac.compare_digest(csrf, self.csrf):
            raise PermissionError("CSRF token required")
        if headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
            raise PermissionError("Cross-site request refused")
        if not isinstance(body, dict):
            raise ValueError("Expected an action object")
        if path == "/api/manage/preview":
            argv = action_commands.command(body)
            result = self.invoke(argv)
            if body["action"] == "logs":
                return {**result, "follow": "sparkring logs --on " + body["on"] + " --follow"}
            document = json.loads(result["stdout"]) if result["code"] == 0 else {}
            ticket = None
            if result["code"] == 0 and document.get("activity", {}).get("grade") != "blocked":
                execute = action_commands.command(body, review=document["review"])
                with self.lock:
                    self.pending = {key: row for key, row in self.pending.items() if row[0] > self.now()}
                    if len(self.pending) >= MAX_PENDING:
                        raise ValueError("Too many pending previews; finish or let one expire")
                    ticket = secrets.token_urlsafe(32)
                    self.pending[ticket] = (self.now() + TTL, execute)
            return {**result, "plan": document, "confirmation": ticket}
        if path == "/api/manage/confirm":
            if set(body) != {"confirmation"} or not isinstance(body["confirmation"], str):
                raise ValueError("Confirm the exact preview")
            with self.lock:
                row = self.pending.pop(body["confirmation"], None)
            if row is None or row[0] <= self.now():
                raise ValueError("Preview expired or already used; review again")
            return self.invoke(row[1])
        raise ValueError("Unknown management endpoint")
