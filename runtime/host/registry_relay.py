"""Serve one digest-pinned registry repository to every node from a single download.

Docker pulls ``127.0.0.1:<port>/<repository>@<digest>`` on each node. Docker
treats loopback registries as plain-HTTP registries without daemon
configuration, so Node A binds this relay to its loopback address and each
worker reaches it through an SSH remote forward on the worker's own loopback
address. The relay fetches each manifest and blob from the upstream registry
once, verifies it against its digest and serves the verified copy to every
node. The internet link carries the image once while every node unpacks it
concurrently. ``Relay.image`` also exposes the pinned configuration and layer
list, so the installer can send a node only the layers it lacks.

Upstream access is anonymous bearer-token pulls, which public GHCR and Docker
Hub repositories use. A registry that requires credentials makes the relay
fail, and the caller falls back to copying a local image between nodes.
"""
import concurrent.futures
import hashlib
import http.server
import json
from pathlib import Path
import re
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
PORT = 5255
STREAMS = 8
RANGED_MINIMUM = 256 * 1024**2
MANIFESTS = ", ".join([
    "application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def parse(reference):
    registry, _, remainder = reference.partition("/")
    repository, _, digest = remainder.partition("@")
    if not ("." in registry or ":" in registry) or not repository or not DIGEST.fullmatch(digest):
        raise ValueError("The relay requires a registry reference pinned by sha256 digest")
    return registry, repository, digest


def _digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(16 * 1024**2):
            value.update(block)
    return "sha256:" + value.hexdigest()


class Upstream:
    """Anonymous pulls from one repository, following one bearer-token challenge."""

    def __init__(self, registry, repository, *, scheme="https", opener=None):
        self.base = scheme + "://" + registry + "/v2/" + repository
        self.repository = repository
        self.opener = opener or urllib.request.build_opener()
        self.token = None

    def open(self, path, headers=None):
        for attempt in range(2):
            request = urllib.request.Request(self.base + path, headers=headers or {})
            if self.token:
                # Unredirected: blob redirects lead to signed storage URLs,
                # which must not receive the registry token.
                request.add_unredirected_header("Authorization", "Bearer " + self.token)
            try:
                return self.opener.open(request, timeout=120)
            except urllib.error.HTTPError as error:
                if error.code != 401 or attempt:
                    raise
                self.authenticate(error.headers.get("WWW-Authenticate", ""))
        raise AssertionError("unreachable")

    def authenticate(self, challenge):
        scheme, _, parameters = challenge.partition(" ")
        values = dict(re.findall(r'(\w+)="([^"]*)"', parameters))
        if scheme.lower() != "bearer" or "realm" not in values:
            raise ValueError("The registry requires credentials that the relay does not hold")
        query = {key: values[key] for key in ("service", "scope") if key in values}
        query.setdefault("scope", "repository:" + self.repository + ":pull")
        with self.opener.open(values["realm"] + "?" + urllib.parse.urlencode(query), timeout=60) as response:
            document = json.load(response)
        self.token = document.get("token") or document.get("access_token")
        if not self.token:
            raise ValueError("The registry issued no anonymous pull token")

    def manifest(self, digest, accept):
        with self.open("/manifests/" + digest, {"Accept": accept or MANIFESTS}) as response:
            return response.headers.get("Content-Type", ""), response.read()

    def blob(self, digest, target):
        """Write the blob to ``target``; large blobs use parallel byte ranges."""
        with self.open("/blobs/" + digest) as response:
            length = int(response.headers.get("Content-Length") or -1)
            if length < RANGED_MINIMUM:
                with open(target, "wb") as stream:
                    shutil.copyfileobj(response, stream, 8 * 1024**2)
                return
        try:
            self._ranges(digest, target, length)
        except ValueError:
            # One stream serves a registry or storage host without byte ranges.
            with self.open("/blobs/" + digest) as response, open(target, "wb") as stream:
                shutil.copyfileobj(response, stream, 8 * 1024**2)

    def _ranges(self, digest, target, length):
        with open(target, "wb") as stream:
            stream.truncate(length)
        size = -(-length // STREAMS)
        bounds = [(start, min(start + size, length) - 1) for start in range(0, length, size)]

        def part(span):
            first, last = span
            with self.open("/blobs/" + digest, {"Range": f"bytes={first}-{last}"}) as response:
                if response.status != 206:
                    raise ValueError("The registry ignored a byte-range request")
                with open(target, "r+b") as stream:
                    stream.seek(first)
                    remaining = last - first + 1
                    while remaining:
                        block = response.read(min(remaining, 8 * 1024**2))
                        if not block:
                            raise ValueError("The registry closed a byte range early")
                        stream.write(block)
                        remaining -= len(block)

        with concurrent.futures.ThreadPoolExecutor(len(bounds)) as pool:
            list(pool.map(part, bounds))


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.respond(body=True)

    def do_HEAD(self):
        self.respond(body=False)

    def log_message(self, *_):
        pass

    def respond(self, *, body):
        relay = self.server.relay
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/v2", "/v2/"):
            return self.reply(200, "application/json", b"{}", body=body)
        prefix = "/v2/" + relay.repository + "/"
        kind, _, digest = path.removeprefix(prefix).partition("/")
        if not path.startswith(prefix) or kind not in ("manifests", "blobs") or not DIGEST.fullmatch(digest):
            return self.reply(404, "application/json", b'{"errors":[{"code":"NAME_UNKNOWN"}]}', body=body)
        try:
            if kind == "manifests":
                media, content = relay.manifest(digest, self.headers.get("Accept"))
                return self.reply(200, media, content, digest=digest, body=body)
            blob = relay.blob(digest)
        except (OSError, ValueError, urllib.error.URLError) as error:
            relay.errors.append(f"{kind[:-1]} {digest}: {error}")
            return self.reply(502, "application/json", b'{"errors":[{"code":"UNKNOWN"}]}', body=body)
        with open(blob, "rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(0)
            self.headers_for(200, "application/octet-stream", size, digest)
            if body:
                self.connection.sendfile(stream)

    def headers_for(self, status, media, length, digest=None):
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(length))
        self.send_header("Docker-Distribution-API-Version", "registry/2.0")
        if digest:
            self.send_header("Docker-Content-Digest", digest)
        self.end_headers()

    def reply(self, status, media, content, *, digest=None, body):
        self.headers_for(status, media, len(content), digest)
        if body:
            self.wfile.write(content)


class Relay:
    """A loopback registry for one pinned repository; ``close`` removes its cache."""

    def __init__(self, reference, directory, *, upstream=None, port=PORT):
        registry, self.repository, self.digest = parse(reference)
        self.upstream = upstream or Upstream(registry, self.repository)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.manifests, self.locks, self.errors = {}, {}, []
        self.guard = threading.Lock()
        try:
            self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        except OSError:
            self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.relay = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def reference(self, port=None):
        return f"127.0.0.1:{port or self.port}/{self.repository}@{self.digest}"

    def _lock(self, digest):
        with self.guard:
            return self.locks.setdefault(digest, threading.Lock())

    def manifest(self, digest, accept):
        with self._lock(digest):
            if digest not in self.manifests:
                media, content = self.upstream.manifest(digest, accept)
                if "sha256:" + hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("Upstream manifest differs from its digest")
                self.manifests[digest] = (media, content)
            return self.manifests[digest]

    def blob(self, digest):
        target = self.directory / digest.removeprefix("sha256:")
        with self._lock(digest):
            if not target.is_file():
                partial = target.with_suffix(".partial")
                self.upstream.blob(digest, partial)
                if _digest(partial) != digest:
                    partial.unlink()
                    raise ValueError("Upstream blob differs from its digest")
                partial.replace(target)
        return target

    def image(self, image_id, platform=("linux", "arm64")):
        """Return the pinned image's configuration bytes and ordered layers.

        Each layer is ``(diff_id, digest, size)``: the uncompressed digest the
        configuration records, and the registry blob digest and size. An index
        selects its single manifest for ``platform``. The configuration digest
        must equal ``image_id``.
        """
        media, content = self.manifest(self.digest, None)
        document = json.loads(content)
        if "manifests" in document:
            entries = [entry for entry in document["manifests"]
                       if (entry.get("platform", {}).get("os"), entry.get("platform", {}).get("architecture")) == platform]
            if len(entries) != 1 or not DIGEST.fullmatch(entries[0].get("digest", "")):
                raise ValueError("The pinned index names no single manifest for this platform")
            media, content = self.manifest(entries[0]["digest"], entries[0].get("mediaType"))
            document = json.loads(content)
        if document.get("config", {}).get("digest") != image_id:
            raise ValueError("The pinned manifest names a different image configuration")
        config = self.blob(image_id).read_bytes()
        diff_ids, layers = json.loads(config)["rootfs"]["diff_ids"], document["layers"]
        # Foreign layers carry external URLs instead of registry blobs.
        if (len(diff_ids) != len(layers) or any(not DIGEST.fullmatch(value) for value in diff_ids)
                or any(not DIGEST.fullmatch(layer.get("digest", "")) or layer.get("urls")
                       or "foreign" in layer.get("mediaType", "") for layer in layers)):
            raise ValueError("The pinned manifest's layers are not loadable registry blobs")
        return config, [(diff, layer["digest"], int(layer["size"])) for diff, layer in zip(diff_ids, layers)]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
