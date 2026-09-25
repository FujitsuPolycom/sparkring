"""Registry relay against a local token-authenticated registry with redirected storage."""
import concurrent.futures
import hashlib
import http.server
import json
import threading
import urllib.error
import urllib.request

import pytest

from runtime.host import registry_relay as relay_module

REPOSITORY = "owner/image"


def digest(content):
    return "sha256:" + hashlib.sha256(content).hexdigest()


class Registry:
    """Anonymous-token registry whose blob URLs redirect to a storage path."""

    def __init__(self, blobs, manifest):
        self.blobs = {digest(b): b for b in blobs}
        self.manifest = manifest
        self.requests = []
        self.corrupt = set()
        registry = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def send(self, status, body=b"", headers=()):
                self.send_response(status)
                for key, value in headers:
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                registry.requests.append((self.path, self.headers.get("Authorization"), self.headers.get("Range")))
                if self.path.startswith("/token"):
                    return self.send(200, json.dumps({"token": "anonymous"}).encode())
                if self.path.startswith("/storage/"):
                    assert self.headers.get("Authorization") is None, "storage received the registry token"
                    name = self.path.removeprefix("/storage/")
                    content = registry.blobs[name]
                    if name in registry.corrupt:
                        content = b"x" * len(content)
                    if self.headers.get("Range"):
                        first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                        return self.send(206, content[first:last + 1],
                                         [("Content-Range", f"bytes {first}-{last}/{len(content)}")])
                    return self.send(200, content)
                if self.headers.get("Authorization") != "Bearer anonymous":
                    realm = f"http://127.0.0.1:{registry.port}/token"
                    return self.send(401, b"", [("WWW-Authenticate",
                                                 f'Bearer realm="{realm}",service="test",scope="repository:{REPOSITORY}:pull"')])
                prefix = "/v2/" + REPOSITORY + "/"
                if self.path == prefix + "manifests/" + digest(registry.manifest):
                    return self.send(200, registry.manifest,
                                     [("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")])
                name = self.path.removeprefix(prefix + "blobs/")
                if name in registry.blobs:
                    return self.send(307, b"", [("Location", f"http://127.0.0.1:{registry.port}/storage/{name}")])
                return self.send(404)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(relay_module, "RANGED_MINIMUM", 1000)
    small, large = b"config" * 10, bytes(range(256)) * 40
    manifest = json.dumps({"layers": [digest(small), digest(large)]}).encode()
    registry = Registry([small, large], manifest)
    reference = f"127.0.0.1:{registry.port}/{REPOSITORY}@{digest(manifest)}"
    upstream = relay_module.Upstream(f"127.0.0.1:{registry.port}", REPOSITORY, scheme="http")
    relay = relay_module.Relay(reference, tmp_path / "relay", upstream=upstream, port=0)
    yield registry, relay, small, large, manifest
    relay.close()
    registry.close()


def get(relay, path, **headers):
    request = urllib.request.Request(f"http://127.0.0.1:{relay.port}{path}", headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, dict(response.headers), response.read()


def test_relay_serves_verified_manifest_and_blobs_through_one_anonymous_token(setup):
    registry, relay, small, large, manifest = setup
    assert get(relay, "/v2/")[0] == 200
    status, headers, body = get(relay, f"/v2/{REPOSITORY}/manifests/{digest(manifest)}", Accept="application/json")
    assert body == manifest and headers["Docker-Content-Digest"] == digest(manifest)
    assert headers["Content-Type"] == "application/vnd.docker.distribution.manifest.v2+json"
    assert get(relay, f"/v2/{REPOSITORY}/blobs/{digest(small)}")[2] == small
    assert get(relay, f"/v2/{REPOSITORY}/blobs/{digest(large)}")[2] == large
    ranged = [r for r in registry.requests if r[0].startswith("/storage/") and r[2]]
    assert len(ranged) == relay_module.STREAMS
    assert relay.reference(5255) == f"127.0.0.1:5255/{REPOSITORY}@{digest(manifest)}"


def test_concurrent_nodes_share_one_upstream_download(setup):
    registry, relay, _, large, _ = setup
    path = f"/v2/{REPOSITORY}/blobs/{digest(large)}"
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        bodies = list(pool.map(lambda _: get(relay, path)[2], range(4)))
    assert bodies == [large] * 4
    whole = [r for r in registry.requests if r[0].startswith("/storage/") and not r[2]]
    assert len(whole) == 1


def test_relay_refuses_other_repositories_tags_and_corrupt_content(setup):
    registry, relay, small, _, _ = setup
    for path in (f"/v2/other/image/blobs/{digest(small)}", f"/v2/{REPOSITORY}/manifests/latest"):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(relay, path)
        assert error.value.code == 404
    registry.corrupt.add(digest(small))
    with pytest.raises(urllib.error.HTTPError) as error:
        get(relay, f"/v2/{REPOSITORY}/blobs/{digest(small)}")
    assert error.value.code == 502 and "differs from its digest" in relay.errors[0]
    assert not list(relay.directory.iterdir())


def test_close_removes_the_blob_cache(setup):
    _, relay, small, _, _ = setup
    get(relay, f"/v2/{REPOSITORY}/blobs/{digest(small)}")
    assert list(relay.directory.iterdir())
    relay.close()
    assert not relay.directory.exists()


class Store:
    """Upstream double holding manifests and blobs by digest."""

    def __init__(self, *documents, blobs=()):
        self.manifests = {digest(json.dumps(d).encode()): json.dumps(d).encode() for d in documents}
        self.blobs = {digest(b): b for b in blobs}

    def manifest(self, name, accept):
        return "application/json", self.manifests[name]

    def blob(self, name, target):
        target.write_bytes(self.blobs[name])


def layered(tmp_path, *, foreign=False, config_for=None):
    layers = [b"layer one", b"layer two"]
    config = json.dumps({"rootfs": {"diff_ids": ["sha256:" + "1" * 64, "sha256:" + "2" * 64]}}).encode()
    kinds = ["application/vnd.docker.image.rootfs.diff.tar.gzip",
             "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip" if foreign else
             "application/vnd.docker.image.rootfs.diff.tar.gzip"]
    arm = {"config": {"digest": config_for or digest(config)},
           "layers": [{"mediaType": kind, "digest": digest(layer), "size": len(layer)} for kind, layer in zip(kinds, layers)]}
    index = {"manifests": [
        {"digest": digest(json.dumps({"other": True}).encode()), "platform": {"os": "linux", "architecture": "amd64"}},
        {"digest": digest(json.dumps(arm).encode()), "platform": {"os": "linux", "architecture": "arm64"}},
        {"digest": digest(b"{}"), "platform": {"os": "unknown", "architecture": "unknown"}},
    ]}
    upstream = Store(index, arm, blobs=[config, *layers])
    relay = relay_module.Relay("ghcr.io/owner/image@" + digest(json.dumps(index).encode()), tmp_path / "relay",
                               upstream=upstream, port=0)
    return relay, config, layers


def test_image_resolves_the_platform_manifest_to_its_configuration_and_layers(tmp_path):
    relay, config, layers = layered(tmp_path)
    try:
        loaded, listed = relay.image(digest(config))
        assert loaded == config
        assert listed == [("sha256:" + "1" * 64, digest(layers[0]), len(layers[0])),
                          ("sha256:" + "2" * 64, digest(layers[1]), len(layers[1]))]
        with pytest.raises(ValueError, match="different image configuration"):
            relay.image("sha256:" + "f" * 64)
    finally:
        relay.close()


def test_image_refuses_layers_that_are_not_registry_blobs(tmp_path):
    relay, config, _ = layered(tmp_path, foreign=True)
    try:
        with pytest.raises(ValueError, match="not loadable registry blobs"):
            relay.image(digest(config))
    finally:
        relay.close()


def test_reference_must_be_pinned_by_digest():
    with pytest.raises(ValueError, match="pinned by sha256"):
        relay_module.parse("ghcr.io/owner/image:latest")
