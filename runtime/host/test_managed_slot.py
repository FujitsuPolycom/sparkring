import json
from types import SimpleNamespace

from runtime.host.managed_slot import inspect_slot


def test_empty_slot_requires_no_docker_calls(tmp_path):
    assert inspect_slot("new", "sha256:a", 0, root=tmp_path)["available"]


def test_existing_unidentified_mesh_is_not_replaced(tmp_path):
    (tmp_path / "etc/sparkring/managed-mesh").mkdir(parents=True)
    assert not inspect_slot("new", "sha256:a", 0, root=tmp_path)["available"]


def test_matching_deployment_may_resume_but_foreign_model_cannot(tmp_path):
    path = tmp_path / "etc/sparkring/managed-mesh/service.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"container_id": "a" * 64, "rank": 0, "container_image": "sha256:a"}))
    def run(*a, **kw):
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"Name": "/old-r0", "Image": "sha256:a",
                              "Config": {"Labels": {"io.sparkring.container-spec": "glm-tp4/v1"}}}]))
    assert inspect_slot("old", "sha256:a", 0, root=tmp_path, run=run)["available"]
    assert not inspect_slot("new", "sha256:a", 0, root=tmp_path, run=run)["available"]
