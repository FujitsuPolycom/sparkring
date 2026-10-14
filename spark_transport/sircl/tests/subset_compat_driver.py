"""Hash ordered JSON setup records and plans with fixed build provenance.

The fixture was generated with this driver against df86c93d, before subset
support. CUDA is mocked; the real configuration, route and plan code runs.
"""

import hashlib
import json
import sys
from types import SimpleNamespace

from teardown_driver import _CudaStandIn

sys.meta_path.insert(0, _CudaStandIn())

from plan_driver import configured
from sparkring_sircl import routes
from sparkring_sircl.oneshot import runtime
from sparkring_sircl.vllm import fabric
from sparkring_sircl.vllm.serve import bundle, plan
from sparkring_sircl.vllm.serve.sitefile import ServeSite
from test_vllm_serve import SITE


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True).encode()).hexdigest()


def records():
    result = {}
    for n in range(4, 9):
        for k in range(3, n + 1):
            text = f"ring:{n}:" + ",".join(map(str, range(k)))
            layout = routes.Layout.parse(text)
            derived = routes.derive_routes(layout)
            maps = [derived.route_map(i) for i in range(k)]
            session = configured(runtime, routes, text, 0)
            session._proxy = None
            session._layout = SimpleNamespace(slots=runtime.proto.SLOTS, flag_stride=runtime.proto.FLAG_STRIDE)
            group = fabric.describe_group(fabric.Layout.ring(n), list(range(k)), parent=list(range(n)))
            for schedule in ("auto", "pieces", "chain", "ring"):
                settings = plan.session_settings(group, name="tp", groups="fixed", scoped=True,
                                                  schedules={"large_schedule": schedule})
                result[f"{text}/{schedule}"] = digest(settings.to_json())
            result[f"{text}/record"] = digest(session._setup_record())
            result[f"{text}/lanes"] = digest(maps)
            result[f"{text}/windows"] = digest([routes.forward_windows(layout, maps, r) for r in range(k)])
    for positions in ((0, 1), (0, 1, 2, 3), tuple(range(8))):
        for schedule in (None, "pieces"):
            built = bundle.build_bundle(ServeSite.from_json(SITE),
                                         bundle.BundleOptions(positions=positions, large_schedule=schedule),
                                         staged_digest="fixed-source-digest", library="fixed-library.so")
            result[f"bundle/{positions}/{schedule}"] = digest(built.to_json("/etc/sircl/site.json"))
    return result


if __name__ == "__main__":
    print(json.dumps(records(), indent=2))
