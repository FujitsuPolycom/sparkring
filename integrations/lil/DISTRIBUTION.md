# Prepare model files and images

Status: **implemented** controller-to-host copying; direct rank-to-rank copying
is a developer helper. Neither path configures SSH access or network routes.

The hosts need the pinned Docker image, model snapshot, and mount directories
before `lil image check` can pass. Existing files can be staged through other
tools; the helpers below are optional.

## Download once and copy from the controller

`distribute.py` downloads each file into a checksum-addressed local cache, then
copies it to the listed hosts with SCP. Sources may be HTTPS URLs or absolute
local file paths. List every required file with its SHA-256:

```json
{
  "schema": "sparkring-artifacts/v1",
  "artifacts": [{
    "path": "target/config.json",
    "source": "/absolute/local/config.json",
    "sha256": "REPLACE_WITH_FILE_SHA256"
  }],
  "destinations": [{"host": "spark0.example.invalid", "root": "/srv/models"}]
}
```

Preview without downloading or contacting hosts:

```bash
python integrations/lil/distribute.py artifacts.json --cache /local/downloads
```

Add `--execute` to download and copy. Each listed file is verified before
publication at its destination. Matching files are reused; conflicting files are
refused. Unlisted files are not checked. Image archives still require
`docker load --input ARCHIVE` on each host.

## Direct rank-to-rank helper

`fanout.py:copy_edge` transfers one file along an explicit authenticated route,
using either a push or pull, and verifies both ends. Pull supports sites where
peer-to-peer SSH authorization works only from the destination to the source.
The controller still requires management SSH access to both hosts for setup,
hash verification, publication and cleanup. The helper has no operator CLI or
automatic route orchestration yet. A 1 MiB fixture across four ranks passed; bulk-transfer
speed and interrupted-transfer behavior need further testing.
