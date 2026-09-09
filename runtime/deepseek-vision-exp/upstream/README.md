# Pinned Compose test fixture

The compressed Compose file is copied without modification from MiaAI-Lab's
DSpark recipe at commit `7440c53c1f0352886e47b1909051784879fa0a24`.
The source URL and uncompressed SHA-256 are recorded in [profile.json](../profile.json).
[LICENSE](LICENSE) preserves its MIT license and copyright notice.

Offline tests render this fixture to check the cycle configuration. Operators
must use the complete upstream checkout because the serving command mounts
additional hotfix files; this fixture cannot launch the runtime by itself.
