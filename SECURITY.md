# Security Policy

No released version receives a security-maintenance guarantee.
Accepted fixes land on the `main` branch.

## Reporting a vulnerability

Report suspected vulnerabilities privately through a
[GitHub security advisory](https://github.com/FujitsuPolycom/sparkring/security/advisories/new)
("Report a vulnerability" under the repository's Security tab), not in a
public issue. Include enough detail to reproduce the problem where possible.

Reports are acknowledged as promptly as we can manage, and any fix and
disclosure are coordinated through the advisory. There is no bug bounty
program.

## Deployment exposure

A SparkRing deployment's model API has no authentication and listens on every
interface of the Spark that serves it. Setup also adds an administration
network and SSH service. The
[install guide](docs/operations/install.md#security) and the
[install reference](docs/operations/install-reference.md#security-and-host-exposure)
describe what is exposed and how to restrict it.
