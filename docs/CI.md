# Continuous integration

Status: **implemented**. `.github/workflows/ci.yml` runs on pull requests
and pushes to `main`. Feature-branch pushes without a pull request do not
run this workflow. Jobs use read-only repository permissions.

- `lint` checks maintained Python source with Ruff.
- `tests` runs GPU-free contracts, including serving configuration and
  measurement-receipt checks. README tests check navigation rather than
  requiring a particular prose or table layout.
- `pinned LIL bridge` builds the source-pinned companion CLI and exercises
  its SparkRing integration.
- `profile and repository contracts` validates deployment profiles, generated
  exports, references, immutable release inputs, and image-builder contracts.
- `docs links` checks tracked inline Markdown links and ATX heading anchors,
  including duplicate-heading suffixes. Front-page lists without a separating
  blank line produce advisory warnings. External links, reference-style links,
  HTML anchors, and rendered visual layout are not validated by this checker.
- `release safety` scans tracked nonbinary files for configured site-address
  and credential shapes. It prints only path, line number, and rule identifier.
  Findings exit with status 1; scan failures exit with status 2. The rule file
  itself is excluded. This bounded pattern scan is not a security certification.

Run the documentation and release checks locally from the repository root:

```bash
python scripts/check_markdown_links.py .
python scripts/check_release_safety.py .
python scripts/check_repository_layout.py
python -m pytest scripts/test_ci_checks.py scripts/test_glm53_flash_profile.py -q
```

The full lint and test commands are defined in the workflow. Hardware-dependent
skips are reported explicitly; a green CPU run does not qualify CUDA, RDMA,
model output, or live deployment performance.

GitHub branch-protection settings are separate from the workflow file.
The intended merge policy requires all jobs listed above, an up-to-date branch,
and enforcement for administrators. Inspect repository settings to confirm
that policy; changing this document does not configure GitHub protection.
