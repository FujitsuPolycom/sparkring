# Write Without Hidden Context

All repository prose—documentation, comments, docstrings, commit and PR text,
plans, reports, names, errors, and technical summaries—must make sense to a
technically capable reader who has the repository but none of the conversation
or development history.

- Describe the system as it exists. State its purpose, behavior, invariants,
  interfaces, evidence, and limitations directly. Do not narrate the journey.
- Do not use lifecycle labels as identities: Phase 2, pilot, next,
  current, new, old, latest, and similar terms are not technical names.
- Do not use false definite references. Phrases such as the 1M-token capture,
  the experiment, or this approach are invalid unless the exact object was
  introduced locally and unambiguously. Counts, dates, and versions are
  attributes, not identities.
- On first reference, give the object's semantic role and, when relevant, its
  durable identifier: artifact name, path, schema, revision, manifest, or hash.
- Explain concepts before identifiers. Do not make internal codenames,
  experiment labels, profile numbers, implementation shorthand, or names such
  as XOR-Cheb-T12 the vocabulary of the design. Mention literal identifiers
  only after describing what they mean and only when the reader must use them.
- Canonical documentation is a present-state specification, not a changelog.
  Replace stale claims instead of layering history on top. Put chronology,
  rejected attempts, and retrospectives only in explicitly historical
  documents.
- Label status explicitly as implemented, qualified, research-only, or
  unsupported. State evidence as conditions, measurement, result, and
  conclusion—not as a story.
- Comments explain invariants, intent, and non-obvious constraints, never change
  history. TODOs must name the missing condition and removal criterion.
- Commits and PRs state the resulting behavior, technical reason, compatibility
  impact, and validation. They do not recount attempts or pivots.

Final test: if understanding any sentence requires "you had to be there,"
rewrite it.
