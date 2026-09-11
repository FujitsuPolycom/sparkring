# Write Without Hidden Context

Write for a technically capable reader who has the repository but none of the
conversation history. This policy applies to documentation, comments,
docstrings, errors, plans, commits, PR descriptions and technical summaries.

Explain purpose, behavior, invariants, interfaces, evidence and limitations.
Introduce a concept before an internal identifier; use hashes, manifests and
revisions when readers need them to operate or verify the system. A version or
date can identify a release, but a development label such as “the next phase”
does not identify a component's responsibility.

Canonical documentation specifies present behavior. Replace stale statements;
do not append a debugging chronology. Historical explanations belong in an
explicitly historical record. Avoid references such as “the experiment” unless
the experiment has been identified in the same document.

Use implemented, qualified, research-only and unsupported with a clear scope.
A useful implementation can merge before hardware qualification. A retired
profile can retain valid historical evidence. State measurements as conditions,
measurement, result, conclusion and limitations. Configured intent, observed
activation, correctness and performance are different kinds of evidence.

Comments explain non-obvious intent or invariants. TODOs name the missing
condition and the criterion for removing the TODO. Commits and PRs explain the
resulting behavior, technical reason, compatibility and validation.

Use ordinary language. Do not attach a formal status to every paragraph or ban
ordinary words with a regex. Prose review is advisory unless ambiguity changes
the technical claim or operation. If understanding a sentence requires having
been in the chat, rewrite it.
