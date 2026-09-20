# Triage

Review open issues on the GitHub tracker against the live repository and produce a disposition per issue: closeable (already fixed, documented, invalid, or transferred), needs a progress comment splitting completed from remaining work, or stays open untouched. Review only by default; any close/comment/PR requires explicit user authorization in the session before it is executed.

## Arguments

`$ARGUMENTS`: optional repository (`owner/name`); default to the `origin` remote of the current checkout.

## Procedure

### 1. Inventory

Fetch all open issues with labels, authors, and timestamps:

```bash
gh issue list -R <repo> --state open --limit 100 --json number,title,author,labels,createdAt,updatedAt
```

Read every issue body and comment thread (comments may contain decisive context — reporter progress, maintainer dispositions, changed scope). Never disposition from title alone.

### 2. Ground truth

For each claim an issue makes about repository behavior, verify against the current checkout: grep for the referenced symbol/flag/template, read the cited docs section, check the changelog entry. An issue is closeable as *completed* only with a named file, symbol, or test present in the tree — not from a commit message. Note the exact line ranges you relied on; the disposition must cite them.

### 3. Disposition

Assign exactly one per issue, with the evidence above:

- **close / completed** — the requested change exists in the tree; the close comment names the implementation with file:line references and any residual scope that moved to another issue.
- **close / not planned** — the premise is wrong (misidentified mechanism, out-of-range value, unread knob) or the maintainer would reject it; the close comment must state the corrected mechanism precisely and, when part of the request remains valid, link or transfer that remainder to the owning issue *in its body*, not by silently patching history.
- **progress comment** — implementation partially landed: split the thread into completed work (with evidence) and remaining work (specific, owned); the remaining work is what keeps it open.
- **stays open** — active investigation, untested knob, awaiting reporter data, or maintainer-run experiment; say what would move it.
- **scope error** — the issue body's literal scope (e.g. which templates or profiles it names) contradicts the assumed scope; report the discrepancy and, when a related lane has no tracker entry, propose a new linked issue describing it as a draft rather than folding it in.

### 4. Close-gates

If a disposition depends on an artifact that does not exist yet (a docs PR, a follow-up issue), create the artifact first, or hold the disposition and list it as gated. Never close citing a future action.

### 5. Report

Output a table: `#issue | disposition | evidence (file:line) | gate (if any)`. Follow with a proposed execution batch in dependency order (create follow-up → body amendment → close comments → closes), and stop. Do not execute without explicit user authorization.

## Rules

- Respect the requested scope and existing authorization. A review is not permission to post.
- Writing style for any drafted comment: concrete evidence, exact paths and line ranges, status labels ("Closing as not planned for three of the four settings…"), no conversation-history references, complies with the repository's AGENTS.md prose rules.
- One primary disposition per issue; fold secondary observations into that disposition's notes rather than spawning parallel verdicts.
