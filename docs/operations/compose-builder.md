# Compose builder

The Compose builder is a web page for one SparkRing source. Pick a profile, a
checkpoint and serving settings, and it writes:

- the `sparkring install` command that deploys them, or
- one Docker Compose file per Spark, exactly as `sparkring compose render`
  writes them for your site, and the whole deployment as a zip.

The page runs in your browser. The addresses and paths you type never leave it.

## Open it

The [Pages workflow](../../.github/workflows/compose-builder-pages.yml)
publishes the page for each release at
`https://fujitsupolycom.github.io/sparkring/`. The repository needs GitHub
Pages enabled with GitHub Actions as its source; the address needs no custom
domain. To publish the page for another tag or commit:

```bash
gh workflow run compose-builder-pages.yml --ref main -f ref=TAG_OR_COMMIT
```

To build the page for your checkout instead:

```bash
python scripts/generate_compose_builder.py --output .sparkring/compose-builder
```

Then open `.sparkring/compose-builder/index.html` in a browser.
`--verify` first compares the page's file generator with SparkRing's own and
needs Node.js ([how the page is checked](#how-the-page-is-checked)).

## Install command

The page writes the command for the choices on the left: `--profile`,
`--checkpoint` when it is not the profile's default, and the
[serving settings](install-reference.md#serving-settings) you changed.

| Option | Command |
|---|---|
| First install, on Node A | The [one-command installer](install-reference.md#get-the-package), pinned to the page's source with `--ref`, or from `main` |
| SparkRing already installed | `sudo sparkring install …` |
| Approval | No flag (asks before changes), `--plan` or `--yes` |
| Download limit | `--download-limit`, such as `850Mbit` |

The SparkCache Qwen profile does not run from `sparkring install`; the page
offers its Compose files only.

## Compose files

Step 4 of the page takes your Sparks: SSH host, IP address, bootstrap
interface, RoCE HCAs and GID index, and the model, cache, checkout and
deployment directories. Four-Spark profiles also take the mesh fabric
reference. **Paste a site file instead** reads an existing
`sparkring-compose-site/v1` file.

**Download *name*.zip** saves the deployment named in step 4, by default the
profile's name, such as `qwen38-flash-next-tp2.zip`:

```text
NAME/deployment.json
NAME/rank0/compose.yaml    NAME/rank0/container.json    (one folder per rank)
NAME/NAME.site.yaml
NAME/README.txt
```

- The folder is what `sparkring compose render` writes for that site file,
  with the same `--checkpoint` and serving-setting flags. The README names the
  command.
- `sparkring compose check --deployment NAME` accepts the unzipped folder.
- Files have mode 0600 in 0700 folders, as `render` writes them.
- **Copy as shell command** writes one rank's `compose.yaml` on its Spark.

The page lists the start steps: pull the image by digest, put the checkpoint in
each model directory, check the RoCE GID index, start the workers before
rank 0. [Compose deployments](compose.md) covers checking and starting the
files with `sparkring compose`. A derived checkpoint, such as
`qad-step5500-mxfp8-attention`, is written only by `sudo sparkring install
--checkpoint` ([derived checkpoints](install-reference.md#derived-checkpoints)).

## Limits

- The page describes the profiles, checkpoints and limits of the source it was
  built from. Its first step names that source.
- Paths may use letters, digits and `. _ / + = @ ~ -`; addresses are IPv4.
  `sparkring compose render` accepts more; the page refuses those values
  instead of encoding them differently.
- Pasted site files are read as YAML 1.2: an unquoted `on` or `yes` stays a
  string, which SparkRing's own reader would read as true.
- The page lists every [Compose profile](compose.md#supported-profiles) except
  `qwen38-flash-next-qad-tp4-sparkcache`.
- Generated files carry no serving qualification, like every `compose render`
  deployment.

## How the page is checked

The page runs no Python. For every profile, checkpoint and save-CPU state,
[export.py](../../scripts/compose_builder/export.py) renders the deployment
with `compose.build` for a site whose values are unique placeholders, and
[engine.js](../../scripts/compose_builder/engine.js) puts a real site's values
in their place line by line. The export stops if a placeholder lands anywhere
the engine does not rewrite.

[verify.py](../../scripts/compose_builder/verify.py) renders random sites,
checkpoints and settings with the engine under Node.js and with
`compose.build`, and requires them to match byte for byte:

- every rank's `compose.yaml` and `container.json`, `deployment.json`,
  `site.yaml` and the deployment ID;
- the refusal message for each invalid site;
- for a sample, the zip: Python's `zipfile` opens it and
  `compose.load_deployment` accepts the unzipped folder.

```bash
python -m pytest scripts/compose_builder -q
```

The test compares the engine only where Node.js is installed. CI's
`Compose builder` job requires it, and `generate_compose_builder.py --verify`
writes no page when any case differs.
