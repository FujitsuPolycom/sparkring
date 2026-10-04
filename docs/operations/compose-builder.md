# Install Builder

The Install Builder is a web page for one SparkRing source. Pick how your
Sparks serve, a profile, a checkpoint and serving settings, and it writes:

- the `sparkring install` commands that deploy them, switch layouts and
  check the result, in order, or
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

Serve that directory over HTTP, for example with
`python -m http.server --directory .sparkring/compose-builder`, and open the
address it prints. Opened as a local file, the page offers only the default
installer image: each other image's data is a file beside the page,
`images/NAME.json`, which a browser loads only over HTTP.
`--default-image-only` writes no image files. `--verify` first compares the
page's file generator with SparkRing's own and needs Node.js
([how the page is checked](#how-the-page-is-checked)).

## Installer image

Step 1 offers the installer images that run the profiles the layout installs,
as [`sparkring images`](commands.md#images) lists them, the default first.
Another image adds `--image NAME` to every install command whose profile it
runs, and to the `render` command in the zip's README, and the Compose files
use that image's digest. A command whose profile the image does not run uses
the default image, and the page says so. `NAME` is the release tag that
published the image, such as `2026.09.5`, or the part of its name that only it
has, such as `statusrows`
([another image](install-reference.md#another-image)).

## Layout

Step 2 chooses how the Sparks serve:

| Layout | Models | Profiles listed in step 3 |
|---|---|---|
| Two Sparks | One model on a pair | Two-Spark profiles |
| Four Sparks – One Model | One model on a four-Spark ring | Four-Spark profiles |
| Four Sparks – Two Models | One model on each half of a ring: Sparks 0 and 1, and Sparks 2 and 3 ([two models on one ring](install-reference.md#two-models-on-one-ring)) | Two-Spark profiles, for each pair |

With two pairs, steps 3 and 4 show one pair at a time behind **First pair**
and **Second pair** tabs. Each pair keeps its own profile, checkpoint,
serving settings and [API endpoint](#api-endpoint), so the pairs can run
different models and checkpoints. The switch commands follow the deployed
models: two pairs switch to the four-Spark profile of the first pair's model,
and one model on all four switches both pairs to that model's two-Spark
profile, each with its default checkpoint, serving settings and the automatic
API endpoint.

The page offers **Four Sparks – Two Models** only when its source's
`sparkring install` takes `--on`. When building the page,
[export.py](../../scripts/compose_builder/export.py) reads the options that
the argument parsers of `runtime/host/install_workflow.py` and
`runtime/host/cabling.py` define, and the serving settings that
`runtime/common/serving.py` defines, and records them in the page's data as
`features`.

## Commands

For the `sparkring install` output the page writes the layout's commands in
order, under three headings. Every command runs on Node A, as a user with
sudo. Each one says what it does and, for an installation, the address its
model serves at, with a copy button.

| Heading | Two Sparks | Four Sparks – One Model | Four Sparks – Two Models |
|---|---|---|---|
| Install | The profile on both Sparks | The profile on all four | `--on 0,1` for the first pair, then `--on 2,3` for the second |
| Switch | None | The two pairs' commands, `--on 0,1` then `--on 2,3`; the first stops the four-Spark model | The four-Spark profile, which stops both pairs' models |
| Check | `sudo sparkring status` | `sudo sparkring status` | `sudo sparkring status`, which shows each pair's model separately |

- A model serves on the first Spark of its Sparks, at its profile's port from
  the page's data: Node A for a pair, a ring and the first pair, and Spark 2's
  own LAN address for the second pair. Without its own LAN connection, Spark 2
  is reachable only from Node A.
- SparkRing runs one installation at a time, so each install command runs
  after the one before it has finished.
- When the source's `sparkring cabling` takes `--bandwidth`, Check also lists
  `sudo sparkring cabling --bandwidth`, which measures each cable and skips the
  cables a running model uses.

The options above the commands apply to every install command:

| Option | Commands |
|---|---|
| New installation | The first command is the [one-command installer](install-reference.md#get-the-package) on Node A, pinned to the page's source with `--ref`, or from `main`. It passes `--on` and every other option to `sparkring install`. Later commands run the `sudo sparkring install` it installed. |
| SparkRing already installed | Every command is `sudo sparkring install …` |
| Approval | No flag (asks before changes), `--plan` or `--yes` |
| Spark order | Automatic, Ask during install or Fill in my Sparks ([below](#spark-order)) |
| Download limit | `--download-limit`, such as `850Mbit` |

Each install command carries `--profile`, `--on` for a pair of a ring,
`--image` and `--checkpoint` when they are not the defaults, the
[serving settings](install-reference.md#serving-settings) changed for that
selection and its [API endpoint](#api-endpoint). A setting outside its limits
shows its problem under the field.

### API endpoint

Step 4 ends with **API endpoint**: where the model's API listens and the
address shown for it ([API endpoint](install-reference.md#api-endpoint)). It
belongs to each model, and with two pairs to each pair.

| Choice | Install command | Compose files |
|---|---|---|
| Automatic (default) | No option: every address of the Spark that serves the model, at its profile's port | The same |
| Ask during install | No `--yes`, so `sparkring install` lists that Spark's addresses and asks which one and which port to use | Automatic; the page says so |
| Set here | `--api-port`, `--api-bind` and `--api-address` from the **Port**, **Listens only on** and **Address people use** fields | The port and the listen address in the API rank's command and health check; the address in the API line and the `curl` check of the README and the page |

- **Port** takes 1024 to 65535 and refuses the ports SparkRing uses, such as
  2222 and the profile's `--master-port`. **Listens only on** takes an IPv4
  address of the Spark that serves the model; the installer checks that the
  Spark has it. **Address people use** is a name or an address, such as
  `llm.example.net`, without `http://` or a port. A value the installer would
  refuse shows its problem under the field and stays out of the commands.
- The address each install command serves at follows the choice: the address
  people use, else the listen address, else the Spark's address, at the
  chosen port.
- The switch commands use the automatic endpoint.
- The page offers each field only when its source's `sparkring install`
  takes the option, and **Ask during install** only when it takes
  `--api-port` or `--api-bind`.

### KV cache estimate

Under **KV cache per Spark** in step 4, and as **KV cache** in the summary of
each model above the commands, the page estimates how many tokens the chosen
KV cache holds. It scales the engine-reported pool that
[profile-capacity.json](../../performance/profile-capacity.json) records for
the profile, with the KV bytes per Spark at that measurement:

- tokens ≈ measured tokens × chosen KV bytes per Spark ÷ measured KV bytes
  per Spark, rounded to two significant figures;
- full-context requests ≈ tokens ÷ the context window, to one decimal.

A measurement of the selected checkpoint is used when the file has one;
otherwise the profile's own, and the note names the checkpoint it measured.
Each record belongs to one profile, so a two-Spark measurement never sizes a
four-Spark profile. A model without a measurement shows "Not measured for
this model". The estimate follows the KV cache and context fields as they
change; the engine reports the exact figure when the model starts.

### Spark order

Node A is the Spark that runs setup, which the first installation does; it
becomes rank 0. The cabling ranks the others: every cable joins port 0 of one
Spark to port 1 of the next, and rank r+1 is the Spark on rank r's port 0. So
the first pair is Node A and the Spark on its port 0, and the second pair is
the other two, headed by Spark 2. Neither the order nor a pair's head can be
chosen.

- **Automatic** writes the commands as the options above set them and names
  the Sparks Node A and Spark 1 to Spark 3. `NODE_A` and `SPARK_2` stand for
  their addresses in the API addresses.
- **Ask during install** leaves out `--yes`, so every installation shows which
  Spark is which and asks before it changes anything.
- **Fill in my Sparks** takes a name, an address or both for each Spark in
  step 5. The commands say where to run by name, and the API addresses use the
  addresses, or the names without one.

A list under the commands says what each Spark becomes.

## Share the choices

On a page served over HTTP, **Copy link to these choices** copies the page's
address with the layout, each selection's profile, checkpoint, changed
serving settings and API endpoint choice, the installer image, the install
options, the Spark order and the output. The link carries no name, address or path from step 5. A link
without a layout opens the layout that fits its profile's size.

## Compose files

Docker Compose covers a pair and a ring serving one model. For two pairs on a
ring the page writes the install commands only: a ring half uses port 0 of
one Spark and port 1 of the other, and needs the ring's mesh stopped first,
and the Compose files cover neither.

Step 5 of the page takes your Sparks. For the commands it takes each Spark's
name and address. For Docker Compose it takes the whole site: SSH host, IP
address, bootstrap interface, RoCE HCAs and GID index, and the model, cache,
checkout and deployment directories. Four-Spark profiles also take the mesh
fabric reference. **Paste a site file instead** reads an existing
`sparkring-compose-site/v1` file. A value the generator refuses shows its
problem under the field, and a collapsed step 5 counts the fields that need a
change. While step 5 holds the example values, the download area says so.

**Download *name*.zip** saves the deployment under the name beside it, which is
also in step 5. By default it is the profile's name for its default
checkpoint, such as `qwen38-flash-next-tp2.zip`, and otherwise names the
checkpoint, such as `glm53-nvidia-nvfp4-tp4.zip`:

```text
NAME/deployment.json
NAME/rank0/compose.yaml    NAME/rank0/container.json    (one folder per rank)
NAME/NAME.site.yaml
NAME/README.txt
```

- The folder is what `sparkring compose render` writes for that site file,
  with the same `--image`, `--checkpoint` and serving-setting flags, the API
  endpoint's `--api-port` and `--api-bind` among them. The README names the
  command and the model's API address.
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
- The page lists the [Compose profiles](compose.md#supported-profiles) that
  `sparkring install` deploys; it omits the SparkCache Qwen profiles,
  `qwen38-flash-next-tp2-sparkcache` and `qwen38-flash-next-qad-tp4-sparkcache`.
- Generated files carry no serving qualification, like every `compose render`
  deployment.

## How the page is checked

The page runs no Python. For every installer image, profile, checkpoint and
save-CPU state, [export.py](../../scripts/compose_builder/export.py) renders the deployment
with `compose.build` for a site whose values are unique placeholders, and
[engine.js](../../scripts/compose_builder/engine.js) puts a real site's values
in their place line by line. The export stops if a placeholder lands anywhere
the engine does not rewrite.

[verify.py](../../scripts/compose_builder/verify.py) renders random sites,
checkpoints and settings, the API endpoint's port and listen address among
them, on every image with the engine under Node.js and with `compose.build`,
and requires them to match byte for byte. `--cases` sets the
sites per checkpoint on the default image, `--image-cases` on each other image:

- every rank's `compose.yaml` and `container.json`, `deployment.json`,
  `site.yaml` and the deployment ID;
- the refusal message for each invalid site and setting, such as a port that
  SparkRing uses or an address no Spark can listen on;
- for a sample, the zip: Python's `zipfile` opens it and
  `compose.load_deployment` accepts the unzipped folder.

```bash
python -m pytest scripts/compose_builder -q
```

The test compares the engine only where Node.js is installed. CI's
`Install Builder` job requires it, and `generate_compose_builder.py --verify`
writes no page when any case differs.
