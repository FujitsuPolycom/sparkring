# Command guide

Run these commands on Node A, the Spark connected to your network.

[Full reference →](../../docs/operations/commands.md)

## Everyday commands

Pick `PROFILE` from the [profile table](../../README.md#profiles).

```bash
sudo sparkring install --profile PROFILE           # set up the Sparks and start the model
sudo sparkring install --profile PROFILE --plan    # show the plan; change nothing
sudo sparkring install --profile PROFILE --on 4-7  # run on Sparks 4 to 7 of a larger ring
sudo sparkring logs --follow                       # follow the installation
sudo sparkring status                              # each Spark and the model
sudo sparkring down --execute                      # stop the model
sudo sparkring up --execute                        # start it again
sparkring images                                   # images for --image
sparkring models                                   # profiles and their short names
sudo sparkring name "Draupnir"                     # set the ring's display name
```

<!-- The All commands table of docs/operations/commands.md (scripts/compose_builder/site.py). -->

## Common install flags

| Flag | Use |
|---|---|
| `--profile PROFILE` | The model and number of Sparks |
| `--on ARC` | Run on these Sparks, such as `0,1` or `4-7` |
| `--plan` | Show the plan; change nothing |
| `--yes` | Approve without asking |
| `--checkpoint NAME` | Another checkpoint of the profile |
| `--image NAME` | Another image from `sparkring images` |
| `--reasoning-effort LEVEL`, `--thinking off` | Thinking default for requests that do not choose |
| `--api-port N` | Port of the model's API |
| `--model-path PATH` | Reuse a checkpoint copy on disk |
| `--no-auto-recover` | Do not restart the model by itself |

## Tools

- **Cabling**: `sudo sparkring cabling` shows how the Sparks are cabled and
  what to move. `sudo sparkring cabling --bandwidth` measures each cable's speed.
- **Fabric**: `sparkring fabric show` prints the recorded fabric.
  `sudo sparkring fabric verify` checks it. `sudo sparkring fabric spread-check`
  tests copying files along the cables. `sudo sparkring fabric tune --execute`
  measures SIRCL's tuning table.
- **Check**: `sudo sparkring check` tests the running model. Add
  `--report DIR` to write a report for a test-report issue.
- **Storage**: `sudo sparkring storage` reports disk use and what can be released.
- **Checkpoints**: `sudo sparkring checkpoints` lists checkpoint directories.
- **Hairpin**: `sudo sparkring hairpin --plan` shows the ConnectX setting relays need.
- **Host load**: `sudo sparkring host-load show --all` shows when host jobs run.

## Dashboard

A read-only page of the ring and every model: state, output rate, requests,
KV cache and settings. Open `http://NODE_A:8090/` in a browser (Experimental).

```bash
sparkring dashboard tui              # the same view in a terminal
sudo sparkring dashboard quiet on    # pause its requests for a benchmark; quiet off resumes
sudo sparkring dashboard disable     # turn it off
sudo sparkring dashboard enable      # turn it on
sudo sparkring dashboard export      # rewrite its site file from the deployments
```

[Full reference →](../../docs/operations/commands.md)
