# Image and download presentation plan

Status: **research-only proposal**. The operator guide is implemented in this
branch; the metadata and publication actions below are not applied. No registry
package, tag, image, Release asset or serving default is renamed or removed.

## Purpose and invariants

Operators should choose a model profile, follow its image digest, and install
only the host tools that profile requires. The GitHub sidebar should not imply
that a host-tool release is the serving-stack version.

- Preserve package names, existing tags/digests, Release tags and asset URLs.
- Preserve checksums, source receipts, licenses and historical measurements.
- Keep model/topology/cache/backend qualification attached to exact profiles.
- Do not turn a local combined-image build into a public recommendation by
  editing prose. Publication and profile promotion require their own evidence.
- No deletion is proposed. Absence of references in `main` alone cannot prove
  that published build inputs or downstream installations no longer need an image.

## Local documentation changes

The [image guide](../../runtime/images/README.md) provides the operator entry
point, separates package families from versions, identifies retained GLM names,
and explains host tools versus build archives. The main README links there
without repeating an image-version inventory. Existing `#container-images` and
`#image-construction` anchors remain available.

## Proposed GitHub presentation changes

Apply these only after approval, using a fresh metadata inventory. Save the
prior descriptions, titles, Release bodies and badge settings for rollback.

| Object | Proposed display text / action |
|---|---|
| Package `sparkring` | “Shared ARM64 SparkRing serving images. Select the exact digest and supported features through a model profile.” |
| Package `gb10-vllm-serving` | “Profile-specific GB10 serving builds, including DeepSeek. Use the matching quickstart and pinned digest.” |
| Package `sparkring-glm53-runtime` | “Retained GLM runtime/base images for pinned deployments and reproducible builds. See the SparkRing image-selection guide.” |
| Package `sparkring-glm53-sparkcache` | “Retained GLM/SparkCache compositions used by pinned deployments and build inputs. See the SparkRing image-selection guide.” |
| Release tag `r33-host-tools-c8646b0` | Display title “Host utility: RoCE forwarding marker (R33)”; lead its notes with “Host executable, not a serving container.” Preserve the existing tool bytes and qualification limits. |
| Release tag `native-runtime-sm121-aa8fa11831af` | Display title “Build inputs: SM121 NCCL and SparkCache libraries”; retain prerelease status, checksums, provenance and license assets. |
| Host/build Release badges | Where GitHub supports it, avoid assigning the **Latest** badge to auxiliary downloads. Verify the resulting repository sidebar; do not create a fake serving release just to occupy the badge. |

Link each package description or supported package README to the canonical image
guide where GitHub permits it. Confirm the editing mechanism for each metadata
field before application; do not rebuild or republish an image merely to change
its presentation. If the UI cannot expose the distinction, retain the explanatory
guide rather than relabeling runtime compatibility.

## Serving-image publication structure

Use the model-neutral `ghcr.io/fujitsupolycom/sparkring` namespace for shared
compositions where the dependency and qualification records support them. This
does not require migrating a profile that needs a separate runtime. Different
runtime environments may coexist in an image only with explicit isolation and
tests of each selected serving path.

For an approved shared-image publication:

1. Record the source revision, descriptor, runtime/dependency pins, platform,
   installed receipt, registry digest and local image ID as distinct fields.
2. Preserve a compatibility table by model, node count, TP/DCP, speculation,
   SparkCache mode and media scope. Link the measured evidence and limitations;
   distinguish untested combinations from qualified ones.
3. Publish an immutable version and verify anonymous pull plus installed-source
   identity. Do not move an existing evidence-bearing tag to different bytes.
4. Change only the qualified profiles' canonical release selections, regenerate
   compatibility/Compose exports, and retain their rollback selections.
5. If a GitHub Release accompanies it, title it as a serving-image release and
   link the GHCR digest and supported quickstarts. Do not upload a second,
   independently named image tar as the default operator download.

Do not make a universal floating `latest` tag the quickstart default. A common
namespace simplifies discovery; it does not remove dependency differences.

## Checks and rollback

Before publication, compare each retained package family against profile and
release selections, Docker `FROM` inputs, composition descriptors, frozen-input
inventories and historical guides. Record each dependent path and digest; keep
unclassified references rather than treating them as unused.

For documentation, run layout, Markdown-link, release-safety and generated-export
checks. Diff canonical profiles, publication receipts and protected inputs to
confirm the presentation-only change did not alter serving selections.

After approved metadata edits, verify the public sidebar and package/Release
pages from a signed-out session. Confirm existing digest pulls and asset URLs
still work. Revert descriptions/titles if presentation regresses; no runtime
rollback is needed because metadata edits do not change serving bytes.

Acceptance: a reader can identify where to choose an image, tell a host utility
from a serving runtime, understand why retained package names exist, and locate
an exact profile pin without consulting development history.
