# SparkRing runtime

Choose deployments through [profiles/](../profiles/README.md). The runtime tree
separates shared behavior from image selection and frozen release inputs.

| Owner | Responsibility |
|---|---|
| [common/](common/) | Configuration resolution, site parsing and guarded command adapters |
| [host/](host/) | The installer and host lifecycle: `sparkring install` and `sparkring setup`, fabric enrollment, the administration network, model switching, status and automatic recovery |
| [images/](images/README.md) | Pinned builder selection |
| [releases/](releases/README.md) | Exact release selection and preserved inputs |

Version-specific directories remain compatibility locations when published
build inputs, source manifests or installed paths depend on them. Their names
are stable filesystem and interface locators, not recommendations. See the
[layout guide](../docs/development/layout.md) before introducing another
implementation directory.

## Components

[builders.json](images/builders.json) lists each version-specific builder
directory; the [layout guide](../docs/development/layout.md) assigns every
other path.

## Builders outside the installer

`sparkring install` uses none of the builders and images below. Each heading
keeps an existing link to this page working; the linked details apply only to
the image, checkpoint and profile they name.

### GLM-5.3 Flash operator image

A four-Spark GLM-5.3 Flash image for retired catalog profiles
([retained operator details](../docs/history/runtime-compositions.md#glm-53-flash-operator-image)).

### GLM-5.2 EXL3 R7 builder

The [builder](exl3-r7/README.md) of the retired GLM-5.2 EXL3 3.5-bpw profile
([retained details](../docs/history/runtime-compositions.md#glm-52-exl3-r7-builder)).

### Faststart lock

[faststart-lock.json](faststart-lock.json) pins the base image and GLM-5.2
model identity of that builder
([retained details](../docs/history/runtime-compositions.md#faststart-lock)).

### Qwen3.8-27B builder

The [builder](qwen38/README.md) of the Qwen3.8-27B EXL3 K5/K6 pair and
four-Spark catalog profiles.

### Public overlay

[build-public-overlay.py](build-public-overlay.py) assembles the Python overlay
that the GLM-5.2 EXL3 builder copies into its image
([retained details](../docs/history/runtime-compositions.md#public-overlay)).

### DeepSeek-V4-Flash-0731

Use the [profile catalog](../profiles/README.md); pair and cycle configurations
remain distinct. Their engine dependencies are not interchangeable with GLM.

## Scope and safety

Resolution and command planning are offline. Building changes the local
container store. Starting, replacing or distributing a deployment requires
applicable host authorization and the selected guide's checks.
