# SparkRing runtime

Choose deployments through [profiles/](../profiles/README.md). The runtime tree
separates shared behavior from image selection and frozen release inputs.

| Owner | Responsibility |
|---|---|
| [common/](common/) | Configuration resolution, site parsing and guarded command adapters |
| [images/](images/README.md) | Pinned builder selection |
| [releases/](releases/README.md) | Exact release selection and preserved inputs |

Version-specific directories remain compatibility locations when published
build inputs, source manifests or installed paths depend on them. Their names
are stable filesystem and interface locators, not recommendations. The
[retained composition index](../docs/history/runtime-compositions.md) records
historical builder and rollback details. See the [layout guide](../docs/development/layout.md)
before introducing another implementation directory.

## Components

See [image builders](images/README.md) and [component ownership](../docs/development/layout.md).

## GLM-5.3 Flash operator image

Select a profile from the catalog; [retained operator details](../docs/history/runtime-compositions.md#glm-53-flash-operator-image)
apply only to their named image and checkpoint.

## DeepSeek-V4-Flash-0731

Use the [profile catalog](../profiles/README.md); pair and cycle configurations
remain distinct. Their engine dependencies are not interchangeable with GLM.

## Scope and safety

Resolution and command planning are offline. Building changes the local
container store. Starting, replacing or distributing a deployment requires
applicable host authorization and the selected guide's checks.
