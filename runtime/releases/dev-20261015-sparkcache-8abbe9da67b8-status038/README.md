# SparkCache CRC32 development image over release 2026.10.4

Status: **research-only**. One local build is recorded in the
composition's [`built-image.json`](../../images/compositions/installer-sparkcache-8abbe9da67b8/built-image.json)
(image ID `sha256:22cf6e030a74201f38cfe6569bc475f430bb9e0358bdaf8297b87ddd71b7f99c`).
It is not published and has not been qualified for serving.
The [source layer](../../images/compositions/installer-sparkcache-8abbe9da67b8/README.md)
pins SparkCache `8abbe9da67b8143831d21b43d0395c4f347a619a` and generates a
receipt-checked Docker context. Its [build inputs](../../images/compositions/installer-sparkcache-8abbe9da67b8/build-inputs.json)
list the parent's fifteen profiles and the two Experimental GLM variants:
`glm53-flash-nvfp4-sparkcache-crc32-tp2` and
`glm53-flash-nvfp4-sparkcache-crc32-tp4`.

The build recipe writes `installer-image.json` after inspecting a built image
and admitting every listed profile. A release selection adds it here with the
published image reference. This directory is not a release selection and
changes no published lock. Qwen and DeepSeek SparkCache variants require
additional model-specific implementation and qualification.
