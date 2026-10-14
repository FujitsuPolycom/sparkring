"""Other names that select an installer profile, and the profile ID each stands for.

This module imports nothing outside the standard library, so the profile generator
(scripts/generate_profiles.py) and the host CLI (runtime/host/models.py) share one table without the
generator importing the installer.
"""

# Other names that select an installer profile. A deployment, its records and its status keep the
# profile's ID, so every command replaces an alias with that ID before it reads anything else. The IDs
# of the GLM-5.3-Flash profiles of two and four Sparks name their NVFP4-Spark checkpoint, which they
# install only on an image whose vLLM cannot read their CSF checkpoint
# (profiles/glm53-checkpoints.md#default-checkpoint-by-profile); an alias names the model and the
# number of Sparks alone; the alias of the pair profile's decode-context-parallel variant adds its
# parallelism. The IDs of the Qwen3.8-Flash-Next profiles of four and eight Sparks name the QAD
# checkpoint, which the two-Spark profile qwen38-flash-next-tp2 installs too; their aliases follow that
# profile's ID.
ALIASES = {"glm53-flash-tp2": "glm53-flash-nvfp4-spark-tp2", "glm53-flash-tp4": "glm53-flash-nvfp4-spark-tp4",
           "glm53-flash-tp2-dcp2": "glm53-flash-nvfp4-spark-tp2-dcp2",
           "qwen38-flash-next-tp4": "qwen38-flash-next-qad-tp4", "qwen38-flash-next-tp8": "qwen38-flash-next-qad-tp8"}


def canonical(value):
    """The profile ID that ``value`` selects: the ID that an alias of ALIASES stands for, else ``value``."""
    return ALIASES.get(value, value)
