"""Model-neutral R35 serving entrypoint with preserved SparkRing launch guards."""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path('/opt/sparkring')


def main():
    spec = importlib.util.spec_from_file_location('r35_verification', ROOT/'bin/verify-r35.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        print('Usage: sparkring verify | serve MODEL [vLLM options...]\nRequires a rendered SparkRing profile; no model starts by default.')
        return
    result = verifier.verify()
    if sys.argv[1:] == ['verify']:
        print(json.dumps(result, indent=2))
        return
    # Retain the model, topology, memory, transport and cache admission checks.
    # Only installed-image verification is supplied by the R35 receipt.
    spec = importlib.util.spec_from_file_location('profile_admission', ROOT/'bin/r35-profile-admission.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    command = admission.serving_argv(sys.argv[1:])
    os.execve(command[0], command, os.environ)


if __name__ == '__main__':
    main()
