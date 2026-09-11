"""Select a pinned image builder; print the command unless execution is explicit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.common.profiles import ROOT, local_path, read_json  # noqa: E402


def plan(name, arguments=(), root=ROOT):
    data = read_json(root/'runtime/images/builders.json')
    if data['schema'] != 'sparkring-image-builders/v1':
        raise ValueError('Unsupported image-builder schema')
    choices = {row['id']: row for row in data['builders']}
    if name not in choices:
        raise ValueError('Select a builder: '+', '.join(choices))
    selected = choices[name]
    command = [sys.executable if selected['kind'] == 'python' else 'bash',
               str(local_path(selected['path'], root)), *arguments]
    return {'builder': name, 'command': command, 'working_directory': str(root),
            'reason': selected['reason'], 'publication': 'not performed by this dispatcher'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('builder')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        result = plan(args.builder, arguments)
        print(json.dumps(result, indent=2), flush=True)
        if args.execute:
            subprocess.run(result['command'], cwd=result['working_directory'], check=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
