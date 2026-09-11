"""Dispatch deployment commands through one profile-selected entry point.

The default prints an argv plan. --execute is an explicit local action; existing
rank launchers retain their receipt, memory guard and host lifecycle gates.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

from runtime.common.profiles import ROOT, load, local_path, resolve


def plan(profile_id, arguments, root=ROOT):
    p, _ = load(profile_id, root)
    resolved = resolve(profile_id, root=root)
    adapter = p['launcher']
    if adapter['kind'] == 'guide':
        raise ValueError(f"{profile_id}: this composition requires the staged procedure in {p['guide']}")
    args = list(arguments)
    if args[:1] == ['--']:
        args.pop(0)
    if not args or args[0] not in adapter['actions']:
        raise ValueError('Select an explicit action: '+', '.join(adapter['actions']))
    path = local_path(adapter['path'], root)
    if adapter['kind'] == 'python':
        command = [sys.executable, str(path), *args, *adapter['fixed_args']]
    elif adapter['kind'] == 'bash':
        command = ['bash', str(path), *args, *adapter['fixed_args']]
    else:
        raise ValueError('Unsupported launcher kind')
    # Public R33 profiles must be planned with their release receipt, not the
    # source-image defaults used by the compatibility command when omitted.
    if adapter.get('receipt'):
        if '--runtime-receipt' in args or any(a.startswith('--runtime-receipt=') for a in args):
            raise ValueError('The catalog owns the runtime receipt; select another profile for another release')
        command += ['--runtime-receipt', str(local_path(adapter['receipt'], root))]
    return {'profile': profile_id, 'status': resolved['status'], 'guide': p['guide'],
            'command': command, 'working_directory': str(root),
            'effect': 'host-action' if args[0] in ('create', 'start', '--run') else 'adapter-check-or-plan'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('profile')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        result = plan(args.profile, args.arguments)
        print(json.dumps(result, indent=2), flush=True)
        if args.execute:
            # Runtime environment belongs to explicit profile/site inputs. Keep
            # only process infrastructure needed to find tools and temporary files.
            environment = {k: v for k, v in os.environ.items() if k in {
                'PATH', 'HOME', 'USERPROFILE', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'LANG', 'LC_ALL'}}
            subprocess.run(result['command'], cwd=result['working_directory'], env=environment, check=True)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
