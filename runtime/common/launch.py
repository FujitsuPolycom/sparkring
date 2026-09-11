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
    modified = False
    if adapter['path'] == 'runtime/common/tp2.py':
        # The adapter accepts argparse abbreviations; apply the same ownership
        # checks to abbreviated flags so they cannot change the composition.
        kv_values = []
        for index, argument in enumerate(args):
            option, separator, value = argument.partition('=')
            if not option.startswith('--'):
                continue
            if '--r33-sparkcache'.startswith(option):
                raise ValueError('The catalog owns SparkCache selection; select the matching profile ID')
            if '--r33-cache-kv-memory-bytes'.startswith(option):
                if not separator:
                    value = args[index + 1] if index + 1 < len(args) else ''
                kv_values.append(value)
        if kv_values:
            if (len(kv_values) != 1 or not kv_values[0].isdigit()
                    or int(kv_values[0]) not in (7247757312, 8053063680, 9395240960)
                    or not resolved['serving'].get('sparkcache')):
                raise ValueError('Specify one 6.75, 7.5 or 8.75 GiB KV override for a SparkCache profile')
            kv_bytes = int(kv_values[0])
            modified = kv_bytes != resolved['serving']['kv_cache_memory_bytes']
            resolved['serving']['kv_cache_memory_bytes'] = kv_bytes
            if modified:
                resolved['status'] = 'research-only'
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
        if any(a.startswith('--') and '--runtime-receipt'.startswith(a.split('=', 1)[0]) for a in args):
            raise ValueError('The catalog owns the runtime receipt; select another profile for another release')
        command += ['--runtime-receipt', str(local_path(adapter['receipt'], root))]
    return {'profile': profile_id, 'configuration_status': resolved['status'],
            'serving': resolved['serving'], 'modified_defaults': modified,
            'execution_status': 'not-run', 'guide': p['guide'],
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
