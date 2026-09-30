"""Measures one Spark's available memory once a second while a load command runs, and stops the load below a floor.

If available memory falls below the floor, the load command is killed so its
requests end before the Spark runs out of memory. Prints the lowest reading.

    python guarded.py --node USER@192.0.2.10 --floor 1.2 -- python image_load.py ...
"""
import argparse
import subprocess
import sys
import threading

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--node", required=True, help="SSH target of the Spark to watch (usually Node A)")
parser.add_argument("--floor", type=float, default=1.2, help="GiB of MemAvailable that stops the load (default 1.2)")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
lowest, fired = [99.0], [False]

load = subprocess.Popen(command)
watch = subprocess.Popen(["ssh", args.node, "while true; do awk '/MemAvailable/{print $2}' /proc/meminfo; sleep 1; done"],
                         stdout=subprocess.PIPE, text=True)


def read():
    for line in watch.stdout:
        try:
            gib = int(line) / 1048576
        except ValueError:
            continue
        lowest[0] = min(lowest[0], gib)
        if gib < args.floor and load.poll() is None:
            fired[0] = True
            load.kill()
            print(f"GUARD: {args.node} MemAvailable {gib:.2f} GiB < {args.floor} GiB; load stopped", flush=True)


threading.Thread(target=read, daemon=True).start()
code = load.wait()
watch.kill()
print(f"load exit {code}; lowest MemAvailable on {args.node}: {lowest[0]:.2f} GiB; guard fired: {fired[0]}", flush=True)
sys.exit(code)
