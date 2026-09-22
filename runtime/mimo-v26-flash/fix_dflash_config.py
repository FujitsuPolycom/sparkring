"""Write a valid-JSON copy of a DFlash draft configuration.

MiMo-V2.6-Flash-RL revisions before b2674c72176d008cb675a09f626f8eba3f3f21c4
ship ``dflash/config.json`` with a trailing comma, which ``json.load`` rejects.
The launchers mount the copy this script writes over the checkpoint's file
only when the original does not parse. Key order and values are unchanged.
"""
import json
import re
import sys


def corrected(text: str) -> dict:
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def main(argv):
    if len(argv) != 3:
        raise SystemExit("usage: fix_dflash_config.py SOURCE DESTINATION")
    with open(argv[1], encoding="utf-8") as source:
        data = corrected(source.read())
    with open(argv[2], "w", encoding="utf-8") as destination:
        json.dump(data, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main(sys.argv)
