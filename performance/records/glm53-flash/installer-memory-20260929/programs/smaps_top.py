"""Measures the largest memory mappings of one process: resident size and each PSS kind, by mapping name."""
import collections
import re
import sys

pid = sys.argv[1]
totals = collections.defaultdict(lambda: collections.Counter())
name = None
for line in open(f"/proc/{pid}/smaps"):
    head = re.match(r"^[0-9a-f]+-[0-9a-f]+ \S+ \S+ \S+ \S+\s*(.*)$", line)
    if head:
        name = head.group(1).strip() or "[anon]"
        continue
    key, _, rest = line.partition(":")
    if key in ("Rss", "Pss_Anon", "Pss_File", "Pss_Shmem"):
        totals[name][key] += int(rest.split()[0])
rows = sorted(totals.items(), key=lambda item: -item[1]["Rss"])
for mapping, counts in rows[:14]:
    print(f"{counts['Rss'] / 1048576:7.2f} GiB rss  anon {counts['Pss_Anon'] / 1048576:5.2f}  "
          f"shmem {counts['Pss_Shmem'] / 1048576:5.2f}  file {counts['Pss_File'] / 1048576:5.2f}  {mapping[:90]}")
print("SysV shm segments (bytes, key):")
for line in open("/proc/sysvipc/shm").readlines()[1:]:
    fields = line.split()
    if int(fields[3]) > 50_000_000:
        print("  ", fields[3], fields[0])
