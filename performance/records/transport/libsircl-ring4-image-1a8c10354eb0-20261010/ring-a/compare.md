# Image gate comparison: ring-a

## Bytes under test

- library `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0`, SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`, 22802384 bytes
- layer receipt SHA-256 `4da7173bf58f2a38fc8d42dfc085275d60ea2acb8ea6fa7d4e0b8501ab939096`, version 0.6.0, source `dbf3607484dd47df4cf6c8238eb5b3272466effb`, nvcc `Build cuda_13.4.r13.4/compiler.38855100_0`, compiler `gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0`
- sircl_fold: `830df3b6acc306bcb57a50304e72c24854aa1ba7f4637eafdbbfc02f0f86bda4` (1333368 bytes)
- sircl_kernels: `16033e7dc95fdcbf738e76c2520b866f9051f3c0b3135dbcf89ee73c68af2b99` (2713256 bytes)
- sircl_links: `153748c17eed14436749151b8743243f9102fb77eb9f2fb2f09884f95b9e931f` (16729480 bytes)
- sircl_p2p: `f2b842eeefca8ed6ac866c4f7a58729d95827f308b509556259f3314edabd757` (325592 bytes)

Explicit ring-schedule arm (nt-ring-coll) receipts' cycle_plan: False (the plan applies only without a schedule setting).

## All-reduce timing sweep (perf_rank.py, bf16, slowest rank's mean eager us)

| Size | image default | image ring | ring-b default | ring-b ring |
|---|---|---|---|---|
| 4 KiB | 17.8 | 11.6 | 11.5 | 11.4 |
| 8 KiB | 19.8 | 12.5 | 12.4 | 12.4 |
| 16 KiB | 20.3 | 14.4 | 15.6 | 14.3 |
| 32 KiB | 22.9 | 17.7 | 17.8 | 17.7 |
| 64 KiB | 28.9 | 24.4 | 24.3 | 24.7 |
| 128 KiB | 39.6 | 38.1 | 37.3 | 37.0 |
| 256 KiB | 54.2 | 45.0 | 45.0 | 45.1 |
| 512 KiB | 77.4 | 71.9 | 71.5 | 71.8 |
| 1 MiB | 112.5 | 112.1 | 114.3 | 114.8 |
| 2 MiB | 203.0 | 202.0 | 205.5 | 205.8 |
| 4 MiB | 369.9 | 372.1 | 378.6 | 372.6 |
| 8 MiB | 619.0 | 611.5 | 612.6 | 613.9 |
| 16 MiB | 1144.6 | 1128.6 | 1132.7 | 1135.3 |
| 32 MiB | 2162.4 | 2158.4 | 2167.0 | 2162.3 |
| 64 MiB | 4221.9 | 4215.1 | 4219.7 | 4220.4 |
| 128 MiB | 8327.2 | 8325.0 | 8339.2 | 8330.9 |
| 256 MiB | 16551.4 | 16542.9 | 16561.7 | 16569.3 |

## nccl-tests all_reduce_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-b default | ring-b ring |
|---|---|---|---|---|
| 1 MiB | 123 | 144 | 128 | 129 |
| 2 MiB | 232 | 241 | 226 | 230 |
| 4 MiB | 413 | 403 | 411 | 385 |
| 8 MiB | 630 | 634 | 626 | 628 |
| 16 MiB | 1132 | 1151 | 1139 | 1143 |
| 32 MiB | 2181 | 2198 | 2195 | 2180 |
| 64 MiB | 4252 | 4257 | 4247 | 4242 |
| 128 MiB | 8350 | 8372 | 8353 | 8356 |
| 256 MiB | 16566 | 16590 | 16604 | 16577 |

## nccl-tests all_gather_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-b default | ring-b ring |
|---|---|---|---|---|
| 1 MiB | 76 | 76 | 76 | 73 |
| 2 MiB | 143 | 130 | 127 | 124 |
| 4 MiB | 242 | 226 | 216 | 222 |
| 8 MiB | 362 | 362 | 356 | 355 |
| 16 MiB | 625 | 619 | 618 | 615 |
| 32 MiB | 1148 | 1134 | 1131 | 1131 |
| 64 MiB | 2188 | 2173 | 2176 | 2173 |
| 128 MiB | 4247 | 4227 | 4227 | 4241 |
| 256 MiB | 8321 | 8346 | 8346 | 8344 |

## nccl-tests reduce_scatter_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-b default | ring-b ring |
|---|---|---|---|---|
| 1 MiB | 83 | 84 | 82 | 84 |
| 2 MiB | 137 | 140 | 133 | 131 |
| 4 MiB | 240 | 219 | 235 | 214 |
| 8 MiB | 353 | 356 | 350 | 349 |
| 16 MiB | 617 | 616 | 620 | 624 |
| 32 MiB | 1145 | 1132 | 1147 | 1150 |
| 64 MiB | 2182 | 2184 | 2209 | 2206 |
| 128 MiB | 4340 | 4343 | 4338 | 4325 |
| 256 MiB | 8646 | 8654 | 8676 | 8656 |

## Verdicts

- PASS bytes: every rank's library and packs match the lock and the layer receipt: 4 of 4 ranks checked, 4 passed, 1 distinct library SHA-256
- PASS bytes: every communicator receipt names the layer receipt's packs: 136 receipts
- PASS cycle plan in receipts: every 4-rank default communicator has cycle_plan true: 48 communicators
- PASS cycle plan in the bit-exact check (LIBRARY_RANK_EXPECT_CYCLE_PLAN=1): 8 rank runs
- PASS cycle plan in time: from 8 MiB the default is within 1.15x of the ring schedules: 24 size comparisons
