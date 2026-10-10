# Image gate comparison: ring-b

## Bytes under test

- library `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0`, SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`, 22802384 bytes
- layer receipt SHA-256 `4da7173bf58f2a38fc8d42dfc085275d60ea2acb8ea6fa7d4e0b8501ab939096`, version 0.6.0, source `dbf3607484dd47df4cf6c8238eb5b3272466effb`, nvcc `Build cuda_13.4.r13.4/compiler.38855100_0`, compiler `gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0`
- sircl_fold: `830df3b6acc306bcb57a50304e72c24854aa1ba7f4637eafdbbfc02f0f86bda4` (1333368 bytes)
- sircl_kernels: `16033e7dc95fdcbf738e76c2520b866f9051f3c0b3135dbcf89ee73c68af2b99` (2713256 bytes)
- sircl_links: `153748c17eed14436749151b8743243f9102fb77eb9f2fb2f09884f95b9e931f` (16729480 bytes)
- sircl_p2p: `f2b842eeefca8ed6ac866c4f7a58729d95827f308b509556259f3314edabd757` (325592 bytes)

Explicit ring-schedule arm (nt-ring-coll) receipts' cycle_plan: False (the plan applies only without a schedule setting).

## All-reduce timing sweep (perf_rank.py, bf16, slowest rank's mean eager us)

| Size | image default | image ring | ring-a default | ring-a ring |
|---|---|---|---|---|
| 4 KiB | 11.5 | 11.4 | 17.8 | 11.6 |
| 8 KiB | 12.4 | 12.4 | 19.8 | 12.5 |
| 16 KiB | 15.6 | 14.3 | 20.3 | 14.4 |
| 32 KiB | 17.8 | 17.7 | 22.9 | 17.7 |
| 64 KiB | 24.3 | 24.7 | 28.9 | 24.4 |
| 128 KiB | 37.3 | 37.0 | 39.6 | 38.1 |
| 256 KiB | 45.0 | 45.1 | 54.2 | 45.0 |
| 512 KiB | 71.5 | 71.8 | 77.4 | 71.9 |
| 1 MiB | 114.3 | 114.8 | 112.5 | 112.1 |
| 2 MiB | 205.5 | 205.8 | 203.0 | 202.0 |
| 4 MiB | 378.6 | 372.6 | 369.9 | 372.1 |
| 8 MiB | 612.6 | 613.9 | 619.0 | 611.5 |
| 16 MiB | 1132.7 | 1135.3 | 1144.6 | 1128.6 |
| 32 MiB | 2167.0 | 2162.3 | 2162.4 | 2158.4 |
| 64 MiB | 4219.7 | 4220.4 | 4221.9 | 4215.1 |
| 128 MiB | 8339.2 | 8330.9 | 8327.2 | 8325.0 |
| 256 MiB | 16561.7 | 16569.3 | 16551.4 | 16542.9 |

## nccl-tests all_reduce_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-a default | ring-a ring |
|---|---|---|---|---|
| 1 MiB | 128 | 129 | 123 | 144 |
| 2 MiB | 226 | 230 | 232 | 241 |
| 4 MiB | 411 | 385 | 413 | 403 |
| 8 MiB | 626 | 628 | 630 | 634 |
| 16 MiB | 1139 | 1143 | 1132 | 1151 |
| 32 MiB | 2195 | 2180 | 2181 | 2198 |
| 64 MiB | 4247 | 4242 | 4252 | 4257 |
| 128 MiB | 8353 | 8356 | 8350 | 8372 |
| 256 MiB | 16604 | 16577 | 16566 | 16590 |

## nccl-tests all_gather_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-a default | ring-a ring |
|---|---|---|---|---|
| 1 MiB | 76 | 73 | 76 | 76 |
| 2 MiB | 127 | 124 | 143 | 130 |
| 4 MiB | 216 | 222 | 242 | 226 |
| 8 MiB | 356 | 355 | 362 | 362 |
| 16 MiB | 618 | 615 | 625 | 619 |
| 32 MiB | 1131 | 1131 | 1148 | 1134 |
| 64 MiB | 2176 | 2173 | 2188 | 2173 |
| 128 MiB | 4227 | 4241 | 4247 | 4227 |
| 256 MiB | 8346 | 8344 | 8321 | 8346 |

## nccl-tests reduce_scatter_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | ring-a default | ring-a ring |
|---|---|---|---|---|
| 1 MiB | 82 | 84 | 83 | 84 |
| 2 MiB | 133 | 131 | 137 | 140 |
| 4 MiB | 235 | 214 | 240 | 219 |
| 8 MiB | 350 | 349 | 353 | 356 |
| 16 MiB | 620 | 624 | 617 | 616 |
| 32 MiB | 1147 | 1150 | 1145 | 1132 |
| 64 MiB | 2209 | 2206 | 2182 | 2184 |
| 128 MiB | 4338 | 4325 | 4340 | 4343 |
| 256 MiB | 8676 | 8656 | 8646 | 8654 |

## Verdicts

- PASS bytes: every rank's library and packs match the lock and the layer receipt: 4 of 4 ranks checked, 4 passed, 1 distinct library SHA-256
- PASS bytes: every communicator receipt names the layer receipt's packs: 136 receipts
- PASS cycle plan in receipts: every 4-rank default communicator has cycle_plan true: 48 communicators
- PASS cycle plan in the bit-exact check (LIBRARY_RANK_EXPECT_CYCLE_PLAN=1): 8 rank runs
- PASS cycle plan in time: from 8 MiB the default is within 1.15x of the ring schedules: 24 size comparisons
