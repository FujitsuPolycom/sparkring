# Image gate comparison: gate-ring8-image-1a8c10354eb0-20261009T185859Z

## Bytes under test

- library `/opt/sparkring/libsircl/lib/libsircl.so.0.6.0`, SHA-256 `8b180879d7c87cd34ad3af78843d5aec7f4328fc265db9c74e95a708200b01b7`, 22802384 bytes
- layer receipt SHA-256 `4da7173bf58f2a38fc8d42dfc085275d60ea2acb8ea6fa7d4e0b8501ab939096`, version 0.6.0, source `dbf3607484dd47df4cf6c8238eb5b3272466effb`, nvcc `Build cuda_13.4.r13.4/compiler.38855100_0`, compiler `gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0`
- sircl_fold: `830df3b6acc306bcb57a50304e72c24854aa1ba7f4637eafdbbfc02f0f86bda4` (1333368 bytes)
- sircl_kernels: `16033e7dc95fdcbf738e76c2520b866f9051f3c0b3135dbcf89ee73c68af2b99` (2713256 bytes)
- sircl_links: `153748c17eed14436749151b8743243f9102fb77eb9f2fb2f09884f95b9e931f` (16729480 bytes)
- sircl_p2p: `f2b842eeefca8ed6ac866c4f7a58729d95827f308b509556259f3314edabd757` (325592 bytes)

Explicit ring-schedule arm (nt-ring-coll) receipts' cycle_plan: False (the plan applies only without a schedule setting).

Baselines not found: e31abc5c (no gate of that snapshot ran on the cycle of eight).

## All-reduce timing sweep (perf_rank.py, bf16, slowest rank's mean eager us)

| Size | image default | image ring | a3477af2 default | a3477af2 ring |
|---|---|---|---|---|
| 4 KiB | 18.2 | 18.5 | 18.2 | 18.3 |
| 8 KiB | 19.9 | 19.9 | 19.9 | 20.0 |
| 16 KiB | 23.3 | 23.6 | 23.5 | 23.2 |
| 32 KiB | 30.6 | 30.7 | 30.7 | 30.7 |
| 64 KiB | 48.7 | 48.8 | 50.8 | 48.4 |
| 128 KiB | 77.2 | 77.1 | 77.1 | 77.0 |
| 256 KiB | 58.4 | 58.5 | 60.3 | 58.5 |
| 512 KiB | 95.6 | 95.6 | 95.2 | 95.4 |
| 1 MiB | 154.8 | 155.1 | 154.4 | 154.5 |
| 2 MiB | 286.8 | 286.9 | 286.4 | 286.3 |
| 4 MiB | 540.9 | 612.7 | 537.0 | 610.6 |
| 8 MiB | 773.8 | 774.5 | 1106.8 | 774.1 |
| 16 MiB | 1306.2 | 1305.0 | 2299.1 | 1304.9 |
| 32 MiB | 2511.9 | 2506.2 | 4601.7 | 2510.2 |
| 64 MiB | 4909.3 | 4910.4 | 9200.9 | 4906.4 |
| 128 MiB | 9698.6 | 9698.7 | 18394.1 | 9698.5 |
| 256 MiB | 19288.4 | 19289.5 | 36794.1 | 19291.4 |

## nccl-tests all_reduce_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | a3477af2 default | a3477af2 ring |
|---|---|---|---|---|
| 1 MiB | 183 | 170 | 165 | 170 |
| 2 MiB | 314 | 309 | 313 | 313 |
| 4 MiB | 606 | 641 | 588 | 666 |
| 8 MiB | 805 | 787 | 1158 | 787 |
| 16 MiB | 1335 | 1322 | 2324 | 1329 |
| 32 MiB | 2542 | 2529 | 4622 | 2523 |
| 64 MiB | 4926 | 4917 | 9223 | 4933 |
| 128 MiB | 9733 | 9726 | 18390 | 9719 |
| 256 MiB | 19314 | 19294 | 36790 | 19305 |

## nccl-tests all_gather_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | a3477af2 default | a3477af2 ring |
|---|---|---|---|---|
| 1 MiB | 96 | 99 | 100 | 91 |
| 2 MiB | 165 | 165 | 164 | 168 |
| 4 MiB | 301 | 308 | 306 | 300 |
| 8 MiB | 436 | 435 | 574 | 431 |
| 16 MiB | 724 | 720 | 1114 | 713 |
| 32 MiB | 1309 | 1313 | 2211 | 1336 |
| 64 MiB | 2534 | 2546 | 4336 | 2544 |
| 128 MiB | 4942 | 4944 | 8656 | 4940 |
| 256 MiB | 9738 | 9736 | 17318 | 9714 |

## nccl-tests reduce_scatter_perf (rank 0's out-of-place us; default = off-coll, ring = ring-coll)

| Size | image default | image ring | a3477af2 default | a3477af2 ring |
|---|---|---|---|---|
| 1 MiB | 114 | 108 | 106 | 109 |
| 2 MiB | 178 | 175 | 194 | 181 |
| 4 MiB | 324 | 378 | 333 | 377 |
| 8 MiB | 432 | 412 | 647 | 444 |
| 16 MiB | 736 | 721 | 1271 | 725 |
| 32 MiB | 1332 | 1320 | 2502 | 1318 |
| 64 MiB | 2536 | 2525 | 4968 | 2528 |
| 128 MiB | 4936 | 4925 | 9920 | 4933 |
| 256 MiB | 9813 | 9835 | 19842 | 9822 |

## Verdicts

- PASS bytes: every rank's library and packs match the lock and the layer receipt: 8 of 8 ranks checked, 8 passed, 1 distinct library SHA-256
- PASS bytes: every communicator receipt names the layer receipt's packs: 272 receipts
- PASS cycle plan in receipts: every eight-rank default communicator has cycle_plan true: 96 communicators
- PASS cycle plan in the bit-exact check (LIBRARY_RANK_EXPECT_CYCLE_PLAN=1): 16 rank runs
- PASS cycle plan in time: from 8 MiB the default is within 1.15x of the ring schedules: 24 size comparisons
