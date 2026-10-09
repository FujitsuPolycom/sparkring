# Gate report: gate-ring8-image-1a8c10354eb0-20261009T185859Z

### Timing sweep `default`: bfloat16 all-reduce, 8 of 8 ranks, 120 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 18.23 | 17.84 |
| 8 KiB | 19.89 | 19.58 |
| 16 KiB | 23.30 | 23.04 |
| 32 KiB | 30.56 | 30.29 |
| 64 KiB | 48.72 | 48.06 |
| 128 KiB | 77.23 | 77.08 |
| 256 KiB | 58.38 | 57.98 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 154.8 | 11.85 | 154.3 |
| 2 MiB | 286.8 | 12.80 | 285.8 |
| 4 MiB | 540.9 | 13.57 | 538.7 |
| 8 MiB | 773.8 | 18.97 | 773.8 |
| 16 MiB | 1306.2 | 22.48 | 1304.0 |
| 32 MiB | 2511.9 | 23.38 | 2508.7 |
| 64 MiB | 4909.3 | 23.92 | 4900.9 |
| 128 MiB | 9698.6 | 24.22 | 9696.2 |
| 256 MiB | 19288.4 | 24.35 | 19285.4 |

### Timing sweep `ring`: bfloat16 all-reduce, 8 of 8 ranks, 120 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 18.45 | 18.01 |
| 8 KiB | 19.87 | 19.74 |
| 16 KiB | 23.65 | 23.22 |
| 32 KiB | 30.66 | 30.39 |
| 64 KiB | 48.78 | 48.11 |
| 128 KiB | 77.11 | 76.99 |
| 256 KiB | 58.51 | 58.13 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 155.1 | 11.83 | 154.0 |
| 2 MiB | 286.9 | 12.79 | 286.2 |
| 4 MiB | 612.7 | 11.98 | 609.4 |
| 8 MiB | 774.5 | 18.95 | 772.1 |
| 16 MiB | 1305.0 | 22.50 | 1303.6 |
| 32 MiB | 2506.2 | 23.43 | 2504.7 |
| 64 MiB | 4910.4 | 23.92 | 4901.7 |
| 128 MiB | 9698.7 | 24.22 | 9696.4 |
| 256 MiB | 19289.5 | 24.35 | 19287.4 |

### nccl-tests arm `nt-off-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 29.12 us 0.25 GB/s; 8 KiB 26.34 us 0.54 GB/s; 16 KiB 35.81 us 0.80 GB/s; 32 KiB 46.30 us 1.24 GB/s; 64 KiB 59.55 us 1.93 GB/s; 128 KiB 89.33 us 2.57 GB/s; 256 KiB 67.23 us 6.82 GB/s; 1 MiB 183.06 us 10.02 GB/s; 2 MiB 314.18 us 11.68 GB/s; 4 MiB 606.18 us 12.11 GB/s; 8 MiB 804.60 us 18.25 GB/s; 16 MiB 1334.79 us 22.00 GB/s; 32 MiB 2542.02 us 23.10 GB/s; 64 MiB 4925.52 us 23.84 GB/s; 128 MiB 9733.32 us 24.13 GB/s; 256 MiB 19313.8 us 24.32 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 26.67 us 0.13 GB/s; 8 KiB 37.29 us 0.19 GB/s; 16 KiB 35.81 us 0.40 GB/s; 32 KiB 32.90 us 0.87 GB/s; 64 KiB 35.24 us 1.63 GB/s; 128 KiB 42.78 us 2.68 GB/s; 256 KiB 39.51 us 5.80 GB/s; 1 MiB 95.66 us 9.59 GB/s; 2 MiB 164.86 us 11.13 GB/s; 4 MiB 301.15 us 12.19 GB/s; 8 MiB 435.50 us 16.85 GB/s; 16 MiB 723.85 us 20.28 GB/s; 32 MiB 1308.57 us 22.44 GB/s; 64 MiB 2534.50 us 23.17 GB/s; 128 MiB 4941.80 us 23.76 GB/s; 256 MiB 9737.60 us 24.12 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 28.84 us 0.12 GB/s; 8 KiB 33.34 us 0.22 GB/s; 16 KiB 33.42 us 0.43 GB/s; 32 KiB 30.50 us 0.94 GB/s; 64 KiB 32.92 us 1.74 GB/s; 128 KiB 35.68 us 3.21 GB/s; 256 KiB 48.86 us 4.69 GB/s; 1 MiB 113.58 us 8.08 GB/s; 2 MiB 177.71 us 10.33 GB/s; 4 MiB 324.40 us 11.31 GB/s; 8 MiB 431.72 us 17.00 GB/s; 16 MiB 736.02 us 19.95 GB/s; 32 MiB 1331.65 us 22.05 GB/s; 64 MiB 2535.51 us 23.16 GB/s; 128 MiB 4935.58 us 23.79 GB/s; 256 MiB 9813.25 us 23.94 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 34.55 us 0.12 GB/s; 8 KiB 34.63 us 0.24 GB/s; 16 KiB 40.12 us 0.41 GB/s; 32 KiB 47.75 us 0.69 GB/s; 64 KiB 64.04 us 1.02 GB/s; 128 KiB 92.06 us 1.42 GB/s; 256 KiB 156.93 us 1.67 GB/s; 1 MiB 554.42 us 1.89 GB/s; 2 MiB 1094.14 us 1.92 GB/s; 4 MiB 2194.84 us 1.91 GB/s; 8 MiB 4357.63 us 1.93 GB/s; 16 MiB 8688.75 us 1.93 GB/s; 32 MiB 17362.2 us 1.93 GB/s; 64 MiB 34660.1 us 1.94 GB/s; 128 MiB 69315.0 us 1.94 GB/s; 256 MiB 138648 us 1.94 GB/s

### nccl-tests arm `nt-off-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3 3 3 3 3
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3 3 3 3 3
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3 3 3 3 3

### nccl-tests arm `nt-on-budget` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `hypercube_perf -b 8 -e 64K -f 4 -d bfloat16 -c 1 -n 5 -w 1`: exit status per rank 3 3 3 3 3 3 3 3

### nccl-tests arm `nt-on-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 23.19 us 0.31 GB/s; 8 KiB 27.02 us 0.53 GB/s; 16 KiB 30.84 us 0.93 GB/s; 32 KiB 33.59 us 1.71 GB/s; 64 KiB 62.47 us 1.84 GB/s; 128 KiB 95.59 us 2.40 GB/s; 256 KiB 63.96 us 7.17 GB/s; 1 MiB 169.44 us 10.83 GB/s; 2 MiB 309.02 us 11.88 GB/s; 4 MiB 577.84 us 12.70 GB/s; 8 MiB 782.95 us 18.75 GB/s; 16 MiB 1327.09 us 22.12 GB/s; 32 MiB 2510.56 us 23.39 GB/s; 64 MiB 4908.07 us 23.93 GB/s; 128 MiB 9702.50 us 24.21 GB/s; 256 MiB 19301.6 us 24.34 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 24.51 us 0.15 GB/s; 8 KiB 25.31 us 0.28 GB/s; 16 KiB 25.40 us 0.56 GB/s; 32 KiB 24.98 us 1.15 GB/s; 64 KiB 38.72 us 1.48 GB/s; 128 KiB 41.82 us 2.74 GB/s; 256 KiB 43.19 us 5.31 GB/s; 1 MiB 93.75 us 9.79 GB/s; 2 MiB 161.44 us 11.37 GB/s; 4 MiB 299.79 us 12.24 GB/s; 8 MiB 429.03 us 17.11 GB/s; 16 MiB 708.60 us 20.72 GB/s; 32 MiB 1312.44 us 22.37 GB/s; 64 MiB 2521.44 us 23.29 GB/s; 128 MiB 4920.06 us 23.87 GB/s; 256 MiB 9716.01 us 24.17 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 29.39 us 0.12 GB/s; 8 KiB 20.88 us 0.34 GB/s; 16 KiB 25.89 us 0.55 GB/s; 32 KiB 32.75 us 0.88 GB/s; 64 KiB 27.40 us 2.09 GB/s; 128 KiB 29.29 us 3.92 GB/s; 256 KiB 41.22 us 5.57 GB/s; 1 MiB 109.18 us 8.40 GB/s; 2 MiB 182.83 us 10.04 GB/s; 4 MiB 323.79 us 11.33 GB/s; 8 MiB 416.97 us 17.60 GB/s; 16 MiB 708.61 us 20.72 GB/s; 32 MiB 1316.74 us 22.30 GB/s; 64 MiB 2526.38 us 23.24 GB/s; 128 MiB 4909.74 us 23.92 GB/s; 256 MiB 9819.39 us 23.92 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 24.51 us 0.17 GB/s; 8 KiB 33.20 us 0.25 GB/s; 16 KiB 39.46 us 0.42 GB/s; 32 KiB 45.71 us 0.72 GB/s; 64 KiB 58.58 us 1.12 GB/s; 128 KiB 89.60 us 1.46 GB/s; 256 KiB 150.37 us 1.74 GB/s; 1 MiB 542.69 us 1.93 GB/s; 2 MiB 1103.75 us 1.90 GB/s; 4 MiB 2184.14 us 1.92 GB/s; 8 MiB 4354.25 us 1.93 GB/s; 16 MiB 8703.24 us 1.93 GB/s; 32 MiB 17370.7 us 1.93 GB/s; 64 MiB 34770.2 us 1.93 GB/s; 128 MiB 69490.3 us 1.93 GB/s; 256 MiB 139020 us 1.93 GB/s

### nccl-tests arm `nt-on-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 137.04 us 0.03 GB/s; 8 KiB 141.52 us 0.05 GB/s; 16 KiB 136.27 us 0.11 GB/s; 32 KiB 133.35 us 0.22 GB/s; 64 KiB 129.10 us 0.44 GB/s; 128 KiB 110.28 us 1.04 GB/s; 256 KiB 95.97 us 2.39 GB/s; 1 MiB 141.34 us 6.49 GB/s; 2 MiB 221.12 us 8.30 GB/s; 4 MiB 357.81 us 10.26 GB/s; 8 MiB 672.73 us 10.91 GB/s; 16 MiB 1279.86 us 11.47 GB/s; 32 MiB 2259.17 us 13.00 GB/s; 64 MiB 4229.48 us 13.88 GB/s; 128 MiB 8351.24 us 14.06 GB/s; 256 MiB 16953.1 us 13.85 GB/s
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 47.35 us 0.09 GB/s; 8 KiB 38.05 us 0.22 GB/s; 16 KiB 34.77 us 0.47 GB/s; 32 KiB 26.53 us 1.24 GB/s; 64 KiB 39.49 us 1.66 GB/s; 128 KiB 33.55 us 3.91 GB/s; 256 KiB 42.52 us 6.17 GB/s; 1 MiB 99.15 us 10.58 GB/s; 2 MiB 173.51 us 12.09 GB/s; 4 MiB 257.30 us 16.30 GB/s; 8 MiB 430.26 us 19.50 GB/s; 16 MiB 783.20 us 21.42 GB/s; 32 MiB 1473.62 us 22.77 GB/s; 64 MiB 2842.45 us 23.61 GB/s; 128 MiB 5594.46 us 23.99 GB/s; 256 MiB 11068.5 us 24.25 GB/s
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked

### nccl-tests arm `nt-ring-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 29.79 us 0.24 GB/s; 8 KiB 27.38 us 0.52 GB/s; 16 KiB 34.23 us 0.84 GB/s; 32 KiB 38.46 us 1.49 GB/s; 64 KiB 59.47 us 1.93 GB/s; 128 KiB 79.91 us 2.87 GB/s; 256 KiB 70.08 us 6.55 GB/s; 1 MiB 169.60 us 10.82 GB/s; 2 MiB 309.28 us 11.87 GB/s; 4 MiB 640.89 us 11.45 GB/s; 8 MiB 787.12 us 18.65 GB/s; 16 MiB 1322.04 us 22.21 GB/s; 32 MiB 2529.03 us 23.22 GB/s; 64 MiB 4916.72 us 23.89 GB/s; 128 MiB 9726.29 us 24.15 GB/s; 256 MiB 19294.2 us 24.35 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 34.55 us 0.10 GB/s; 8 KiB 41.27 us 0.17 GB/s; 16 KiB 31.94 us 0.45 GB/s; 32 KiB 30.09 us 0.95 GB/s; 64 KiB 33.50 us 1.71 GB/s; 128 KiB 41.68 us 2.75 GB/s; 256 KiB 51.55 us 4.45 GB/s; 1 MiB 99.08 us 9.26 GB/s; 2 MiB 165.47 us 11.09 GB/s; 4 MiB 308.29 us 11.90 GB/s; 8 MiB 435.02 us 16.87 GB/s; 16 MiB 720.42 us 20.38 GB/s; 32 MiB 1313.20 us 22.36 GB/s; 64 MiB 2545.96 us 23.06 GB/s; 128 MiB 4944.40 us 23.75 GB/s; 256 MiB 9736.28 us 24.12 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0 0 0 0 0, every value checked
  4 KiB 26.82 us 0.13 GB/s; 8 KiB 24.04 us 0.30 GB/s; 16 KiB 26.35 us 0.54 GB/s; 32 KiB 33.41 us 0.86 GB/s; 64 KiB 27.78 us 2.06 GB/s; 128 KiB 34.93 us 3.28 GB/s; 256 KiB 45.83 us 5.01 GB/s; 1 MiB 108.50 us 8.46 GB/s; 2 MiB 174.98 us 10.49 GB/s; 4 MiB 377.89 us 9.71 GB/s; 8 MiB 412.00 us 17.82 GB/s; 16 MiB 720.51 us 20.37 GB/s; 32 MiB 1319.63 us 22.25 GB/s; 64 MiB 2524.73 us 23.26 GB/s; 128 MiB 4924.62 us 23.85 GB/s; 256 MiB 9834.79 us 23.88 GB/s

