# Gate report: ring-b

### Timing sweep `default`: bfloat16 all-reduce, 4 of 4 ranks, 60 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 11.55 | 11.24 |
| 8 KiB | 12.37 | 12.69 |
| 16 KiB | 15.63 | 14.20 |
| 32 KiB | 17.85 | 17.66 |
| 64 KiB | 24.30 | 23.74 |
| 128 KiB | 37.32 | 36.75 |
| 256 KiB | 45.02 | 44.88 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 114.3 | 13.76 | 113.8 |
| 2 MiB | 205.5 | 15.31 | 206.4 |
| 4 MiB | 378.6 | 16.62 | 377.6 |
| 8 MiB | 612.6 | 20.54 | 610.3 |
| 16 MiB | 1132.7 | 22.22 | 1129.1 |
| 32 MiB | 2167.0 | 23.23 | 2162.3 |
| 64 MiB | 4219.7 | 23.86 | 4221.9 |
| 128 MiB | 8339.2 | 24.14 | 8337.8 |
| 256 MiB | 16561.7 | 24.31 | 16559.0 |

### Timing sweep `ring`: bfloat16 all-reduce, 4 of 4 ranks, 60 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 11.38 | 11.37 |
| 8 KiB | 12.35 | 12.17 |
| 16 KiB | 14.34 | 14.20 |
| 32 KiB | 17.75 | 17.38 |
| 64 KiB | 24.69 | 23.83 |
| 128 KiB | 36.96 | 36.80 |
| 256 KiB | 45.06 | 44.64 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 114.8 | 13.70 | 113.8 |
| 2 MiB | 205.8 | 15.28 | 204.5 |
| 4 MiB | 372.6 | 16.88 | 371.7 |
| 8 MiB | 613.9 | 20.50 | 620.6 |
| 16 MiB | 1135.3 | 22.17 | 1132.9 |
| 32 MiB | 2162.3 | 23.28 | 2161.2 |
| 64 MiB | 4220.4 | 23.85 | 4217.4 |
| 128 MiB | 8330.9 | 24.17 | 8330.7 |
| 256 MiB | 16569.3 | 24.30 | 16551.4 |

### nccl-tests arm `nt-off-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 18.49 us 0.33 GB/s; 8 KiB 21.13 us 0.58 GB/s; 16 KiB 19.47 us 1.26 GB/s; 32 KiB 25.45 us 1.93 GB/s; 64 KiB 27.98 us 3.51 GB/s; 128 KiB 47.92 us 4.10 GB/s; 256 KiB 53.56 us 7.34 GB/s; 1 MiB 127.85 us 12.30 GB/s; 2 MiB 226.10 us 13.91 GB/s; 4 MiB 410.64 us 15.32 GB/s; 8 MiB 625.87 us 20.10 GB/s; 16 MiB 1139.08 us 22.09 GB/s; 32 MiB 2194.97 us 22.93 GB/s; 64 MiB 4247.06 us 23.70 GB/s; 128 MiB 8352.68 us 24.10 GB/s; 256 MiB 16604.2 us 24.25 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 20.18 us 0.15 GB/s; 8 KiB 23.47 us 0.26 GB/s; 16 KiB 22.24 us 0.55 GB/s; 32 KiB 17.11 us 1.44 GB/s; 64 KiB 21.17 us 2.32 GB/s; 128 KiB 28.51 us 3.45 GB/s; 256 KiB 31.80 us 6.18 GB/s; 1 MiB 75.83 us 10.37 GB/s; 2 MiB 127.03 us 12.38 GB/s; 4 MiB 216.45 us 14.53 GB/s; 8 MiB 356.20 us 17.66 GB/s; 16 MiB 617.99 us 20.36 GB/s; 32 MiB 1131.23 us 22.25 GB/s; 64 MiB 2175.97 us 23.13 GB/s; 128 MiB 4226.75 us 23.82 GB/s; 256 MiB 8345.64 us 24.12 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 12.95 us 0.24 GB/s; 8 KiB 19.24 us 0.32 GB/s; 16 KiB 22.66 us 0.54 GB/s; 32 KiB 19.55 us 1.26 GB/s; 64 KiB 24.21 us 2.03 GB/s; 128 KiB 27.81 us 3.53 GB/s; 256 KiB 40.84 us 4.81 GB/s; 1 MiB 81.72 us 9.62 GB/s; 2 MiB 133.41 us 11.79 GB/s; 4 MiB 234.73 us 13.40 GB/s; 8 MiB 349.90 us 17.98 GB/s; 16 MiB 620.23 us 20.29 GB/s; 32 MiB 1146.62 us 21.95 GB/s; 64 MiB 2208.81 us 22.79 GB/s; 128 MiB 4338.10 us 23.20 GB/s; 256 MiB 8675.51 us 23.21 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 25.07 us 0.16 GB/s; 8 KiB 25.73 us 0.32 GB/s; 16 KiB 20.16 us 0.81 GB/s; 32 KiB 32.67 us 1.00 GB/s; 64 KiB 36.59 us 1.79 GB/s; 128 KiB 48.29 us 2.71 GB/s; 256 KiB 68.14 us 3.85 GB/s; 1 MiB 205.20 us 5.11 GB/s; 2 MiB 403.52 us 5.20 GB/s; 4 MiB 796.04 us 5.27 GB/s; 8 MiB 1576.90 us 5.32 GB/s; 16 MiB 3158.60 us 5.31 GB/s; 32 MiB 6276.11 us 5.35 GB/s; 64 MiB 12504.9 us 5.37 GB/s; 128 MiB 25000.8 us 5.37 GB/s; 256 MiB 50055.4 us 5.36 GB/s

### nccl-tests arm `nt-off-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3

### nccl-tests arm `nt-on-budget` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `hypercube_perf -b 8 -e 64K -f 4 -d bfloat16 -c 1 -n 5 -w 1`: exit status per rank 0 0 0 0, every value checked

### nccl-tests arm `nt-on-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 19.90 us 0.31 GB/s; 8 KiB 22.40 us 0.55 GB/s; 16 KiB 18.64 us 1.32 GB/s; 32 KiB 27.98 us 1.76 GB/s; 64 KiB 30.29 us 3.25 GB/s; 128 KiB 46.19 us 4.26 GB/s; 256 KiB 49.75 us 7.90 GB/s; 1 MiB 134.08 us 11.73 GB/s; 2 MiB 224.95 us 13.98 GB/s; 4 MiB 415.23 us 15.15 GB/s; 8 MiB 629.67 us 19.98 GB/s; 16 MiB 1142.36 us 22.03 GB/s; 32 MiB 2170.81 us 23.19 GB/s; 64 MiB 4233.97 us 23.78 GB/s; 128 MiB 8375.07 us 24.04 GB/s; 256 MiB 16579.1 us 24.29 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 17.81 us 0.17 GB/s; 8 KiB 19.00 us 0.32 GB/s; 16 KiB 19.22 us 0.64 GB/s; 32 KiB 22.23 us 1.11 GB/s; 64 KiB 21.43 us 2.29 GB/s; 128 KiB 22.36 us 4.40 GB/s; 256 KiB 32.70 us 6.01 GB/s; 1 MiB 73.00 us 10.77 GB/s; 2 MiB 121.87 us 12.91 GB/s; 4 MiB 218.81 us 14.38 GB/s; 8 MiB 352.97 us 17.82 GB/s; 16 MiB 618.33 us 20.35 GB/s; 32 MiB 1132.90 us 22.21 GB/s; 64 MiB 2159.61 us 23.31 GB/s; 128 MiB 4216.21 us 23.88 GB/s; 256 MiB 8323.51 us 24.19 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 22.37 us 0.14 GB/s; 8 KiB 13.58 us 0.45 GB/s; 16 KiB 13.92 us 0.88 GB/s; 32 KiB 15.64 us 1.57 GB/s; 64 KiB 21.06 us 2.33 GB/s; 128 KiB 27.21 us 3.61 GB/s; 256 KiB 35.44 us 5.55 GB/s; 1 MiB 86.45 us 9.10 GB/s; 2 MiB 139.92 us 11.24 GB/s; 4 MiB 235.21 us 13.37 GB/s; 8 MiB 355.84 us 17.68 GB/s; 16 MiB 619.17 us 20.32 GB/s; 32 MiB 1134.54 us 22.18 GB/s; 64 MiB 2184.76 us 23.04 GB/s; 128 MiB 4347.48 us 23.15 GB/s; 256 MiB 8647.63 us 23.28 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 23.43 us 0.17 GB/s; 8 KiB 20.64 us 0.40 GB/s; 16 KiB 23.67 us 0.69 GB/s; 32 KiB 26.13 us 1.25 GB/s; 64 KiB 30.22 us 2.17 GB/s; 128 KiB 48.74 us 2.69 GB/s; 256 KiB 67.01 us 3.91 GB/s; 1 MiB 197.44 us 5.31 GB/s; 2 MiB 394.33 us 5.32 GB/s; 4 MiB 784.25 us 5.35 GB/s; 8 MiB 1565.47 us 5.36 GB/s; 16 MiB 3129.59 us 5.36 GB/s; 32 MiB 6211.73 us 5.40 GB/s; 64 MiB 12422.9 us 5.40 GB/s; 128 MiB 24864.6 us 5.40 GB/s; 256 MiB 49675.0 us 5.40 GB/s

### nccl-tests arm `nt-on-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 80.44 us 0.04 GB/s; 8 KiB 79.97 us 0.08 GB/s; 16 KiB 77.09 us 0.16 GB/s; 32 KiB 73.89 us 0.33 GB/s; 64 KiB 63.05 us 0.78 GB/s; 128 KiB 57.47 us 1.71 GB/s; 256 KiB 62.51 us 3.15 GB/s; 1 MiB 104.68 us 7.51 GB/s; 2 MiB 163.48 us 9.62 GB/s; 4 MiB 274.65 us 11.45 GB/s; 8 MiB 508.20 us 12.38 GB/s; 16 MiB 830.85 us 15.14 GB/s; 32 MiB 1470.27 us 17.12 GB/s; 64 MiB 2707.41 us 18.59 GB/s; 128 MiB 5209.83 us 19.32 GB/s; 256 MiB 10239.9 us 19.66 GB/s
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 39.82 us 0.10 GB/s; 8 KiB 41.52 us 0.20 GB/s; 16 KiB 39.24 us 0.42 GB/s; 32 KiB 26.55 us 1.23 GB/s; 64 KiB 32.64 us 2.01 GB/s; 128 KiB 35.70 us 3.67 GB/s; 256 KiB 49.54 us 5.29 GB/s; 1 MiB 99.71 us 10.52 GB/s; 2 MiB 179.73 us 11.67 GB/s; 4 MiB 275.95 us 15.20 GB/s; 8 MiB 449.59 us 18.66 GB/s; 16 MiB 803.82 us 20.87 GB/s; 32 MiB 1475.16 us 22.75 GB/s; 64 MiB 2854.34 us 23.51 GB/s; 128 MiB 5583.32 us 24.04 GB/s; 256 MiB 11078.0 us 24.23 GB/s
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked

### nccl-tests arm `nt-ring-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 19.99 us 0.31 GB/s; 8 KiB 20.21 us 0.61 GB/s; 16 KiB 18.47 us 1.33 GB/s; 32 KiB 24.13 us 2.04 GB/s; 64 KiB 26.97 us 3.64 GB/s; 128 KiB 44.94 us 4.37 GB/s; 256 KiB 51.62 us 7.62 GB/s; 1 MiB 129.32 us 12.16 GB/s; 2 MiB 229.80 us 13.69 GB/s; 4 MiB 384.97 us 16.34 GB/s; 8 MiB 627.71 us 20.05 GB/s; 16 MiB 1143.42 us 22.01 GB/s; 32 MiB 2179.70 us 23.09 GB/s; 64 MiB 4241.61 us 23.73 GB/s; 128 MiB 8355.84 us 24.09 GB/s; 256 MiB 16576.6 us 24.29 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 24.45 us 0.13 GB/s; 8 KiB 27.58 us 0.22 GB/s; 16 KiB 25.51 us 0.48 GB/s; 32 KiB 21.66 us 1.13 GB/s; 64 KiB 25.04 us 1.96 GB/s; 128 KiB 27.99 us 3.51 GB/s; 256 KiB 30.01 us 6.55 GB/s; 1 MiB 73.06 us 10.76 GB/s; 2 MiB 124.19 us 12.67 GB/s; 4 MiB 222.17 us 14.16 GB/s; 8 MiB 354.63 us 17.74 GB/s; 16 MiB 614.88 us 20.46 GB/s; 32 MiB 1131.04 us 22.25 GB/s; 64 MiB 2173.23 us 23.16 GB/s; 128 MiB 4240.84 us 23.74 GB/s; 256 MiB 8344.35 us 24.13 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 16.40 us 0.19 GB/s; 8 KiB 21.30 us 0.29 GB/s; 16 KiB 18.64 us 0.66 GB/s; 32 KiB 15.19 us 1.62 GB/s; 64 KiB 24.40 us 2.01 GB/s; 128 KiB 32.11 us 3.06 GB/s; 256 KiB 45.67 us 4.30 GB/s; 1 MiB 84.28 us 9.33 GB/s; 2 MiB 131.07 us 12.00 GB/s; 4 MiB 214.18 us 14.69 GB/s; 8 MiB 349.11 us 18.02 GB/s; 16 MiB 624.39 us 20.15 GB/s; 32 MiB 1150.45 us 21.87 GB/s; 64 MiB 2206.45 us 22.81 GB/s; 128 MiB 4324.51 us 23.28 GB/s; 256 MiB 8655.87 us 23.26 GB/s

