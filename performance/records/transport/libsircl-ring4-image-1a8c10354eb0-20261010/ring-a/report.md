# Gate report: ring-a

### Timing sweep `default`: bfloat16 all-reduce, 4 of 4 ranks, 60 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 17.76 | 16.38 |
| 8 KiB | 19.75 | 19.32 |
| 16 KiB | 20.28 | 21.23 |
| 32 KiB | 22.92 | 22.64 |
| 64 KiB | 28.93 | 28.65 |
| 128 KiB | 39.60 | 38.47 |
| 256 KiB | 54.20 | 51.35 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 112.5 | 13.98 | 112.6 |
| 2 MiB | 203.0 | 15.49 | 202.3 |
| 4 MiB | 369.9 | 17.01 | 370.1 |
| 8 MiB | 619.0 | 20.33 | 614.5 |
| 16 MiB | 1144.6 | 21.99 | 1141.8 |
| 32 MiB | 2162.4 | 23.28 | 2174.4 |
| 64 MiB | 4221.9 | 23.84 | 4214.6 |
| 128 MiB | 8327.2 | 24.18 | 8324.2 |
| 256 MiB | 16551.4 | 24.33 | 16554.5 |

### Timing sweep `ring`: bfloat16 all-reduce, 4 of 4 ranks, 60 size checks, 0 wrong

Latency (slowest rank's mean time per call):

| Size | eager us | graph us |
|---|---|---|
| 4 KiB | 11.61 | 11.43 |
| 8 KiB | 12.48 | 12.31 |
| 16 KiB | 14.36 | 14.42 |
| 32 KiB | 17.74 | 17.41 |
| 64 KiB | 24.35 | 23.74 |
| 128 KiB | 38.12 | 36.54 |
| 256 KiB | 45.03 | 44.46 |

Bandwidth (bus bandwidth at the slowest rank's mean eager time):

| Size | eager us | bus GB/s | graph us |
|---|---|---|---|
| 1 MiB | 112.1 | 14.03 | 116.4 |
| 2 MiB | 202.0 | 15.58 | 202.4 |
| 4 MiB | 372.1 | 16.91 | 369.7 |
| 8 MiB | 611.5 | 20.58 | 609.3 |
| 16 MiB | 1128.6 | 22.30 | 1127.1 |
| 32 MiB | 2158.4 | 23.32 | 2163.7 |
| 64 MiB | 4215.1 | 23.88 | 4213.6 |
| 128 MiB | 8325.0 | 24.18 | 8321.6 |
| 256 MiB | 16542.9 | 24.34 | 16541.3 |

### nccl-tests arm `nt-off-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 25.60 us 0.24 GB/s; 8 KiB 24.02 us 0.51 GB/s; 16 KiB 29.57 us 0.83 GB/s; 32 KiB 31.57 us 1.56 GB/s; 64 KiB 29.92 us 3.29 GB/s; 128 KiB 46.54 us 4.22 GB/s; 256 KiB 57.37 us 6.85 GB/s; 1 MiB 123.08 us 12.78 GB/s; 2 MiB 232.03 us 13.56 GB/s; 4 MiB 412.96 us 15.24 GB/s; 8 MiB 630.20 us 19.97 GB/s; 16 MiB 1132.07 us 22.23 GB/s; 32 MiB 2181.21 us 23.08 GB/s; 64 MiB 4252.18 us 23.67 GB/s; 128 MiB 8349.59 us 24.11 GB/s; 256 MiB 16566.1 us 24.31 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 22.18 us 0.14 GB/s; 8 KiB 25.59 us 0.24 GB/s; 16 KiB 21.60 us 0.57 GB/s; 32 KiB 21.93 us 1.12 GB/s; 64 KiB 29.00 us 1.69 GB/s; 128 KiB 28.04 us 3.51 GB/s; 256 KiB 39.48 us 4.98 GB/s; 1 MiB 75.96 us 10.35 GB/s; 2 MiB 142.99 us 11.00 GB/s; 4 MiB 242.18 us 12.99 GB/s; 8 MiB 362.01 us 17.38 GB/s; 16 MiB 625.20 us 20.13 GB/s; 32 MiB 1147.95 us 21.92 GB/s; 64 MiB 2187.95 us 23.00 GB/s; 128 MiB 4246.86 us 23.70 GB/s; 256 MiB 8320.76 us 24.20 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 24.55 us 0.13 GB/s; 8 KiB 14.80 us 0.42 GB/s; 16 KiB 24.88 us 0.49 GB/s; 32 KiB 26.62 us 0.92 GB/s; 64 KiB 28.31 us 1.74 GB/s; 128 KiB 27.08 us 3.63 GB/s; 256 KiB 38.04 us 5.17 GB/s; 1 MiB 83.25 us 9.45 GB/s; 2 MiB 136.57 us 11.52 GB/s; 4 MiB 240.32 us 13.09 GB/s; 8 MiB 353.39 us 17.80 GB/s; 16 MiB 617.09 us 20.39 GB/s; 32 MiB 1145.21 us 21.97 GB/s; 64 MiB 2182.28 us 23.06 GB/s; 128 MiB 4339.86 us 23.20 GB/s; 256 MiB 8645.51 us 23.29 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 22.24 us 0.18 GB/s; 8 KiB 25.46 us 0.32 GB/s; 16 KiB 24.20 us 0.68 GB/s; 32 KiB 31.30 us 1.05 GB/s; 64 KiB 34.41 us 1.90 GB/s; 128 KiB 48.07 us 2.73 GB/s; 256 KiB 63.88 us 4.10 GB/s; 1 MiB 200.80 us 5.22 GB/s; 2 MiB 399.62 us 5.25 GB/s; 4 MiB 799.36 us 5.25 GB/s; 8 MiB 1608.28 us 5.22 GB/s; 16 MiB 3194.54 us 5.25 GB/s; 32 MiB 6384.44 us 5.26 GB/s; 64 MiB 12731.5 us 5.27 GB/s; 128 MiB 25405.8 us 5.28 GB/s; 256 MiB 50804.7 us 5.28 GB/s

### nccl-tests arm `nt-off-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 3 3 3 3

### nccl-tests arm `nt-on-budget` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `hypercube_perf -b 8 -e 64K -f 4 -d bfloat16 -c 1 -n 5 -w 1`: exit status per rank 0 0 0 0, every value checked

### nccl-tests arm `nt-on-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 19.95 us 0.31 GB/s; 8 KiB 17.25 us 0.71 GB/s; 16 KiB 19.76 us 1.24 GB/s; 32 KiB 23.78 us 2.07 GB/s; 64 KiB 28.84 us 3.41 GB/s; 128 KiB 46.48 us 4.23 GB/s; 256 KiB 51.55 us 7.63 GB/s; 1 MiB 127.67 us 12.32 GB/s; 2 MiB 227.81 us 13.81 GB/s; 4 MiB 415.79 us 15.13 GB/s; 8 MiB 621.96 us 20.23 GB/s; 16 MiB 1141.97 us 22.04 GB/s; 32 MiB 2166.13 us 23.24 GB/s; 64 MiB 4231.19 us 23.79 GB/s; 128 MiB 8350.46 us 24.11 GB/s; 256 MiB 16555.7 us 24.32 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 20.66 us 0.15 GB/s; 8 KiB 18.50 us 0.33 GB/s; 16 KiB 19.38 us 0.63 GB/s; 32 KiB 24.25 us 1.01 GB/s; 64 KiB 23.62 us 2.08 GB/s; 128 KiB 28.39 us 3.46 GB/s; 256 KiB 38.37 us 5.12 GB/s; 1 MiB 75.95 us 10.35 GB/s; 2 MiB 126.74 us 12.41 GB/s; 4 MiB 220.15 us 14.29 GB/s; 8 MiB 358.98 us 17.53 GB/s; 16 MiB 623.26 us 20.19 GB/s; 32 MiB 1135.61 us 22.16 GB/s; 64 MiB 2167.34 us 23.22 GB/s; 128 MiB 4232.04 us 23.79 GB/s; 256 MiB 8334.84 us 24.15 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 26.88 us 0.11 GB/s; 8 KiB 27.33 us 0.22 GB/s; 16 KiB 27.68 us 0.44 GB/s; 32 KiB 19.17 us 1.28 GB/s; 64 KiB 21.53 us 2.28 GB/s; 128 KiB 25.52 us 3.85 GB/s; 256 KiB 36.63 us 5.37 GB/s; 1 MiB 88.05 us 8.93 GB/s; 2 MiB 133.74 us 11.76 GB/s; 4 MiB 238.67 us 13.18 GB/s; 8 MiB 363.79 us 17.29 GB/s; 16 MiB 612.54 us 20.54 GB/s; 32 MiB 1140.43 us 22.07 GB/s; 64 MiB 2184.94 us 23.04 GB/s; 128 MiB 4334.34 us 23.22 GB/s; 256 MiB 8641.24 us 23.30 GB/s
- `broadcast_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 25.03 us 0.16 GB/s; 8 KiB 24.55 us 0.33 GB/s; 16 KiB 26.33 us 0.62 GB/s; 32 KiB 28.28 us 1.16 GB/s; 64 KiB 31.01 us 2.11 GB/s; 128 KiB 43.91 us 2.98 GB/s; 256 KiB 66.54 us 3.94 GB/s; 1 MiB 197.12 us 5.32 GB/s; 2 MiB 393.23 us 5.33 GB/s; 4 MiB 784.19 us 5.35 GB/s; 8 MiB 1569.63 us 5.34 GB/s; 16 MiB 3123.08 us 5.37 GB/s; 32 MiB 6224.43 us 5.39 GB/s; 64 MiB 12446.4 us 5.39 GB/s; 128 MiB 24896.6 us 5.39 GB/s; 256 MiB 49775.0 us 5.39 GB/s

### nccl-tests arm `nt-on-p2p` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `alltoall_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 78.44 us 0.04 GB/s; 8 KiB 81.32 us 0.08 GB/s; 16 KiB 82.98 us 0.15 GB/s; 32 KiB 72.94 us 0.34 GB/s; 64 KiB 62.11 us 0.79 GB/s; 128 KiB 52.57 us 1.87 GB/s; 256 KiB 60.27 us 3.26 GB/s; 1 MiB 109.82 us 7.16 GB/s; 2 MiB 167.58 us 9.39 GB/s; 4 MiB 276.77 us 11.37 GB/s; 8 MiB 513.01 us 12.26 GB/s; 16 MiB 828.52 us 15.19 GB/s; 32 MiB 1451.62 us 17.34 GB/s; 64 MiB 2705.75 us 18.60 GB/s; 128 MiB 5176.34 us 19.45 GB/s; 256 MiB 10180.1 us 19.78 GB/s
- `sendrecv_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 48.78 us 0.08 GB/s; 8 KiB 41.46 us 0.20 GB/s; 16 KiB 38.08 us 0.43 GB/s; 32 KiB 33.19 us 0.99 GB/s; 64 KiB 29.54 us 2.22 GB/s; 128 KiB 37.81 us 3.47 GB/s; 256 KiB 46.97 us 5.58 GB/s; 1 MiB 102.13 us 10.27 GB/s; 2 MiB 176.49 us 11.88 GB/s; 4 MiB 276.36 us 15.18 GB/s; 8 MiB 443.21 us 18.93 GB/s; 16 MiB 786.72 us 21.33 GB/s; 32 MiB 1468.96 us 22.84 GB/s; 64 MiB 2844.21 us 23.59 GB/s; 128 MiB 5588.74 us 24.02 GB/s; 256 MiB 11072.0 us 24.24 GB/s
- `hypercube_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked

### nccl-tests arm `nt-ring-coll` (libsircl), rank 0's out-of-place time (nccl-tests' per-iteration time) and bus bandwidth

- `all_reduce_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 35.39 us 0.17 GB/s; 8 KiB 21.96 us 0.56 GB/s; 16 KiB 24.85 us 0.99 GB/s; 32 KiB 43.52 us 1.13 GB/s; 64 KiB 38.56 us 2.55 GB/s; 128 KiB 43.30 us 4.54 GB/s; 256 KiB 58.78 us 6.69 GB/s; 1 MiB 144.39 us 10.89 GB/s; 2 MiB 241.20 us 13.04 GB/s; 4 MiB 403.33 us 15.60 GB/s; 8 MiB 634.32 us 19.84 GB/s; 16 MiB 1151.32 us 21.86 GB/s; 32 MiB 2198.17 us 22.90 GB/s; 64 MiB 4256.94 us 23.65 GB/s; 128 MiB 8372.15 us 24.05 GB/s; 256 MiB 16590.3 us 24.27 GB/s
- `all_gather_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 22.30 us 0.14 GB/s; 8 KiB 22.08 us 0.28 GB/s; 16 KiB 24.08 us 0.51 GB/s; 32 KiB 23.10 us 1.06 GB/s; 64 KiB 24.73 us 1.99 GB/s; 128 KiB 25.93 us 3.79 GB/s; 256 KiB 37.98 us 5.18 GB/s; 1 MiB 76.27 us 10.31 GB/s; 2 MiB 130.26 us 12.07 GB/s; 4 MiB 226.00 us 13.92 GB/s; 8 MiB 362.43 us 17.36 GB/s; 16 MiB 619.10 us 20.32 GB/s; 32 MiB 1133.56 us 22.20 GB/s; 64 MiB 2172.92 us 23.16 GB/s; 128 MiB 4227.26 us 23.81 GB/s; 256 MiB 8346.32 us 24.12 GB/s
- `reduce_scatter_perf -b 8 -e 256M -f 2 -d bfloat16 -c 1 -n 20 -w 5`: exit status per rank 0 0 0 0, every value checked
  4 KiB 20.17 us 0.15 GB/s; 8 KiB 16.15 us 0.38 GB/s; 16 KiB 18.62 us 0.66 GB/s; 32 KiB 21.33 us 1.15 GB/s; 64 KiB 22.85 us 2.15 GB/s; 128 KiB 24.65 us 3.99 GB/s; 256 KiB 36.86 us 5.33 GB/s; 1 MiB 84.41 us 9.32 GB/s; 2 MiB 140.03 us 11.23 GB/s; 4 MiB 219.48 us 14.33 GB/s; 8 MiB 355.65 us 17.69 GB/s; 16 MiB 616.37 us 20.41 GB/s; 32 MiB 1131.69 us 22.24 GB/s; 64 MiB 2183.99 us 23.05 GB/s; 128 MiB 4343.37 us 23.18 GB/s; 256 MiB 8654.24 us 23.26 GB/s

