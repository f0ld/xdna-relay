# bench/

Benchmark scripts, run via `scripts/bench.sh bench/<script>.py`. Each appends rows to
`bench/results/<name>.csv` with header:

`date,commit,task,kernel,shape,dtype,iters,median_us,gflops,notes`

`RELAY_COMMIT` is set by `scripts/bench.sh`. Raw logs go to `bench/results/raw/` (gitignored).
