# Benchmarks

Currently, two benchmarks are available comparing a serial against a parallel assembly
of the mass matrix and respectively curl matrix.

At the root of the build directory, make sure to configure the project in
Release mode
```
    cmake .. -DCMAKE_BUILD_TYPE=Release
```

## Mass matrix

The program MassAssemblyBenchmark provides benchmarks for mass matrix assembly
in serial and parallel.
Check out the preprocessor variables for various configurations at the top of
the source file.

Compile the binary with
```console
$    make -jn MassAssemblyBenchmark
```
and run
```console
$    cd benchmark/
$    ./MassAsemblyBenchmark  --benchmark_counters_tabular=true
```

The benchmark can take some time and the output shout look like
```
------------------------------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations        DoF      DoF/s
------------------------------------------------------------------------------------------------------------
MassMatrixAssembly/1/iterations:10               0.001 s         0.001 s            10        162 120.639k/s
MassMatrixAssembly/2/iterations:10               0.011 s         0.011 s            10     1.296k 121.546k/s
MassMatrixAssembly/3/iterations:10               0.036 s         0.036 s            10     4.374k 122.495k/s
MassMatrixAssembly/4/iterations:10               0.087 s         0.087 s            10    10.368k 119.777k/s
MassMatrixAssembly/5/iterations:10               0.169 s         0.169 s            10     20.25k 119.612k/s
MassMatrixAssembly/6/iterations:10               0.291 s         0.291 s            10    34.992k 120.445k/s
MassMatrixAssembly/7/iterations:10               0.461 s         0.460 s            10    55.566k 120.677k/s
MassMatrixAssembly/8/iterations:10               0.689 s         0.689 s            10    82.944k 120.375k/s
MassMatrixAssembly/9/iterations:10               0.991 s         0.991 s            10   118.098k  119.21k/s
MassMatrixAssembly/10/iterations:10               1.39 s          1.39 s            10       162k 116.301k/s
MassMatrixAssembly/11/iterations:10               1.86 s          1.86 s            10   215.622k 115.988k/s
MassMatrixAssembly/12/iterations:10               2.37 s          2.37 s            10   279.936k 118.116k/s
MassMatrixAssembly/13/iterations:10               3.02 s          3.02 s            10   355.914k 118.021k/s
MassMatrixAssembly/14/iterations:10               3.73 s          3.73 s            10   444.528k 119.244k/s
MassMatrixAssembly/15/iterations:10               4.58 s          4.58 s            10    546.75k 119.286k/s
MassMatrixAssemblyParallel/1/iterations:10       0.002 s         0.002 s            10        162 107.266k/s
MassMatrixAssemblyParallel/2/iterations:10       0.011 s         0.011 s            10     1.296k 118.954k/s
MassMatrixAssemblyParallel/3/iterations:10       0.020 s         0.020 s            10     4.374k 216.042k/s
MassMatrixAssemblyParallel/4/iterations:10       0.033 s         0.033 s            10    10.368k 313.403k/s
MassMatrixAssemblyParallel/5/iterations:10       0.051 s         0.051 s            10     20.25k 398.718k/s
MassMatrixAssemblyParallel/6/iterations:10       0.079 s         0.078 s            10    34.992k 451.185k/s
MassMatrixAssemblyParallel/7/iterations:10       0.111 s         0.108 s            10    55.566k 515.808k/s
MassMatrixAssemblyParallel/8/iterations:10       0.160 s         0.159 s            10    82.944k 521.658k/s
MassMatrixAssemblyParallel/9/iterations:10       0.230 s         0.228 s            10   118.098k 517.389k/s
MassMatrixAssemblyParallel/10/iterations:10      0.317 s         0.301 s            10       162k  538.08k/s
MassMatrixAssemblyParallel/11/iterations:10      0.420 s         0.418 s            10   215.622k 516.136k/s
MassMatrixAssemblyParallel/12/iterations:10      0.544 s         0.539 s            10   279.936k 519.133k/s
MassMatrixAssemblyParallel/13/iterations:10      0.687 s         0.680 s            10   355.914k 523.761k/s
MassMatrixAssemblyParallel/14/iterations:10      0.856 s         0.837 s            10   444.528k 530.941k/s
MassMatrixAssemblyParallel/15/iterations:10       1.06 s          1.05 s            10    546.75k 522.954k/s
```

## Curl matrix

The program CurlAssemblyBenchmark provides benchmarks for curl matrix assembly
in serial and parallel.
Check out the preprocessor variables for various configurations at the top of
the source file.

Compile the binary with
```console
$   make -jn CurlAssemblyBenchmark
```
and run it with
```console
$    cd benchmark/
$    ./CurlAsemblyBenchmark --benchmark_counters_tabular=true
```

The benchmark can take some time and the output shout look like
```
------------------------------------------------------------------------------------------------------------
Benchmark                                            Time             CPU   Iterations        DoF      DoF/s
------------------------------------------------------------------------------------------------------------
CurlMatrixAssembly/1/iterations:10               0.002 s         0.002 s            10        162 89.1221k/s
CurlMatrixAssembly/2/iterations:10               0.026 s         0.026 s            10     1.296k 49.4069k/s
CurlMatrixAssembly/3/iterations:10               0.100 s         0.100 s            10     4.374k 43.7247k/s
CurlMatrixAssembly/4/iterations:10               0.251 s         0.251 s            10    10.368k 41.3689k/s
CurlMatrixAssembly/5/iterations:10               0.514 s         0.514 s            10     20.25k 39.4241k/s
CurlMatrixAssembly/6/iterations:10               0.895 s         0.894 s            10    34.992k 39.1206k/s
CurlMatrixAssembly/7/iterations:10                1.44 s          1.44 s            10    55.566k 38.4624k/s
CurlMatrixAssembly/8/iterations:10                2.17 s          2.17 s            10    82.944k 38.2023k/s
CurlMatrixAssembly/9/iterations:10                3.08 s          3.08 s            10   118.098k 38.3745k/s
CurlMatrixAssembly/10/iterations:10               4.25 s          4.25 s            10       162k 38.0809k/s
CurlMatrixAssembly/11/iterations:10               5.80 s          5.79 s            10   215.622k 37.2119k/s
CurlMatrixAssembly/12/iterations:10               7.75 s          7.75 s            10   279.936k 36.1247k/s
CurlMatrixAssembly/13/iterations:10               9.50 s          9.49 s            10   355.914k 37.4878k/s
CurlMatrixAssembly/14/iterations:10               12.0 s          12.0 s            10   444.528k 36.9093k/s
CurlMatrixAssembly/15/iterations:10               14.8 s          14.8 s            10    546.75k 36.9656k/s
CurlMatrixAssemblyParallel/1/iterations:10       0.007 s         0.007 s            10        162 22.3076k/s
CurlMatrixAssemblyParallel/2/iterations:10       0.033 s         0.033 s            10     1.296k 39.2955k/s
CurlMatrixAssemblyParallel/3/iterations:10       0.056 s         0.056 s            10     4.374k 78.5724k/s
CurlMatrixAssemblyParallel/4/iterations:10       0.110 s         0.110 s            10    10.368k 94.3666k/s
CurlMatrixAssemblyParallel/5/iterations:10       0.202 s         0.202 s            10     20.25k 100.162k/s
CurlMatrixAssemblyParallel/6/iterations:10       0.322 s         0.319 s            10    34.992k 109.592k/s
CurlMatrixAssemblyParallel/7/iterations:10       0.483 s         0.483 s            10    55.566k 115.106k/s
CurlMatrixAssemblyParallel/8/iterations:10       0.676 s         0.674 s            10    82.944k 122.985k/s
CurlMatrixAssemblyParallel/9/iterations:10       0.921 s         0.910 s            10   118.098k 129.755k/s
CurlMatrixAssemblyParallel/10/iterations:10       1.22 s          1.21 s            10       162k 133.407k/s
CurlMatrixAssemblyParallel/11/iterations:10       1.56 s          1.56 s            10   215.622k  138.45k/s
CurlMatrixAssemblyParallel/12/iterations:10       2.05 s          2.04 s            10   279.936k  137.05k/s
CurlMatrixAssemblyParallel/13/iterations:10       2.52 s          2.52 s            10   355.914k 141.356k/s
CurlMatrixAssemblyParallel/14/iterations:10       3.22 s          3.22 s            10   444.528k 138.102k/s
CurlMatrixAssemblyParallel/15/iterations:10       3.83 s          3.83 s            10    546.75k 142.941k/s
```