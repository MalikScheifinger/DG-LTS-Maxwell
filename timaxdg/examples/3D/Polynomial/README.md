# Example: Polynomial 3D

This example calculates solutions for various integrators against a polynomial solution 
in order to estimate an convergence order of the integrator.

## Build

Check out different pre processor variables at the top of the file `Polynomial.cpp` to set up different time
step widths and integrators

After setting up the CMake project, enter at the root of the build folder the example folder and build the target
```bash
    cd examples/3D/Polynomial/
    make -j<n>
```

## Execute
Execute the binary with
```bash
./Polynomial_3D
```
## Output

The software will print some process information on the terminal and will
finally show a convergence table. The output should look like

```
| time step width | inverse time step width | max l2 error | max l2 error...red.rate.log2 | 
| 0.01000000      | 100.00000000            | 0.00014527   | -                            | 
| 0.00905724      | 110.40895137            | 0.00011743   | 2.15                         | 
| 0.00820335      | 121.90136542            | 0.00009493   | 2.15                         | 
| 0.00742997      | 134.59001926            | 0.00007881   | 1.88                         | 
| 0.00672950      | 148.59942891            | 0.00006474   | 1.99                         | 
| 0.00609507      | 164.06707120            | 0.00005388   | 1.85                         | 
| 0.00552045      | 181.14473285            | 0.00004413   | 2.02                         | 
| 0.00500000      | 200.00000000            | 0.00003632   | 1.97                         | 
```

The last column shows an estimated convergence order.