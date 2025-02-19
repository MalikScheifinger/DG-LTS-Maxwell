# Example: Minimal 3D

This example sets up a basic environment for assembly and time integration.
It calculates a solution, outputs `vtu`-files and calculates an error against
the cavity solution.

## Build

Check out different pre processor variables at the top of the file `Polynomial.cpp` to set up different time
step widths and integrators

After setting up the CMake project, enter at the root of the build folder the example folder and build the target
```bash
    cd examples/3D/
    make -j<n>
```

## Execute
Execute the binary with
```bash
./Minimal_3D
```
## Output

The software will print some process information on the terminal.

It will further create an folder `output/` where you can find the
stored solution in `vtu`-format. The solution can be visualized with
software like `ParaView`.
