# TiMaxdG

TiMaxdG is a C++ code library for simulations of Maxwell equations in 2D and 3D.
The library contains discontinuous Galerkin finite element methods for the space discretization 
and explicit as well as implicit schemes for the time integration.
The code is based on the finite element library [deal.II](https://www.dealii.org).

TiMaxdG is developed in project [A4](https://www.waves.kit.edu/A4.php)
of the Collaborative Research Center 1173 [»Wave phenomena: analysis and numerics«](https://www.waves.kit.edu)
at Karlsruhe Institute of Technology.
The [authors](./AUTHORS.md) acknowledge funding by the 
Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 258734477 - SFB 1173.


## Requirements

The following versions are known to work with this library. 
- GCC 9.4 or later
- CMake 3.16 or later
- deal.II 9.3.0 or later

You will find the latest version of deal.II [here](https://www.dealii.org/download.html).

## Compilation

Download the repository or clone it with git:
```bash
    git clone https://git.scc.kit.edu/dg-maxwell/timaxdg.git
```
Enter the repository and create a build folder
```bash
    cd timaxdg/
    mkdir build
    cd build/
```
Next, configure the CMake project
```bash
    cmake .. -DCMAKE_BUILD_TYPE=Release
```
Alternatively, configure the project in debug mode. 
Notice that this will slow down the calculation immensely.
```bash
    cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Finally, build the library
```bash
    make -j<n>
```
The variable `n` describes the number of processes involved in building. 

## Tests

At the root of the build folder, change to test folder and compile the unit tests with
```
    cd tests
    make -jn TEMatrixAssemblingTest
    make -jn 3DMatrixAssemblingTest
```
The variable `n` describes the number of processes involved in building.

Run the assembling tests with
```
    ./TEMatrixAssemblingTest
    ./3DMatrixAssemblingTest
```

## Benchmarks

For assembly benchmarks, read the [README.md](./benchmark/README.md) in `./benchmark`.

## Examples

Check out the example folder. It contains basic examples
that show how the library can be used.

## Documentation

You can generate a documentation for different functions and classes with
doxygen.

```bash
    doxygen Doxyfile
```

You can open the generated web page with your browser. The index.html is located
under `build/doc_doxygen/html/index.html`.
