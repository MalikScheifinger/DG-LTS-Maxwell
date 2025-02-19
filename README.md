# Maxwell-DG-LTS

# LTS FOR DG-DISCRETIZED LINEAR MAXWELLS' EQUATIONS
This code was used for the numerical experiments in the paper

> LOCAL TIME-STEPPING METHODS FOR FRIEDRICHS' SYSTEMS
>
> By M. Hochbruck, M. Scheifinger

This software is published in accordance with the guidelines for safeguarding good research practice and serves to reproduce the experiments in the above-mentioned publication. 

## Requirements

The code is written in C++17 and uses the software packages
    
*   [deal.II](https://www.dealii.org/), Version 9.5.0
*   [CMake](https://cmake.org/), Version 3.22.1

The code is based on the software package [TiMaxdG](https://gitlab.kit.edu/kit/ianm/ag-numerik/projects/dg-maxwell/timaxdg) which is currently under development by Julian Dörner and Malik Scheifinger. In this program used locally implicit and local time-stepping algorithms and assembling routines are not included in the open version yet. Hence we include them in this repository.

The plots are generated with Python 3.

For the reproduction of the numerical experiments one may use [docker](https://www.docker.com/) and [enroot](https://github.com/NVIDIA/enroot).
For that, please consider the installation instructions for the tools and the files:

*   .devcontainer/Dockerfile
*   .devcontainer/README.md
*   .devcontainer/ENROOT.md

provided by Julian Dörner.

## Reproduction

### Experiment 1 (Figure 2):

The experiment is build with the commands
```bash

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cd TE-example/convergence/
    make convg_time_cavity_TE
```

The experiment is executed with the command
```bash
    ./convg_time_cavity_TE
```
The experiment outputs a convergence table into the terminal and further produces a tabular file `errors_time_cavity_LTS_LFC_eta1_dg5_globalref3_localref3_threshold1,2.txt` with the results.

The plot can be generated within the jupyter notebook `python/TE/plots.ipynb`.

### Experiment 2:

The experiment is build with the commands
```bash

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cd TE-example/LTSvsLI/
    make LTSvsLI
```

The experiment is executed with the command
```bash
    ./LTSvsLI
```
The experiment prints tabulars into the terminal listing the cpu- and wall-times of the different methods.

### Experiment 3 (Figure 3):

The experiment is build with the commands
```bash

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cd 1D-example/convergence
    make convg_cavity_1D
```

The experiment is executed with the commands
```bash
    ./convg_cavity_1D
```
The experiment prints a convergence table into the terminal and further produces two tabular files `errors_cavity_LTS_LFC_eta0,1_dg2_coarse-fine-ratio4.txt` and `errors_cavity_LTS_LFC_eta0_dg2_coarse-fine-ratio4.txt` with the results.

The plot can be generated within the jupyter notebook `python/1D/plots.ipynb`.

