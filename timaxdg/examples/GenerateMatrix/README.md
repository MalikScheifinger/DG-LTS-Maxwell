# Example: Generate Matrix

This example sets up a basic environment to work with the library, assembles several matrices and outputs them to the filesystem. It further shows how to save the matrices to disk.

## Build

Configure the global refinement level defined by the global variable `GLOBAL_REFINEMENT` at the top of the file `GenerateMatrix.cpp`

After setting up the CMake project, enter at the root of the build folder the example folder and build the target
```bash
    cd examples/GenerateMatrix/
    make -j<n>
```

## Execute
Execute the binary with
```bash
./GenerateMatrix
```
## Output

The following matrices are generated and saved in the folder `matrices/`.

- 3D
  - Mass matrix: `mass_matrix_3D.dat`
  - Inverse mass matrix: `mass_matrix_inv_3D.dat`
  - Curl matrix: `curl_matrix_3D.dat`
  - Stabilization matrix: `stabilization_matrix_3D.dat`
- TE
  - Mass matrix: `mass_matrix_te.dat`
  - Inverse mass matrix: `mass_matrix_inv_te.dat`
  - Curl matrix: `curl_matrix_te.dat`
  - Stabilization matrix: `stabilization_matrix_te.dat`

The files can be easily imported in `python` or `matlab` for further analysis and visualization. 

The following snippet shows how to plot the sparsity pattern in `python`.

```python
# import matrix
[values, rows, columns] = numpy.loadtxt('curl_matrix_te.dat')
rows = rows.astype('int64')
columns = columns.astype('int64')
matrix = scipy.sparse.csr_matrix((values,(rows, columns)))
# generate spy plot
matplotlib.pyplot.spy(matrix)
```