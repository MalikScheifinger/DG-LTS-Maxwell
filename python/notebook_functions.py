import numpy as np
import scipy

def Cheb(p, x):
    if p == 0:
        return 1.
    if p == 1:
        return x
    else:
        return 2 * x * Cheb(p-1,x) - Cheb(p-2,x)

def DxCheb(p, x):
    if p == 0:
        return 0.
    if p == 1:
        return 1.
    else:
        return 2 * Cheb(p-1,x) + 2 * x * DxCheb(p-1, x) - DxCheb(p-2, x)
    
def nu(eta, p): return 1. + eta**2 / (2.*p**2)

def alpha(eta, p): return 2. * (DxCheb(p, nu(eta, p)))/(Cheb(p, nu(eta, p)))

def beta_hat_sqr(eta, p): return alpha(eta, p) * (nu(eta, p) + 1.)

def load_data(path, data_str):

    dofs = np.loadtxt(path + 'block_sizes' + data_str + '.dat')
    dofs_H = dofs[:4].astype('int64')
    sum_dofs_H = sum(dofs_H)
    dofs_E = dofs[4:].astype('int64')
    sum_dofs_E = sum(dofs_E)

    indH = np.cumsum(dofs_H)
    indH = np.insert(indH, 0, 0)
    indE = np.cumsum(dofs_E)
    indE = np.insert(indE, 0, 0)

    [values, rows, columns] = np.loadtxt(path + 'mass_matrix' + data_str + '.dat')
    rows = rows.astype('int64')
    columns = columns.astype('int64')
    mass_matrix = scipy.sparse.csr_array((values,(rows, columns)))

    [values, rows, columns] = np.loadtxt(path + 'inv_mass_matrix' + data_str + '.dat')
    rows = rows.astype('int64')
    columns = columns.astype('int64')
    inv_mass_matrix = scipy.sparse.csr_array((values,(rows, columns)))

    [values, rows, columns] = np.loadtxt(path + 'curl_matrix' + data_str + '.dat')
    rows = rows.astype('int64')
    columns = columns.astype('int64')
    curl_matrix = scipy.sparse.csr_array((values,(rows, columns)))

    curl_H = curl_matrix[sum_dofs_H:, :sum_dofs_H]
    curl_E = curl_matrix[:sum_dofs_H, sum_dofs_H:]

    inv_mass_H = inv_mass_matrix[:sum_dofs_H, :sum_dofs_H]
    inv_mass_E = inv_mass_matrix[sum_dofs_H:, sum_dofs_H:]

    mass_H = mass_matrix[:sum_dofs_H, :sum_dofs_H]
    mass_E = mass_matrix[sum_dofs_H:, sum_dofs_H:]

    return indH, indE, mass_H, mass_E, inv_mass_H, inv_mass_E, curl_H, curl_E

if __name__ == "__main__":
    pass