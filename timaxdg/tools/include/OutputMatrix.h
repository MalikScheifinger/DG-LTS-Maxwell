#ifndef TOOLS_OUTPUTMATRIX_H_
#define TOOLS_OUTPUTMATRIX_H_

#include <fstream>
#include <iostream>
#include <vector>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

/**
 * @brief Contains a bunch of useful functions.
 * 
 */
namespace MaxwellProblem::Tools {

/**
 * @brief Writes dealii::SparseMatrix to stream in consumable format.
 * 
 * Writes values with corresponding rows and columns in different lines separated
 * by white space characters.
 * 
 * This format is consumable with different languages.
 * 
 * A typical python import would look like
 * \code {.py}
 * [values, rows, columns] = numpy.loadtxt('matrix.dat')
 * rows = rows.astype('int64')
 * columns = columns.astype('int64')
 * matrix = scipy.sparse.csr_matrix((values,(rows, columns)))
 * \endcode
 * 
 * A typical matlab import would look like
 * \code {.m}
 * data = readmatrix('matrix.dat');
 * mat = sparse(int64(data(2,:))+1, int64(data(3,:))+1, data(1,:));
 * \endcode
 * s
 * @tparam number Number type used by the matrix.
 * @param matrix 
 * @param out Output stream data is written to.
 * @param eps Values with smaller absolute values are treated as zeros.
 * @param precision Float precision used to write values to stream.
 */
template<typename number>
void output_matrix(dealii::SparseMatrix<number> &matrix,
				   std::ostream &out,
				   const double eps = 10e-12,
				   const unsigned int precision = 9) {

  out.precision(precision);

  const auto &sparsity_pattern = matrix.get_sparsity_pattern();

  std::vector<number> rows;
  std::vector<number> cols;
  std::vector<number> values;

  rows.reserve(matrix.n_nonzero_elements());
  cols.reserve(matrix.n_nonzero_elements());
  values.reserve(matrix.n_nonzero_elements());

  for (const auto &entry : sparsity_pattern) {
	//row , column
	if (!entry.is_valid_entry()) continue;
	const auto row = entry.row();
	const auto col = entry.column();

	const auto value = matrix(row, col);
	if (fabs(value) > eps) {
	  rows.push_back(row);
	  cols.push_back(col);
	  values.push_back(value);
	}
  }

  for (const auto &d : values)
	out << d << ' ';
  out << '\n';

  for (const auto &r : rows)
	out << r << ' ';
  out << '\n';

  for (const auto &c : cols)
	out << c << ' ';
  out << '\n';
  out << std::flush;
}

/**
 * @brief Writes dealii::BlockSparseMatrix to stream in consumable format.
 * 
 * See output_matrix()
 */
template<typename number>
void output_matrix(dealii::BlockSparseMatrix<number> &matrix,
				   std::ostream &out,
				   const double eps = 10e-12,
				   const unsigned int precision = 9) {

  out.precision(precision);

  const auto &global_sparsity_pattern = matrix.get_sparsity_pattern();
  const auto &row_indices = global_sparsity_pattern.get_row_indices();
  const auto &col_indices = global_sparsity_pattern.get_column_indices();

  std::vector<number> global_rows;
  std::vector<number> global_cols;
  std::vector<number> global_values;

  global_rows.reserve(matrix.n_nonzero_elements());
  global_cols.reserve(matrix.n_nonzero_elements());
  global_values.reserve(matrix.n_nonzero_elements());

  for (std::size_t row_idx = 0; row_idx < global_sparsity_pattern.n_block_rows(); row_idx++) {
	for (std::size_t col_idx = 0; col_idx < global_sparsity_pattern.n_block_cols(); col_idx++) {

	  const auto &block_sparsity_pattern = global_sparsity_pattern.block(row_idx, col_idx);
	  const auto &block_matrix = matrix.block(row_idx, col_idx);

	  for (const auto &entry : block_sparsity_pattern) {

		if (!entry.is_valid_entry()) continue;

		const auto block_row = entry.row();
		const auto block_col = entry.column();
		const auto global_row = row_indices.local_to_global(row_idx, block_row);
		const auto global_col = col_indices.local_to_global(col_idx, block_col);

		const auto value = block_matrix(block_row, block_col);
		if (fabs(value) > eps) {
		  global_rows.push_back(global_row);
		  global_cols.push_back(global_col);
		  global_values.push_back(value);
		}
	  }
	}
  }

  for (const auto &d : global_values)
	out << d << ' ';
  out << '\n';

  for (const auto &r : global_rows)
	out << r << ' ';
  out << '\n';

  for (const auto &c : global_cols)
	out << c << ' ';
  out << '\n';
  out << std::flush;
}

}// namespace MaxwellProblem::Tools

#endif /* TOOLS_OUTPUTMATRIX_H_ */
