#ifndef DATA_TYPES_LOCALLY_IMPLICIT_OPERATORS_H_
#define DATA_TYPES_LOCALLY_IMPLICIT_OPERATORS_H_

#include <cstddef>
#include <utility>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator_tools.h>

namespace MaxwellProblem::LocallyImplicit {

template<typename Range = dealii::BlockVector<double>,
		 typename Domain = Range,
		 typename BlockPayload = dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>,
		 typename BlockMatrixType>
BlockLinearOperator<Range, Domain, BlockPayload>
block_operator_patch(
	const BlockMatrixType &block_matrix,
	const std::pair<std::size_t, std::size_t> range_patch,
	const std::pair<std::size_t, std::size_t> domain_patch) {
  using BlockType =
	  typename dealii::BlockLinearOperator<Range, Domain, BlockPayload>::BlockType;

  // rows -> range
  // cols -> domain

#ifdef DEBUG
  const unsigned int n_rows = block_matrix.n_block_rows();
  const unsigned int n_cols = block_matrix.n_block_cols();
  // range check
  AssertIndexRange(range_patch.first, n_rows);
  AssertIndexRange(range_patch.second - 1, n_rows);

  // domain check
  AssertIndexRange(domain_patch.first, n_cols);
  AssertIndexRange(domain_patch.second - 1, n_cols);
#endif// DEBUG

  dealii::BlockLinearOperator<Range, Domain, BlockPayload> return_op{
	  BlockPayload(block_matrix, block_matrix)};

  return_op.n_block_rows = [range_patch]() -> unsigned int {
	return range_patch.second - range_patch.first;
  };

  return_op.n_block_cols = [domain_patch]() -> unsigned int {
	return domain_patch.second - domain_patch.first;
  };

  return_op.block = [&block_matrix, range_patch, domain_patch](
						unsigned int i,
						unsigned int j) -> BlockType {
#ifdef DEBUG
	const unsigned int n_rows = range_patch.second - range_patch.first;
	const unsigned int n_cols = domain_patch.second - domain_patch.first;
	AssertIndexRange(i, n_rows);
	AssertIndexRange(j, n_cols);
#endif

	return BlockType(block_matrix.block(i + range_patch.first, j + domain_patch.first));
  };

  dealii::internal::BlockLinearOperatorImplementation::populate_linear_operator_functions(return_op);
  return return_op;
}

template<std::size_t m_rows,
		 std::size_t n_cols,
		 typename Range = dealii::BlockVector<double>,
		 typename Domain = Range,
		 typename BlockPayload = dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>,
		 typename BlockMatrixType>
BlockLinearOperator<Range, Domain, BlockPayload>
block_operator_masked(
	const BlockMatrixType &block_matrix,
	const std::array<std::array<bool, n_cols>, m_rows> &mask) {
// check if mask has the same block size as block_matrix
#ifdef DEBUG
  AssertThrow(m_rows == block_matrix.n_block_rows(), dealii::ExcInternalError());
  AssertThrow(n_cols == block_matrix.n_block_cols(), dealii::ExcInternalError());
#endif//Debug

  using BlockType =
	  typename BlockLinearOperator<Range, Domain, BlockPayload>::BlockType;

  // TODO: Create block payload so that this can be initialized correctly
  BlockLinearOperator<Range, Domain, BlockPayload> return_op{BlockPayload()};

  return_op.n_block_rows = []() -> unsigned int { return m_rows; };

  return_op.n_block_cols = []() -> unsigned int { return n_cols; };

  return_op.block = [&block_matrix, &mask](unsigned int i, unsigned int j) -> BlockType {
#ifdef DEBUG
	AssertIndexRange(i, m_rows);
	AssertIndexRange(j, n_cols);
#endif//Debug

	// j -> col
	// i -> row
	auto op = dealii::linear_operator(block_matrix.block(i, j));
	switch (mask[i][j]) {
	  case true:
		return op;
	  case false:
		return dealii::null_operator(op);
	}
  };

  dealii::internal::BlockLinearOperatorImplementation::populate_linear_operator_functions(return_op);
  return return_op;
}

template<std::size_t m_rows,
		 std::size_t n_cols,
		 typename Range = dealii::BlockVector<double>,
		 typename Domain = Range,
		 typename BlockPayload = dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>,
		 typename BlockMatrixType>
BlockLinearOperator<Range, Domain, BlockPayload>
block_operator_masked_patch(
	const BlockMatrixType &block_matrix,
	const std::pair<std::size_t, std::size_t> range_patch,
	const std::pair<std::size_t, std::size_t> domain_patch,
	const std::array<std::array<bool, n_cols>, m_rows> &mask) {
  using BlockType =
	  typename dealii::BlockLinearOperator<Range, Domain, BlockPayload>::BlockType;

  // rows -> range
  // cols -> domain

#ifdef DEBUG
  const unsigned int block_rows = block_matrix.n_block_rows();
  const unsigned int block_cols = block_matrix.n_block_cols();
  // range check
  AssertIndexRange(range_patch.first, block_rows);
  AssertIndexRange(range_patch.second - 1, block_rows);

  // domain check
  AssertIndexRange(domain_patch.first, block_cols);
  AssertIndexRange(domain_patch.second - 1, block_cols);

  // mask size check
  AssertThrow(range_patch.second - range_patch.first == m_rows, dealii::ExcInternalError());
  AssertThrow(domain_patch.second - domain_patch.first == n_cols, dealii::ExcInternalError());
#endif// DEBUG

  dealii::BlockLinearOperator<Range, Domain, BlockPayload> return_op{
	  BlockPayload(block_matrix, block_matrix)};

  return_op.n_block_rows = [range_patch]() -> unsigned int {
	return range_patch.second - range_patch.first;
  };

  return_op.n_block_cols = [domain_patch]() -> unsigned int {
	return domain_patch.second - domain_patch.first;
  };

  return_op.block = [&block_matrix, &mask, range_patch, domain_patch](
						unsigned int i,
						unsigned int j) -> BlockType {
#ifdef DEBUG
	const unsigned int block_rows = range_patch.second - range_patch.first;
	const unsigned int block_cols = domain_patch.second - domain_patch.first;
	AssertIndexRange(i, block_rows);
	AssertIndexRange(j, block_cols);
#endif

	auto op = dealii::linear_operator(block_matrix.block(i + range_patch.first, j + domain_patch.first));
	switch (mask[i][j]) {
	  case true:
		return op;
	  case false:
		return dealii::null_operator(op);
	}
  };

  dealii::internal::BlockLinearOperatorImplementation::populate_linear_operator_functions(return_op);
  return return_op;
}

// H curl patches and masks
const std::pair<std::size_t, std::size_t> curl_H_range_patch{4, 8};
const std::pair<std::size_t, std::size_t> curl_H_domain_patch{0, 4};
const std::pair<std::size_t, std::size_t> sub_curl_H_range_patch{5, 8};
const std::pair<std::size_t, std::size_t> sub_curl_H_domain_patch{1, 4};
const std::array<std::array<bool, 4>, 4> curl_H_exp_mask =
	{{{{true, true, true, true}},
	  {{true, true, false, true}},
	  {{true, true, false, false}},
	  {{true, true, false, false}}}};
const std::array<std::array<bool, 4>, 4> curl_H_imp_mask =
	{{{{false, false, true, true}},
	  {{false, false, true, true}},
	  {{true, false, true, true}},
	  {{true, true, true, true}}}};
const std::array<std::array<bool, 3>, 3> sub_curl_H_imp_mask =
	{{{{false, true, true}},
	  {{false, true, true}},
	  {{true, true, true}}}};

// E curl patches and masks
const std::pair<std::size_t, std::size_t> curl_E_range_patch{0, 4};
const std::pair<std::size_t, std::size_t> curl_E_domain_patch{4, 8};
const std::pair<std::size_t, std::size_t> sub_curl_E_range_patch{1, 4};
const std::pair<std::size_t, std::size_t> sub_curl_E_domain_patch{5, 8};
const std::array<std::array<bool, 4>, 4> curl_E_exp_mask =
	{{{{true, true, true, true}},
	  {{true, true, false, true}},
	  {{true, true, false, false}},
	  {{true, true, false, false}}}};
const std::array<std::array<bool, 4>, 4> curl_E_imp_mask =
	{{{{false, false, true, true}},
	  {{false, false, false, true}},
	  {{true, true, true, true}},
	  {{true, true, true, true}}}};
const std::array<std::array<bool, 3>, 3> sub_curl_E_imp_mask =
	{{{{false, false, true}},
	  {{true, true, true}},
	  {{true, true, true}}}};

// mass patches and masks
const std::pair<std::size_t, std::size_t> mass_H_diagonal_patch{0, 4};
const std::pair<std::size_t, std::size_t> sub_mass_H_diagonal_patch{1, 4};
const std::pair<std::size_t, std::size_t> mass_E_diagonal_patch{4, 8};
const std::pair<std::size_t, std::size_t> sub_mass_E_diagonal_patch{5, 8};
const std::array<std::array<bool, 4>, 4> full_mass_mask =
	{{{{true, false, false, false}},
	  {{false, true, false, false}},
	  {{false, false, true, false}},
	  {{false, false, false, true}}}};
const std::array<std::array<bool, 3>, 3> sub_mass_mask =
	{{{{true, false, false}},
	  {{false, true, false}},
	  {{false, false, true}}}};

struct LocallyImplicitOperators {
  using EmptyBlockPayload 
    = typename dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>;
  using Vectortype
    = typename dealii::BlockVector<double>;
  
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_H;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_H_imp;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_H_exp;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_curl_H_imp;

  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_E;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_E_imp;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> curl_E_exp;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_curl_E_imp;

  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> mass_inv_H;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_mass_inv_H;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> mass_inv_E;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_mass_inv_E;

  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> mass_H;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_mass_H;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> mass_E;
  dealii::BlockLinearOperator<Vectortype,Vectortype,EmptyBlockPayload> sub_mass_E;

  LocallyImplicitOperators(
    const dealii::BlockSparseMatrix<double>& curl_matrix,
    const dealii::BlockSparseMatrix<double>& mass_inv_matrix,
    const dealii::BlockSparseMatrix<double>& mass_matrix)
    :
    curl_H{
      block_operator_patch(
        curl_matrix, curl_H_range_patch, curl_H_domain_patch)
    },
    curl_H_imp{
      block_operator_masked_patch(
        curl_matrix, curl_H_range_patch, curl_H_domain_patch, curl_H_imp_mask)
    },
    curl_H_exp{
      block_operator_masked_patch(
        curl_matrix, curl_H_range_patch, curl_H_domain_patch, curl_H_exp_mask)
    },
    sub_curl_H_imp{
      block_operator_masked_patch(
        curl_matrix, sub_curl_H_range_patch, sub_curl_H_domain_patch, sub_curl_H_imp_mask)
    },
    curl_E{
      block_operator_patch(
        curl_matrix, curl_E_range_patch, curl_E_domain_patch)
    },
    curl_E_imp{
      block_operator_masked_patch(
        curl_matrix, curl_E_range_patch, curl_E_domain_patch, curl_E_imp_mask)
    },
    curl_E_exp{
      block_operator_masked_patch(
        curl_matrix, curl_E_range_patch, curl_E_domain_patch, curl_E_exp_mask)
    },
    sub_curl_E_imp{
      block_operator_masked_patch(
        curl_matrix, sub_curl_E_range_patch, sub_curl_E_domain_patch, sub_curl_E_imp_mask)
    },
    mass_inv_H{
      block_operator_masked_patch(
        mass_inv_matrix, mass_H_diagonal_patch, mass_H_diagonal_patch, full_mass_mask)
    },
    sub_mass_inv_H{
      block_operator_masked_patch(
        mass_inv_matrix, sub_mass_H_diagonal_patch, sub_mass_H_diagonal_patch, sub_mass_mask)
    },
    mass_inv_E{
      block_operator_masked_patch(
        mass_inv_matrix, mass_E_diagonal_patch, mass_E_diagonal_patch, full_mass_mask)
    },
    sub_mass_inv_E{
      block_operator_masked_patch(
        mass_inv_matrix, sub_mass_E_diagonal_patch, sub_mass_E_diagonal_patch, sub_mass_mask)
    },
    mass_H{
      block_operator_masked_patch(
        mass_matrix, mass_H_diagonal_patch, mass_H_diagonal_patch, full_mass_mask)
    },
    sub_mass_H{
      block_operator_masked_patch(
        mass_matrix, sub_mass_H_diagonal_patch, sub_mass_H_diagonal_patch, sub_mass_mask)
    },
    mass_E{
      block_operator_masked_patch(
        mass_matrix, mass_E_diagonal_patch, mass_E_diagonal_patch, full_mass_mask)
    },
    sub_mass_E{
      block_operator_masked_patch(
        mass_matrix, sub_mass_E_diagonal_patch, sub_mass_E_diagonal_patch, sub_mass_mask)
    }
    {}
};

} // namespace MaxwellProblem::LocallyImplicit
#endif//DATA_TYPES_LOCALLY_IMPLICIT_OPERATORS_H_