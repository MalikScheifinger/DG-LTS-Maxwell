#ifndef ASSEMBLING_ASSEMBLER3D_H_
#define ASSEMBLING_ASSEMBLER3D_H_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>


/**
 * @brief Collects classes and functions for assembling of various matrices and vectors.
 * 
 * This namespace collects different classes for assembling mass, curl and stabilization
 * matrices in different modes.
 * 
 * It further provides some general purpose functions for assembling right hand sides. 
 */
namespace MaxwellProblem::Assembling {

/**
 * @brief Collects functions for assembling mass, curl and stabilization matrices in 3D mode.
 * 
 * Gathers the necessary dealii objects and provides functions for the assembling of
 * - mass matrix
 * - inverse mass matrix
 * - curl matrix
 * - stabilization matrix
 * 
 * Most of the assembling routines provide a serial and a parallel implementation.
 * Both provide the same matrix and therfore the parallel implementation should be preferred.
 * The serial implementation is easier to read and is included for expository reasons.
 */
class Assembler3D {
 public:
  /**
   * @brief Construct an assembler object.
   * 
   * The typical dealii objects necessary for assembling should be familiar to the user.
   * Further documentation is provided by the authors of dealii.
   * 
   * The dealii::FESystem<3> assumes dealii::FE_DGQ<3> to be used and the ordering of
   * the fields is assumed to be (H_x, H_y, H_z, E_x, E_y, E_z). The various examples
   * included in this library show the correct setup of this object.
   * Note that this is not checked by the code and violation can lead to errors or
   * misleading results.
   * 
   * @param fe 
   * @param mapping 
   * @param quadrature 
   * @param face_quadrature 
   * @param dof_handler 
   * @param mu_function Magnetic permeability
   * @param eps_function Electric permittivity
   */
  Assembler3D(
	  dealii::FESystem<3> &fe,
	  const dealii::MappingQ1<3> &mapping,
	  const dealii::Quadrature<3> &quadrature,
	  const dealii::Quadrature<2> &face_quadrature,
	  dealii::DoFHandler<3> &dof_handler,
	  dealii::Function<3> &mu_function,
	  dealii::Function<3> &eps_function);

  /**
   * @brief Sets up the sparsity pattern of the mass matrix.
   * 
   * The pattern must be alive as long as the matrix is used.
   * 
   * @param mass_matrix 
   * @param mass_pattern 
   */
  void generate_mass_pattern(
	  dealii::BlockSparseMatrix<double> &mass_matrix,
	  dealii::BlockSparsityPattern &mass_pattern);

  /**
   * @brief Assembles the mass matrix.
   * 
   * @param mass_matrix 
   */
  void assemble_mass_matrix(dealii::BlockSparseMatrix<double> &mass_matrix);

  /**
   * @brief Assembles the mass matrix in parallel.
   * 
   * @param mass_matrix 
   */
  void assemble_mass_matrix_parallel(dealii::BlockSparseMatrix<double> &mass_matrix);

  /**
   * @brief Assembles the mass matrix and the inverse of the mass matrix.
   * 
   * The assembling routine inverts the individual blocks.
   * 
   * @param mass_matrix 
   * @param mass_matrix_inv 
   */
  void assemble_mass_matrix(
	  dealii::BlockSparseMatrix<double> &mass_matrix,
	  dealii::BlockSparseMatrix<double> &mass_matrix_inv);

  /**
   * @brief Assembles the mass matrix and the inverse of the mass matrix in parallel.
   * 
   * The assembling routine inverts the individual blocks.
   * 
   * @param mass_matrix 
   * @param mass_matrix_inv 
   */
  void assemble_mass_matrix_parallel(
	  dealii::BlockSparseMatrix<double> &mass_matrix,
	  dealii::BlockSparseMatrix<double> &mass_matrix_inv);

  /**
   * @brief Sets up the sparsity pattern of the mass matrix.
   * 
   * The pattern must be alive as long as the matrix is used.
   * 
   * @param curl_matrix 
   * @param curl_pattern 
   */
  void generate_curl_pattern(
	  dealii::BlockSparseMatrix<double> &curl_matrix,
	  dealii::BlockSparsityPattern &curl_pattern);

  /**
   * @brief Assembles the curl matrix.
   * 
   * @param curl_matrix 
   */
  void assemble_curl_matrix(dealii::BlockSparseMatrix<double> &curl_matrix);

  /**
   * @brief Assembles the curl matrix in parallel.
   * 
   * @param curl_matrix 
   */
  void assemble_curl_matrix_parallel(dealii::BlockSparseMatrix<double> &curl_matrix);

  /**
   * @brief Sets up the sparsity pattern of the mass matrix.
   * 
   * @param stab_matrix 
   * @param stab_pattern 
   */
  void generate_stabilization_pattern(
	  dealii::BlockSparseMatrix<double> &stab_matrix,
	  dealii::BlockSparsityPattern &stab_pattern);

  /**
   * @brief Assembles the stabilization matrix in serial.
   * 
   * 
   * @param stab_matrix 
   * @param alpha Stabilization parameter.
   */
  void assemble_stabilization_matrix(
	  dealii::BlockSparseMatrix<double> &stab_matrix,
	  double alpha = 1.);

  /**
   * @brief Assembles the stabilization matrix in serial.
   * 
   * @param stab_matrix
   * @param alpha Stabilization parameter
   */
  void assemble_stabilization_matrix_parallel(
	  dealii::BlockSparseMatrix<double> &stab_matrix,
	  double alpha = 1.);

  /**
   * @brief Assembles the load vector.
   * 
   * @param rhs_function functions that gets assembled
   * @param rhs_vector load vector
   */
  void assemble_rhs(
      const dealii::Function<3> &rhs_function,
      dealii::BlockVector<double> &rhs_vector);

 private:
  dealii::FESystem<3> &fe;
  const dealii::MappingQ1<3> &mapping;
  const dealii::Quadrature<3> &quadrature;
  const dealii::Quadrature<2> &face_quadrature;
  dealii::DoFHandler<3> &dof_handler;

  const dealii::Function<3> &mu_function;
  const dealii::Function<3> &eps_function;

  dealii::FEValues<3> fe_v;
  dealii::FEFaceValues<3> fe_v_face;
  dealii::FESubfaceValues<3> fe_v_subface;
  dealii::FEFaceValues<3> fe_v_face_neighbor;
};
}// namespace MaxwellProblem::Assembling

#endif//ASSEMBLING_3DASSEMBLER_H_