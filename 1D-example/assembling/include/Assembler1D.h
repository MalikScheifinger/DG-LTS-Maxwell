#ifndef ASSEMBLING_ASSEMBLER1D_H_
#define ASSEMBLING_ASSEMBLER1D_H_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/base/function.h>

namespace MaxwellProblem1D::Assembling {

/**
 * @brief Collects functions for assembling mass, curl and stabilization matrices in 1D.
 * 
 * Gathers the necessary dealii objects and provides functions for the assembling of
 * - mass matrix
 * - inverse mass matrix
 * - curl matrix
 * - stabilization matrix
 * 
 */
class Assembler1D {
 public:
  /**
  * @brief Construct an assembler object.
   * 
   * The typical dealii objects necessary for assembling should be familiar to the user.
   * Further documentation is provided by the authors of dealii.
  * 
  * @param fe 
  * @param mapping 
  * @param quadrature 
  * @param face_quadrature 
  * @param dof_handler 
  * @param mu_function Magnetic permeability
  * @param eps_function Electric permittivity
  */
  Assembler1D(
	  dealii::FESystem<1> &fe,
	  const dealii::MappingQ1<1> &mapping,
	  const dealii::Quadrature<1> &quadrature,
	  const dealii::Quadrature<0> &face_quadrature,
	  dealii::DoFHandler<1> &dof_handler,
	  dealii::Function<1> &mu_function,
	  dealii::Function<1> &eps_function);

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
   * @brief TODO!
   * 
   * @param mass_matrix 
   * @param mass_pattern 
   */
  void generate_mass_pattern_locally_implicit(
	  dealii::BlockSparseMatrix<double> &mass_matrix,
	  dealii::BlockSparsityPattern &mass_pattern);

  /**
   * @brief Assembles the mass matrix.
   * 
   * @param mass_matrix 
   */
  void assemble_mass_matrix(dealii::BlockSparseMatrix<double> &mass_matrix);


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
   * @brief Todo
   * 
   * @param curl_matrix 
   * @param curl_pattern 
   */
  void generate_curl_pattern_locally_implicit(
	  dealii::BlockSparseMatrix<double> &curl_matrix,
	  dealii::BlockSparsityPattern &curl_pattern);

  /**
   * @brief Assembles the curl matrix.
   * 
   * @param curl_matrix 
   */
  void assemble_curl_matrix(dealii::BlockSparseMatrix<double> &curl_matrix);

  
  /**
   * @brief Assembles the load vector.
   * 
   * @param rhs_function functions that gets assembled
   * @param rhs_vector load vector
   */
  void assemble_rhs(
	  const dealii::Function<1> &rhs_function,
	  dealii::BlockVector<double> &rhs_vector);

 private:
  dealii::FESystem<1> &fe;
  const dealii::MappingQ1<1> &mapping;
  const dealii::Quadrature<1> &quadrature;
  const dealii::Quadrature<0> &face_quadrature;
  dealii::DoFHandler<1> &dof_handler;

  const dealii::Function<1> &mu_function;
  const dealii::Function<1> &eps_function;

  dealii::FEValues<1> fe_v;
  dealii::FEFaceValues<1> fe_v_face;
  dealii::FESubfaceValues<1> fe_v_subface;
  dealii::FEFaceValues<1> fe_v_face_neighbor;
};
}// namespace MaxwellProblem1D::Assembling

#endif//ASSEMBLING_ASSEMBLER1D_H_