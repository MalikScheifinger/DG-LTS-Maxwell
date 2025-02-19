#include <iostream>
#include <list>

#include "gtest/gtest.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/sparse_direct.h>

#include "IsotropicConstant.h"
#include "OutputMatrix.h"
#include "AssemblerTE.h"

//#define OUTPUT_MATRICES
#define TOLERANCE (10e-8)

class TEGloballyRefined : public ::testing::Test {
 protected:
  void SetUp() override {

	triangulation = std::make_shared<dealii::Triangulation<2>>();
	dealii::GridGenerator::hyper_cube(*triangulation);
	triangulation->refine_global(6);

	dof_handler = std::make_shared<dealii::DoFHandler<2>>(*triangulation);

	mapping = std::make_shared<const dealii::MappingQ1<2>>();
	quadrature = std::make_shared<const dealii::QGauss<2>>(2);
	face_quadrature = std::make_shared<const dealii::QGauss<1>>(2);

	fe = std::make_shared<dealii::FESystem<2>>(
		dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 1), 1,
		dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 2), 1);

	dof_handler->distribute_dofs(*fe);
	std::vector<unsigned int> block_components = {0, 1, 1};
	dealii::DoFRenumbering::component_wise(*dof_handler, block_components);

	mu = std::make_shared<MaxwellProblem::Data::IsotropicConstant<2>>(1);
  eps = std::make_shared<MaxwellProblem::Data::IsotropicConstant<2>>(1);

  assembler = std::make_shared<MaxwellProblem::Assembling::AssemblerTE>(
		*fe,
		*mapping,
		*quadrature,
		*face_quadrature,
		*dof_handler,
		*mu,
		*eps);
  }

  std::shared_ptr<dealii::Triangulation<2>> triangulation;
  std::shared_ptr<dealii::DoFHandler<2>> dof_handler;
  std::shared_ptr<const dealii::MappingQ1<2>> mapping;

  std::shared_ptr<const dealii::QGauss<2>> quadrature;
  std::shared_ptr<const dealii::QGauss<1>> face_quadrature;
  std::shared_ptr<dealii::FESystem<2>> fe;
  std::shared_ptr<MaxwellProblem::Data::IsotropicConstant<2>> mu;
  std::shared_ptr<MaxwellProblem::Data::IsotropicConstant<2>> eps;

  std::shared_ptr<MaxwellProblem::Assembling::AssemblerTE> assembler;
};

class TELocallyRefined : public ::testing::Test {
 protected:
  void SetUp() override {

	triangulation = std::make_shared<dealii::Triangulation<2>>();
	dealii::GridGenerator::hyper_cube(*triangulation);
	triangulation->refine_global(5);

  for (const auto &cell : triangulation->active_cell_iterators()) {
	  const auto center = cell->center();
	  if (0.25 < center[0]
		  && center[0] < 0.75
		  && 0.25 < center[1]
		  && center[1] < 0.75) {
		cell->set_refine_flag();
	  }
	}

	triangulation->prepare_coarsening_and_refinement();
	triangulation->execute_coarsening_and_refinement();
  

	dof_handler = std::make_shared<dealii::DoFHandler<2>>(*triangulation);

	mapping = std::make_shared<const dealii::MappingQ1<2>>();
	quadrature = std::make_shared<const dealii::QGauss<2>>(2);
	face_quadrature = std::make_shared<const dealii::QGauss<1>>(2);

	fe = std::make_shared<dealii::FESystem<2>>(
		dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 1), 1,
		dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 2), 1);

	dof_handler->distribute_dofs(*fe);
	std::vector<unsigned int> block_components = {0, 1, 1};
	dealii::DoFRenumbering::component_wise(*dof_handler, block_components);

	mu = std::make_shared<MaxwellProblem::Data::IsotropicConstant<2>>(1);
  eps = std::make_shared<MaxwellProblem::Data::IsotropicConstant<2>>(1);

  assembler = std::make_shared<MaxwellProblem::Assembling::AssemblerTE>(
		*fe,
		*mapping,
		*quadrature,
		*face_quadrature,
		*dof_handler,
		*mu,
		*eps);
  }

  std::shared_ptr<dealii::Triangulation<2>> triangulation;
  std::shared_ptr<dealii::DoFHandler<2>> dof_handler;
  std::shared_ptr<const dealii::MappingQ1<2>> mapping;

  std::shared_ptr<const dealii::QGauss<2>> quadrature;
  std::shared_ptr<const dealii::QGauss<1>> face_quadrature;
  std::shared_ptr<dealii::FESystem<2>> fe;
  std::shared_ptr<MaxwellProblem::Data::IsotropicConstant<2>> mu;
  std::shared_ptr<MaxwellProblem::Data::IsotropicConstant<2>> eps;

  std::shared_ptr<MaxwellProblem::Assembling::AssemblerTE> assembler;
};

// Mass Matrix

TEST_F(TEGloballyRefined, assemble_mass_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_globally.dat");
    mass_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_globally.dat");
    mass_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_globally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_globally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_locally.dat");
    mass_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_locally.dat");
    mass_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_locally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_locally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_mass_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix_parallel(mass_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_parallel_globally.dat");
    mass_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_parallel_globally.dat");
    mass_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_parallel_globally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_parallel_globally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix_parallel(mass_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_parallel_locally.dat");
    mass_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_parallel_locally.dat");
    mass_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_parallel_locally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_parallel_locally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_mass_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix);
  assembler->assemble_mass_matrix_parallel(mass_matrix_parallel);

  mass_matrix.add(-1, mass_matrix_parallel);

  // check matrices
  ASSERT_NEAR(mass_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix);
  assembler->assemble_mass_matrix_parallel(mass_matrix_parallel);

  mass_matrix.add(-1, mass_matrix_parallel);

  // check matrices
  ASSERT_NEAR(mass_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_mass_matrix_inv) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_inv;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix, mass_matrix_inv);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_inv_globally.dat");
    mass_matrix_inv.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_inv_globally.dat");
    mass_matrix_inv.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_inv_globally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_inv_globally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix_inv);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix_inv) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_inv;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix, mass_matrix_inv);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_inv_locally.dat");
    mass_matrix_inv.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_inv_locally.dat");
    mass_matrix_inv.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_inv_locally.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_inv_locally.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix_inv);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_mass_matrix_inv_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_inv;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix_parallel(mass_matrix, mass_matrix_inv);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_inv_globally_parallel.dat");
    mass_matrix_inv.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_inv_globally_parallel.dat");
    mass_matrix_inv.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_inv_globally_parallel.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_inv_globally_parallel.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix_inv);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix_inv_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_inv;
  dealii::BlockSparseMatrix<double> mass_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix_parallel(mass_matrix, mass_matrix_inv);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/mass_HH_inv_locally_parallel.dat");
    mass_matrix_inv.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/mass_EE_inv_locally_parallel.dat");
    mass_matrix_inv.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/mass_HH_inv_locally_parallel.dat");
    mass_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/mass_EE_inv_locally_parallel.dat");
    mass_matrix_compare.block(1,1).block_read(input); 
  }

  mass_matrix_compare.add(-1., mass_matrix_inv);

  ASSERT_NEAR(mass_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_mass_matrix_inv_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_parallel;
  dealii::BlockSparseMatrix<double> mass_matrix_inv, mass_matrix_inv_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_parallel.reinit(pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_inv_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix, mass_matrix_inv);
  assembler->assemble_mass_matrix_parallel(mass_matrix_parallel, mass_matrix_inv_parallel);

  mass_matrix.add(-1., mass_matrix_parallel);
  mass_matrix_inv.add(-1., mass_matrix_inv_parallel);

  // check matrices
  ASSERT_NEAR(mass_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_inv.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_inv.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_mass_matrix_inv_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> mass_matrix, mass_matrix_parallel;
  dealii::BlockSparseMatrix<double> mass_matrix_inv, mass_matrix_inv_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_mass_pattern(mass_matrix, pattern);
  mass_matrix_parallel.reinit(pattern);
  mass_matrix_inv.reinit(pattern);
  mass_matrix_inv_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_mass_matrix(mass_matrix, mass_matrix_inv);
  assembler->assemble_mass_matrix_parallel(mass_matrix_parallel, mass_matrix_inv_parallel);

  mass_matrix.add(-1., mass_matrix_parallel);
  mass_matrix_inv.add(-1., mass_matrix_inv_parallel);

  // check matrices
  ASSERT_NEAR(mass_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_inv.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(mass_matrix_inv.block(1,1).linfty_norm(), 0, TOLERANCE);
}

// Curl

TEST_F(TEGloballyRefined, assemble_curl_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparseMatrix<double> curl_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix(curl_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/curl_HH_globally.dat");
    curl_matrix.block(0,1).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/curl_EE_globally.dat");
    curl_matrix.block(1,0).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/curl_HH_globally.dat");
    curl_matrix_compare.block(0,1).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/curl_EE_globally.dat");
    curl_matrix_compare.block(1,0).block_read(input); 
  }

  curl_matrix_compare.add(-1., curl_matrix);

  ASSERT_NEAR(curl_matrix_compare.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix_compare.block(1,0).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_curl_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparseMatrix<double> curl_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix(curl_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/curl_HH_locally.dat");
    curl_matrix.block(0,1).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/curl_EE_locally.dat");
    curl_matrix.block(1,0).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/curl_HH_locally.dat");
    curl_matrix_compare.block(0,1).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/curl_EE_locally.dat");
    curl_matrix_compare.block(1,0).block_read(input); 
  }

  curl_matrix_compare.add(-1., curl_matrix);

  ASSERT_NEAR(curl_matrix_compare.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix_compare.block(1,0).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_curl_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparseMatrix<double> curl_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix_parallel(curl_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/curl_HH_parallel_globally.dat");
    curl_matrix.block(0,1).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/curl_EE_parallel_globally.dat");
    curl_matrix.block(1,0).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/curl_HH_parallel_globally.dat");
    curl_matrix_compare.block(0,1).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/curl_EE_parallel_globally.dat");
    curl_matrix_compare.block(1,0).block_read(input); 
  }

  curl_matrix_compare.add(-1., curl_matrix);

  ASSERT_NEAR(curl_matrix_compare.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix_compare.block(1,0).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_curl_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparseMatrix<double> curl_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix_parallel(curl_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/curl_HH_parallel_locally.dat");
    curl_matrix.block(0,1).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/curl_EE_parallel_locally.dat");
    curl_matrix.block(1,0).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/curl_HH_parallel_locally.dat");
    curl_matrix_compare.block(0,1).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/curl_EE_parallel_locally.dat");
    curl_matrix_compare.block(1,0).block_read(input); 
  }

  curl_matrix_compare.add(-1., curl_matrix);

  ASSERT_NEAR(curl_matrix_compare.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix_compare.block(1,0).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_curl_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix, curl_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix(curl_matrix);
  assembler->assemble_curl_matrix_parallel(curl_matrix_parallel);

  curl_matrix.add(-1, curl_matrix_parallel);

  // check matrices
  ASSERT_NEAR(curl_matrix.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix.block(1,0).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_curl_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> curl_matrix, curl_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_curl_pattern(curl_matrix, pattern);
  curl_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_curl_matrix(curl_matrix);
  assembler->assemble_curl_matrix_parallel(curl_matrix_parallel);

  curl_matrix.add(-1, curl_matrix_parallel);

  // check matrices
  ASSERT_NEAR(curl_matrix.block(0,1).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(curl_matrix.block(1,0).linfty_norm(), 0, TOLERANCE);
}

// Stab

TEST_F(TEGloballyRefined, assemble_stab_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix;
  dealii::BlockSparseMatrix<double> stab_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix(stab_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/stab_HH_globally.dat");
    stab_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/stab_EE_globally.dat");
    stab_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/stab_HH_globally.dat");
    stab_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/stab_EE_globally.dat");
    stab_matrix_compare.block(1,1).block_read(input); 
  }

  stab_matrix_compare.add(-1., stab_matrix);

  ASSERT_NEAR(stab_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_stab_matrix) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix;
  dealii::BlockSparseMatrix<double> stab_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix(stab_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/stab_HH_locally.dat");
    stab_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/stab_EE_locally.dat");
    stab_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/stab_HH_locally.dat");
    stab_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/stab_EE_locally.dat");
    stab_matrix_compare.block(1,1).block_read(input); 
  }

  stab_matrix_compare.add(-1., stab_matrix);

  ASSERT_NEAR(stab_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_stab_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix;
  dealii::BlockSparseMatrix<double> stab_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix_parallel(stab_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/stab_HH_parallel_globally.dat");
    stab_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/stab_EE_parallel_globally.dat");
    stab_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/stab_HH_parallel_globally.dat");
    stab_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/stab_EE_parallel_globally.dat");
    stab_matrix_compare.block(1,1).block_read(input); 
  }

  stab_matrix_compare.add(-1., stab_matrix);

  ASSERT_NEAR(stab_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_stab_matrix_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix;
  dealii::BlockSparseMatrix<double> stab_matrix_compare;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_compare.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix_parallel(stab_matrix);

  #ifdef OUTPUT_MATRICES
  {
    std::ofstream output("matrices/TE/stab_HH_parallel_locally.dat");
    stab_matrix.block(0,0).block_write(output);
  }
  {
    std::ofstream output("matrices/TE/stab_EE_parallel_locally.dat");
    stab_matrix.block(1,1).block_write(output);
  }
  #endif

  {
    std::ifstream input("matrices/TE/stab_HH_parallel_locally.dat");
    stab_matrix_compare.block(0,0).block_read(input); 
  }
  {
    std::ifstream input("matrices/TE/stab_EE_parallel_locally.dat");
    stab_matrix_compare.block(1,1).block_read(input); 
  }

  stab_matrix_compare.add(-1., stab_matrix);

  ASSERT_NEAR(stab_matrix_compare.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix_compare.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TEGloballyRefined, assemble_stab_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix, stab_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix(stab_matrix);
  assembler->assemble_stabilization_matrix_parallel(stab_matrix_parallel);

  stab_matrix.add(-1, stab_matrix_parallel);

  // check matrices
  ASSERT_NEAR(stab_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
}

TEST_F(TELocallyRefined, assemble_stab_matrix_serial_against_parallel) {

  // matrices
  dealii::BlockSparseMatrix<double> stab_matrix, stab_matrix_parallel;
  dealii::BlockSparsityPattern pattern;
  assembler->generate_stabilization_pattern(stab_matrix, pattern);
  stab_matrix_parallel.reinit(pattern);

  // calculate matrices
  assembler->assemble_stabilization_matrix(stab_matrix);
  assembler->assemble_stabilization_matrix_parallel(stab_matrix_parallel);

  stab_matrix.add(-1, stab_matrix_parallel);

  // check matrices
  ASSERT_NEAR(stab_matrix.block(0,0).linfty_norm(), 0, TOLERANCE);
  ASSERT_NEAR(stab_matrix.block(1,1).linfty_norm(), 0, TOLERANCE);
}