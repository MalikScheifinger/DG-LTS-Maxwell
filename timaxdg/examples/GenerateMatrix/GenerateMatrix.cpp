#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "Assembler3D.h"
#include "AssemblerTE.h"
#include "IsotropicConstant.h"
#include "OutputMatrix.h"

#define GLOBAL_REFINEMENT 3

class GenerateMatrix {
 public:
  GenerateMatrix();

  void run(unsigned int refinement_level,
		   const std::filesystem::path &path = std::filesystem::current_path(),
		   std::ostream &out = std::cout);

 private:
  void assemble_grid(unsigned int refinement_level);
  void setup_system();
  void assemble_matrices(std::ostream &out);

  // 3D
  dealii::Triangulation<3> triangulation_3d;
  dealii::MappingQ1<3> mapping_3d;
  dealii::QGauss<3> quadrature_3d;
  dealii::QGauss<2> face_quadrature_3d;
  dealii::DoFHandler<3> dof_handler_3d;
  dealii::FESystem<3> fe_3d;

  MaxwellProblem::Data::IsotropicConstant<3> mu_3d;
  MaxwellProblem::Data::IsotropicConstant<3> eps_3d;

  MaxwellProblem::Assembling::Assembler3D assembler_3d;

  dealii::BlockSparseMatrix<double> mass_matrix_3d;
  dealii::BlockSparsityPattern mass_pattern_3d;
  dealii::BlockSparseMatrix<double> mass_matrix_inv_3d;
  dealii::BlockSparsityPattern mass_pattern_inv_3d;
  dealii::BlockSparseMatrix<double> curl_matrix_3d;
  dealii::BlockSparsityPattern curl_pattern_3d;
  dealii::BlockSparseMatrix<double> stabilization_matrix_3d;
  dealii::BlockSparsityPattern stabilization_patter_3d;

  // TE
  dealii::Triangulation<2> triangulation_te;
  dealii::MappingQ1<2> mapping_te;
  dealii::QGauss<2> quadrature_te;
  dealii::QGauss<1> face_quadrature_te;
  dealii::DoFHandler<2> dof_handler_te;
  dealii::FESystem<2> fe_te;

  MaxwellProblem::Data::IsotropicConstant<2> mu_te;
  MaxwellProblem::Data::IsotropicConstant<2> eps_te;

  MaxwellProblem::Assembling::AssemblerTE assembler_te;

  dealii::BlockSparseMatrix<double> mass_matrix_te;
  dealii::BlockSparsityPattern mass_pattern_te;
  dealii::BlockSparseMatrix<double> mass_matrix_inv_te;
  dealii::BlockSparsityPattern mass_pattern_inv_te;
  dealii::BlockSparseMatrix<double> curl_matrix_te;
  dealii::BlockSparsityPattern curl_pattern_te;
  dealii::BlockSparseMatrix<double> stabilization_matrix_te;
  dealii::BlockSparsityPattern stabilization_patter_te;
};

GenerateMatrix::GenerateMatrix() : mapping_3d(),
								   quadrature_3d(4),
								   face_quadrature_3d(3),
								   fe_3d(dealii::FESystem<3>(dealii::FE_DGQ<3>(1), 3), 1,
										 dealii::FESystem<3>(dealii::FE_DGQ<3>(1), 3), 1),
								   mu_3d(1),
								   eps_3d(1),
								   assembler_3d(fe_3d, mapping_3d, quadrature_3d, face_quadrature_3d, dof_handler_3d, mu_3d, eps_3d),
								   mapping_te(),
								   quadrature_te(4),
								   face_quadrature_te(3),
								   fe_te(dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 1), 1,
										 dealii::FESystem<2>(dealii::FE_DGQ<2>(1), 2), 1),
								   mu_te(1),
								   eps_te(1),
								   assembler_te(fe_te, mapping_te, quadrature_te, face_quadrature_te, dof_handler_te, mu_te, eps_te) {}

void GenerateMatrix::assemble_grid(unsigned int refinement_level) {
  dealii::GridGenerator::hyper_cube(triangulation_3d);
  triangulation_3d.refine_global(refinement_level);

  dealii::GridGenerator::hyper_cube(triangulation_te);
  triangulation_te.refine_global(refinement_level);
}

void GenerateMatrix::setup_system() {
  // 3D
  {
	dof_handler_3d.reinit(triangulation_3d);
	dof_handler_3d.distribute_dofs(fe_3d);
	const static std::vector<unsigned int> block_components = {1, 1, 1, 0, 0, 0};
	dealii::DoFRenumbering::component_wise(dof_handler_3d, block_components);

	assembler_3d.generate_mass_pattern(mass_matrix_3d, mass_pattern_3d);
	assembler_3d.generate_mass_pattern(mass_matrix_inv_3d, mass_pattern_inv_3d);
	assembler_3d.generate_curl_pattern(curl_matrix_3d, curl_pattern_3d);
	assembler_3d.generate_stabilization_pattern(stabilization_matrix_3d, stabilization_patter_3d);
  }
  // TE
  {
	dof_handler_te.reinit(triangulation_te);
	dof_handler_te.distribute_dofs(fe_te);
	const static std::vector<unsigned int> block_components = {1, 0, 0};
	dealii::DoFRenumbering::component_wise(dof_handler_te, block_components);

	assembler_te.generate_mass_pattern(mass_matrix_te, mass_pattern_te);
	assembler_te.generate_mass_pattern(mass_matrix_inv_te, mass_pattern_inv_te);
	assembler_te.generate_curl_pattern(curl_matrix_te, curl_pattern_te);
	assembler_te.generate_stabilization_pattern(stabilization_matrix_te, stabilization_patter_te);
  }
}

void GenerateMatrix::assemble_matrices(std::ostream &out) {
  dealii::Timer timer;

  // 3D

  timer.start();
  assembler_3d.assemble_mass_matrix_parallel(mass_matrix_3d, mass_matrix_inv_3d);
  timer.stop();

  out << " Assembling mass matrix and inverse 3D:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";

  timer.reset();

  timer.start();
  assembler_3d.assemble_curl_matrix_parallel(curl_matrix_3d);
  timer.stop();

  out << " Assembling curl matrix 3D:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";

  timer.reset();

  timer.start();
  assembler_3d.assemble_stabilization_matrix_parallel(stabilization_matrix_3d);
  timer.stop();

  out << " Assembling stabilization matrix 3D:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";

  // TE

  timer.start();
  assembler_te.assemble_mass_matrix_parallel(mass_matrix_te, mass_matrix_inv_te);
  timer.stop();

  out << " Assembling mass matrix and inverse TE:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";

  timer.reset();

  timer.start();
  assembler_te.assemble_curl_matrix_parallel(curl_matrix_te);
  timer.stop();

  out << " Assembling curl matrix TE:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";

  timer.reset();

  timer.start();
  assembler_te.assemble_stabilization_matrix_parallel(stabilization_matrix_te);
  timer.stop();

  out << " Assembling stabilization matrix TE:\n\tCPU time:\t"
	  << timer.cpu_time()
	  << " seconds.\n";
  out << "\twall time:\t"
	  << timer.wall_time()
	  << " seconds.\n";
}

void GenerateMatrix::run(unsigned int refinement_level,
						 const std::filesystem::path &path,
						 std::ostream &out) {

  out.precision(4);

  out << "Example Program:\n";
  out << " GenerateMatrix.cpp\n";
  out << '\n';

  assemble_grid(refinement_level);
  setup_system();

  out << "Timing: \n";
  assemble_matrices(out);
  out << '\n';

  out << "Writing: \n";

  // 3D
  {
	const std::filesystem::path out_path = path / "matrices" / "3D";
	std::filesystem::create_directories(out_path);

	// mass matrix
	{
	  const auto out_file_path = out_path / "mass_matrix_3D.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(mass_matrix_3d, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// inv mass matrix
	{
	  const auto out_file_path = out_path / "mass_matrix_inv_3D.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(mass_matrix_inv_3d, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// curl matrix
	{
	  const auto out_file_path = out_path / "curl_matrix_3D.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(curl_matrix_3d, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// stabilization matrix
	{
	  const auto out_file_path = out_path / "stabilization_matrix_3D.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(stabilization_matrix_3d, out_file);
	  out << ' ' << out_file_path << '\n';
	}
  }

  // TE
  {
	const std::filesystem::path out_path = path / "matrices" / "TE";
	std::filesystem::create_directories(out_path);

	// mass matrix
	{
	  const auto out_file_path = out_path / "mass_matrix_TE.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(mass_matrix_te, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// inv mass matrix
	{
	  const auto out_file_path = out_path / "mass_matrix_inv_TE.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(mass_matrix_inv_te, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// curl matrix
	{
	  const auto out_file_path = out_path / "curl_matrix_TE.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(curl_matrix_te, out_file);
	  out << ' ' << out_file_path << '\n';
	}

	// stabilization matrix
	{
	  const auto out_file_path = out_path / "stabilization_matrix_TE.dat";
	  std::ofstream out_file(out_file_path);
	  MaxwellProblem::Tools::output_matrix(stabilization_matrix_te, out_file);
	  out << ' ' << out_file_path << '\n';
	}
  }
}

int main() {

  try {
	GenerateMatrix generate_matrix;
	generate_matrix.run(GLOBAL_REFINEMENT);

  } catch (std::exception &exc) {
	std::cerr << std::endl
			  << std::endl
			  << "----------------------------------------------------"
			  << std::endl;
	std::cerr << "Exception on processing: " << std::endl
			  << exc.what() << std::endl
			  << "Aborting!" << std::endl
			  << "----------------------------------------------------"
			  << std::endl;
	return 1;
  } catch (...) {
	std::cerr << std::endl
			  << std::endl
			  << "----------------------------------------------------"
			  << std::endl;
	std::cerr << "Unknown exception!" << std::endl
			  << "Aborting!" << std::endl
			  << "----------------------------------------------------"
			  << std::endl;
	return 1;
  }
  return 0;
}