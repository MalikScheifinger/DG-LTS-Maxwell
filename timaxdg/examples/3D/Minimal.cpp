#include <filesystem>
#include <fstream>
#include <iostream>

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include "Assembler3D.h"
#include "IsotropicConstant.h"
#include "CavitySolution.h"
#include "CrankNicolson.h"
#include "CrankNicolson.hh"
#include "Leapfrog.h"
#include "Leapfrog.hh"

// Setup
#define GEOMETRY_X 1
#define GEOMETRY_Y 1
#define GEOMETRY_Z 1

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 2
#define GLOBAL_REFINEMENTS 3

//#define USE_LEAPFROG
#define USE_CRANK_NICOLSON
#define TIME_STEP_WIDTH 300// width is given by 1/TIME_STEP_WIDTH

#define USE_UPWIND
#define UPWIND_ALPHA 0.5

class Minimal_3D {
 private:
  const double a_x = GEOMETRY_X;
  const double a_y = GEOMETRY_Y;
  const double a_z = GEOMETRY_Z;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;
  const unsigned int global_refinements = GLOBAL_REFINEMENTS;

  const double time_step_width = 1. / TIME_STEP_WIDTH;
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;

  const double upwind_alpha = UPWIND_ALPHA;

  const std::filesystem::path path;
  std::ostream &out;

  // dealii objects
  dealii::Triangulation<3> triangulation;
  dealii::DoFHandler<3> dof_handler;
  dealii::FESystem<3> fe;
  dealii::MappingQ1<3> mapping;
  const dealii::QGauss<3> cell_quadrature;
  const dealii::QGauss<2> face_quadrature;
  MaxwellProblem::Data::IsotropicConstant<3> mu;
  MaxwellProblem::Data::IsotropicConstant<3> eps;
  MaxwellProblem::Assembling::Assembler3D assembler;

  MaxwellProblem::Data::CavitySolution3D cav_solution;

  double current_time = start_time;
  unsigned int current_time_step = 0;

  // matrices and patterns
  dealii::BlockSparsityPattern mass_pattern;
  dealii::BlockSparseMatrix<double> mass;
  dealii::BlockSparsityPattern inv_mass_pattern;
  dealii::BlockSparseMatrix<double> inv_mass;
  dealii::BlockSparsityPattern curl_pattern;
  dealii::BlockSparseMatrix<double> curl;
  dealii::BlockSparseMatrix<double> stab;
  dealii::BlockSparsityPattern stab_pattern;

  // vectors
  dealii::Vector<double> j_current;
  dealii::BlockVector<double> solution;

  std::vector<std::pair<double, std::string>> times_and_names;

  void output_step();
  double compute_error_L2();

 public:
  Minimal_3D(
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void setup_triangulation();
  void setup_system();
  void assemble_system();
  void run();
};

void Minimal_3D::output_step() {

  std::string filename = "solution-" + dealii::Utilities::int_to_string(current_time_step) + ".vtu";

  std::vector<std::string> solution_names = {"H", "H", "H", "E", "E", "E"};
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  interpretation = {
      DataComponentInterpretation::component_is_part_of_vector,
      DataComponentInterpretation::component_is_part_of_vector,
      DataComponentInterpretation::component_is_part_of_vector,
		  DataComponentInterpretation::component_is_part_of_vector,
		  DataComponentInterpretation::component_is_part_of_vector,
		  DataComponentInterpretation::component_is_part_of_vector};

  dealii::DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(dof_handler, solution, solution_names, interpretation);

  auto out_path = path / "output";
  std::filesystem::create_directories(out_path);

  auto out_file = out_path / filename;

  data_out.build_patches(3);
  std::ofstream output(out_file);
  data_out.write_vtu(output);

  times_and_names.push_back(std::pair<double, std::string>(current_time, filename));
  std::ofstream pvd_output("output/solution.pvd");
  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

double Minimal_3D::compute_error_L2() {

  cav_solution.set_time(current_time);

  dealii::Vector<double> local_errors(triangulation.n_active_cells());
  const dealii::QGauss<3> quad(6);

  dealii::VectorTools::integrate_difference(
	  mapping,
	  dof_handler,
	  solution,
	  cav_solution,
	  local_errors,
	  quad,
	  dealii::VectorTools::L2_norm);

  const double L2_error =
	  dealii::VectorTools::compute_global_error(triangulation,
												local_errors,
												dealii::VectorTools::L2_norm);
  return L2_error;
}

Minimal_3D::Minimal_3D(
	const std::filesystem::path &path,
	std::ostream &out) : path(path),
						 out(out),
						 dof_handler(triangulation),
						 fe(dealii::FESystem<3>(FE_DGQ<3>(degree), 3), 1,
							dealii::FESystem<3>(FE_DGQ<3>(degree), 3), 1),
						 cell_quadrature(degree + 2),
						 face_quadrature(degree + 1),
						 mu(MU),
						 eps(EPS),
						 assembler(
							 fe,
							 mapping,
							 cell_quadrature,
							 face_quadrature,
							 dof_handler,
							 mu,
							 eps),
						 cav_solution(MU, EPS, start_time) {}

void Minimal_3D::setup_triangulation() {
  dealii::Point<3> p0(0, 0, 0);
  dealii::Point<3> p1(a_x, a_y, a_z);
  GridGenerator::hyper_rectangle(triangulation, p0, p1);
  triangulation.refine_global(global_refinements);
  dealii::GridTools::distort_random(0.15, triangulation);
}

void Minimal_3D::setup_system() {

  // dof handling
  dof_handler.distribute_dofs(fe);
  std::vector<unsigned int> block_components = {0, 0, 0, 1, 1, 1};
  dealii::DoFRenumbering::component_wise(dof_handler, block_components);

  // setup matrices
  assembler.generate_mass_pattern(mass, mass_pattern);
  assembler.generate_mass_pattern(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern(curl, curl_pattern);
  assembler.generate_stabilization_pattern(stab, stab_pattern);

  // setup vectors
  {
	std::vector<dealii::types::global_dof_index> dofs_per_block =
		DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});

	const auto n_H = dofs_per_block[0];
	const auto n_E = dofs_per_block[1];

	j_current.reinit(n_E);
	solution.reinit(2);
	solution.block(0).reinit(n_H);
	solution.block(1).reinit(n_E);
	solution.collect_sizes();
  }
}

void Minimal_3D::assemble_system() {

  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);
#ifdef USE_UPWIND
  assembler.assemble_stabilization_matrix_parallel(stab, upwind_alpha);
#endif
}

void Minimal_3D::run() {

  out.precision(4);

  out << "Example Program:\n";
  out << " Minimal.cpp\n";
  out << '\n';

// setup integrator
#ifdef USE_LEAPFROG
#ifndef USE_UPWIND
  curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::Leapfrog<SparseMatrix<double>, SparseMatrix<double>>
	  integrator(
		  inv_mass.block(0, 0),
		  inv_mass.block(1, 1),
		  curl.block(1, 0),
		  curl.block(0, 1),
		  time_step_width);
#endif
#ifdef USE_UPWIND
  curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::LeapfrogUpwind<SparseMatrix<double>, SparseMatrix<double>>
	  integrator(
		  inv_mass.block(0, 0),
		  inv_mass.block(1, 1),
		  curl.block(1, 0),
		  curl.block(0, 1),
		  time_step_width,
		  stab.block(0, 0),
		  stab.block(1, 1));
#endif
#endif
#ifdef USE_CRANK_NICOLSON
#ifndef USE_UPWIND
  curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::CrankNicolson<SparseMatrix<double>, SparseMatrix<double>>
	  integrator(
      mass.block(0, 0),
      mass.block(1, 1),
		  inv_mass.block(0, 0),
		  inv_mass.block(1, 1),
		  curl.block(1, 0),
		  curl.block(0, 1),
		  time_step_width);
#endif
#ifdef USE_UPWIND
  curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::CrankNicolsonUpwind<SparseMatrix<double>, SparseMatrix<double>>
	  integrator(
		  inv_mass.block(0, 0),
		  inv_mass.block(1, 1),
		  curl.block(1, 0),
		  curl.block(0, 1),
      stab.block(0, 0),
      stab.block(1, 1),
		  time_step_width);
#endif
#endif

  out << "Time Integration:" << std::endl;

  // initialize H_0, E_0
  dealii::VectorTools::interpolate(
	  dof_handler, cav_solution, solution);
  output_step();

  out
	  << " Time: " << current_time << "\t"
	  << " Time step: " << current_time_step << "\t"
	  << " L2 Error: " << compute_error_L2() << "\t"
	  << " \n"
	  << std::flush;

  while (current_time_step < total_time_steps) {
	current_time += time_step_width;
	current_time_step += 1;

	integrator.integrate_step(solution.block(0), solution.block(1), j_current);

	output_step();

  out
	  << " Time: " << current_time << "\t"
	  << " Time step: " << current_time_step << "\t"
	  << " L2 Error: " << compute_error_L2() << "\t"
	  << " \n"
	  << std::flush;
  }
}

int main() {
  try {
	Minimal_3D te_minimal;
	te_minimal.setup_triangulation();
	te_minimal.setup_system();
	te_minimal.assemble_system();
	te_minimal.run();

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
