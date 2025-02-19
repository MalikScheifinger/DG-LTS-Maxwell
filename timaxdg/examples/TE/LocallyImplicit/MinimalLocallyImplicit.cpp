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
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_matrix_base.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include "AssemblerTE.h"
#include "CavitySolution.h"
#include "LocallyImplicit.h"
#include "LocallyImplicit.hh"
#include "IsotropicConstant.h"
#include "LocallyImplicitReordering.h"
#include "OutputMatrix.h"

// Setup
#define GEOMETRY_X 1
#define GEOMETRY_Y 1

#define GEOMETRY_INNER_X 0.1
#define GEOMETRY_INNER_Y 0.1

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 2
#define GLOBAL_REFINEMENTS 3
#define INNER_REFINEMENTS 8

// #define USE_LEAPFROG
// #define USE_CRANK_NICOLSON
#define TIME_STEP_WIDTH 1500// width is given by 1/TIME_STEP_WIDTH

// #define USE_UPWIND
#define UPWIND_ALPHA 0.5

class MinimalLocallyImplicit {
 private:
  const double a_x = GEOMETRY_X;
  const double a_y = GEOMETRY_Y;
  const double diam_inner_x = GEOMETRY_INNER_X;
  const double diam_inner_y = GEOMETRY_INNER_Y;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;
  const unsigned int global_refinements = GLOBAL_REFINEMENTS;
  const unsigned int inner_refinements = INNER_REFINEMENTS;

  const double time_step_width = 1. / TIME_STEP_WIDTH;
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;

  const double upwind_alpha = UPWIND_ALPHA;

  const std::filesystem::path path;
  std::ostream &out;

  // dealii objects
  dealii::Triangulation<2> triangulation;
  dealii::DoFHandler<2> dof_handler;
  dealii::FESystem<2> fe;
  dealii::MappingQ1<2> mapping;
  const dealii::QGauss<2> cell_quadrature;
  const dealii::QGauss<1> face_quadrature;
  MaxwellProblem::Data::IsotropicConstant<2> mu;
  MaxwellProblem::Data::IsotropicConstant<2> eps;
  MaxwellProblem::Assembling::AssemblerTE assembler;

  MaxwellProblem::Data::CavitySolutionTE cav_solution;

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
  dealii::BlockVector<double> j_current;
  dealii::BlockVector<double> solution;

  std::vector<std::pair<double, std::string>> times_and_names;

  void output_step();
  double compute_error_L2();

 public:
  MinimalLocallyImplicit(
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void setup_triangulation();
  void setup_system();
  void assemble_system();
  void run();
};

void MinimalLocallyImplicit::output_step() {

  std::string filename = "solution-" + dealii::Utilities::int_to_string(current_time_step) + ".vtu";

  std::vector<std::string> solution_names = {"H", "E", "E"};
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
	  interpretation = {
		  DataComponentInterpretation::component_is_scalar,
		  DataComponentInterpretation::component_is_part_of_vector,
		  DataComponentInterpretation::component_is_part_of_vector};

  dealii::DataOut<2> data_out;
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

double MinimalLocallyImplicit::compute_error_L2() {

  cav_solution.set_time(current_time);

  dealii::Vector<double> local_errors(triangulation.n_active_cells());
  const dealii::QGauss<2> quad(6);

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

MinimalLocallyImplicit::MinimalLocallyImplicit(
	const std::filesystem::path &path,
	std::ostream &out) : path(path),
						 out(out),
						 dof_handler(triangulation),
						 fe(dealii::FESystem<2>(FE_DGQ<2>(degree), 1), 1,
							dealii::FESystem<2>(FE_DGQ<2>(degree), 2), 1),
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

void MinimalLocallyImplicit::setup_triangulation() {
  dealii::Point<2> p0(0, 0);
  dealii::Point<2> p1(a_x, a_y);
  GridGenerator::subdivided_hyper_rectangle(triangulation, {3, 3}, p0, p1);

  //for (unsigned int i = 0; i < inner_refinements; i++) {
	//const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);
	////const double diam_x = diam_inner_x - min_cell_diam/4;
	////const double diam_y = diam_inner_y - min_cell_diam/4;
	//const double diam_x = min_cell_diam;
	//const double diam_y = min_cell_diam;
	//const auto x_inner = a_x / 2 - diam_x / 2;
	//const auto y_inner = a_y / 2 - diam_y / 2;
	//for (const auto &cell : triangulation.active_cell_iterators()) {
	//  const auto center = cell->center();
	//  if (x_inner <= center[0]
	//	  && center[0] <= x_inner + diam_x
	//	  && y_inner <= center[1]
	//	  && center[1] <= y_inner + diam_y) {
	//	cell->set_refine_flag();
	//  }
	//}
	//triangulation.prepare_coarsening_and_refinement();
	//triangulation.execute_coarsening_and_refinement();
  //}

	triangulation.refine_global(global_refinements);

	double explicit_diam = dealii::GridTools::minimal_cell_diameter(triangulation);

	for (unsigned int i = 0; i < inner_refinements; i++) {
	const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);

	for (const auto &cell : triangulation.active_cell_iterators()) {
	  const auto center = cell->center();
		double diff_0 = std::abs(center[0] - 0.5);
		double diff_1 = std::abs(center[1] - 0.5);
		double diff = std::sqrt(diff_0*diff_0 + diff_1*diff_1);
	  if (diff < min_cell_diam / std::sqrt(2)) {
		cell->set_refine_flag();
	  }
	}
	triangulation.prepare_coarsening_and_refinement();
	triangulation.execute_coarsening_and_refinement();
  }

  dealii::GridTools::distort_random(0.1, triangulation);

	const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);
	const auto max_cell_diam = dealii::GridTools::maximal_cell_diameter(triangulation);
	const double CFL = 12 * time_step_width;
  MaxwellProblem::LocallyImplicit::set_cells_explicit_implicit(triangulation, explicit_diam/5);

  std::ofstream out("grid.vtk");
  dealii::GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
}

void MinimalLocallyImplicit::setup_system() {

  // dof handling
  dof_handler.distribute_dofs(fe);
  MaxwellProblem::LocallyImplicit::renumbering_dofs_explicit_to_implicit(dof_handler);
  std::vector<unsigned int> block_components = {0, 1, 1};
  dealii::DoFRenumbering::component_wise(dof_handler, block_components);

  // setup matrices
  assembler.generate_mass_pattern_locally_implicit(mass, mass_pattern);
  assembler.generate_mass_pattern_locally_implicit(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern_locally_implicit(curl, curl_pattern);
	// ToDo, stab pattern!
  assembler.generate_stabilization_pattern(stab, stab_pattern);

  // setup vectors
  {
	std::vector<dealii::types::global_dof_index> imp_exp_dofs(8,0);
	MaxwellProblem::LocallyImplicit::get_number_dofs_exp_imp(dof_handler, imp_exp_dofs);

	solution.reinit(imp_exp_dofs);
	// remove the dofs corresponding to H
	imp_exp_dofs.erase(imp_exp_dofs.begin(), imp_exp_dofs.begin() + 4);
	j_current.reinit(imp_exp_dofs);
  }
}

void MinimalLocallyImplicit::assemble_system() {

  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
	// ToDo remove!
	//{
	//	std::ofstream out_file("mass_matrix.dat");
	//  MaxwellProblem::Tools::output_matrix(mass, out_file);
	//}
  assembler.assemble_curl_matrix_parallel(curl);
	// ToDo remove!
	//{
	//	std::ofstream out_file("curl_matrix.dat");
	//  MaxwellProblem::Tools::output_matrix(curl, out_file);
	//}
#ifdef USE_UPWIND
  assembler.assemble_stabilization_matrix_parallel(stab, upwind_alpha);
#endif
}

void MinimalLocallyImplicit::run() {

  out.precision(4);

  out << "Example Program:\n";
  out << " Minimal.cpp\n";
  out << '\n';

	// setup integrator
	MaxwellProblem::TimeIntegration::LocallyImplicit<dealii::BlockSparseMatrix<double>,
	dealii::BlockSparseMatrix<double>> integrator(
		mass,
		inv_mass,
		curl,
		time_step_width
	);

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

	integrator.integrate_step(solution, j_current);

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
	MinimalLocallyImplicit te_minimal;
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
