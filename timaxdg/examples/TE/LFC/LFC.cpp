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

#include "AssemblerTE.h"
#include "IsotropicConstant.h"
#include "CavitySolution.h"
#include "CrankNicolson.h"
#include "CrankNicolson.hh"
#include "Leapfrog.h"
#include "Leapfrog.hh"
#include "LeapfrogChebychev.h"
#include "LeapfrogChebychev.hh"

// Setup
#define GEOMETRY_X 1
#define GEOMETRY_Y 1

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 3
#define GLOBAL_REFINEMENTS 4

// #define USE_LEAPFROG
// #define USE_CRANK_NICOLSON
#define USE_LEAPFROG_CHEBYCHEV
#define TIME_STEP_WIDTH 300// width is given by 1/TIME_STEP_WIDTH
#define LFC_DEGREE 4
#define ETA 0.1

class LFC_TE {
 private:
  const double a_x = GEOMETRY_X;
  const double a_y = GEOMETRY_Y;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;
  const unsigned int global_refinements = GLOBAL_REFINEMENTS;

  const double time_step_width = 1. / TIME_STEP_WIDTH;
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;

  const unsigned int lfc_degree = LFC_DEGREE;
  const double eta = ETA;

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

  // vectors
  dealii::Vector<double> j_current;
  dealii::BlockVector<double> solution;

  std::vector<std::pair<double, std::string>> times_and_names;

  void output_step();
  double compute_error_L2();

 public:
  LFC_TE(
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void setup_triangulation();
  void setup_system();
  void assemble_system();
  void run();
};

void LFC_TE::output_step() {

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

double LFC_TE::compute_error_L2() {

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

LFC_TE::LFC_TE(
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

void LFC_TE::setup_triangulation() {
  dealii::Point<2> p0(0, 0);
  dealii::Point<2> p1(a_x, a_y);
  GridGenerator::hyper_rectangle(triangulation, p0, p1);
  triangulation.refine_global(global_refinements);
  dealii::GridTools::distort_random(0.15, triangulation);
}

void LFC_TE::setup_system() {

  // dof handling
  dof_handler.distribute_dofs(fe);
  std::vector<unsigned int> block_components = {0, 1, 1};
  dealii::DoFRenumbering::component_wise(dof_handler, block_components);

  // setup matrices
  assembler.generate_mass_pattern(mass, mass_pattern);
  assembler.generate_mass_pattern(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern(curl, curl_pattern);

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

void LFC_TE::assemble_system() {

  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);
}

void LFC_TE::run() {

  out.precision(4);

  out << "Example Program:\n";
  out << " LFC.cpp\n";
  out << '\n';

// setup integrator
#ifdef USE_LEAPFROG
  curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::Leapfrog<SparseMatrix<double>, SparseMatrix<double>>
    integrator(
      inv_mass.block(0, 0),
      inv_mass.block(1, 1),
      curl.block(1, 0),
      curl.block(0, 1),
      time_step_width);
#endif
#ifdef USE_LEAPFROG_CHEBYCHEV
  // curl.block(0, 1).operator*=(-1.0); // since we work with C_E instead of -C_E
  MaxwellProblem::TimeIntegration::LeapfrogChebychev<SparseMatrix<double>, SparseMatrix<double>>
    integrator(
      inv_mass.block(0, 0),
      inv_mass.block(1, 1),
      curl.block(1, 0),
      curl.block(0, 1),
      time_step_width,
      lfc_degree,
      eta);
#endif
#ifdef USE_CRANK_NICOLSON
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
    LFC_TE te_lfc;
    te_lfc.setup_triangulation();
    te_lfc.setup_system();
    te_lfc.assemble_system();
    te_lfc.run();

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
