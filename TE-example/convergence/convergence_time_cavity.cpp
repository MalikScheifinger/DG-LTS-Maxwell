#include <filesystem>
#include <fstream>
#include <iostream>

#include <iterator>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/timer.h>

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
#include "SolutionTE.h"
#include "IsotropicConstant.h"
#include "CavitySolution.h"
#include "LocallyImplicitReordering.h"
#include "LTS_LFC.h"
#include "LTS_LFC.hh"

#include "NumberSpaces.h"
#include "OutputMatrix.h"

// Setup
#define GEOMETRY_X 1
#define GEOMETRY_Y 1

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 5
#define GLOBAL_REFINEMENTS 3
#define INNER_REFINEMENTS 3 // must be at least 1 for LI/LTS
#define GLOBAL_LOCAL_THRESHOLD 1.2

#define ETA 1.0

#define START_TIME_STEP_WIDTH 0.0025
#define END_TIME_STEP_WIDTH 0.0001
#define NUMBER_OF_RUNS 40

#define OUTPUT_TABLE
#define OUTPUT_MESH
// #define OUTPUT_MATRICES

#define CFLS

class convergence_cavity_TE {
 private:
  const double a_x = GEOMETRY_X;
  const double a_y = GEOMETRY_Y;

  const double global_local_threshold = GLOBAL_LOCAL_THRESHOLD;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;
  
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

  SolutionTE ex_solution;
  RhsTE rhs;

  // matrices and patterns
  dealii::BlockSparsityPattern mass_pattern;
  dealii::BlockSparseMatrix<double> mass;
  dealii::BlockSparsityPattern inv_mass_pattern;
  dealii::BlockSparseMatrix<double> inv_mass;
  dealii::BlockSparsityPattern curl_pattern;
  dealii::BlockSparseMatrix<double> curl;

  // vectors
  dealii::BlockVector<double> j_current;
  dealii::BlockVector<double> solution;

  std::vector<std::pair<double, std::string>> times_and_names;

  dealii::TimerOutput computing_timer;

  void output_step(const unsigned int current_time_step, const double current_time);
  double compute_error_L2(const double current_time);

 public:
  convergence_cavity_TE(
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void setup_triangulation(const unsigned int global_refinements, const unsigned int inner_refinements);
  void setup_system();
  void assemble_system();
  void output_matrices(const unsigned int global_ref, const unsigned int local_ref);
  double time_integration(const double time_step_width, const unsigned int lfc_degree);
  void run();
};

void convergence_cavity_TE::output_step(const unsigned int current_time_step, const double current_time) {

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

void convergence_cavity_TE::output_matrices(const unsigned int global_ref, const unsigned int local_ref) {

  TimerOutput::Scope timer_section(computing_timer, "Output matrices");

  out << "Writing matrices: \n";
  const std::filesystem::path out_path = path / "matrices";
  std::filesystem::create_directories(out_path);

  std::stringstream ss;
  ss << "_dg" << FE_DEGREE << "_globalref" << global_ref << "_localref" << local_ref << ".dat";
  std::string data_str = ss.str();

  // mass matrix
  {
    auto out_file_path = out_path / "mass_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(mass, out_file);
    out << "  " << out_file_path << "\n";
  }

  // inv mass matrix
  {
    auto out_file_path = out_path / "inv_mass_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(inv_mass, out_file);
    out << "  " << out_file_path << "\n";
  }

  // curl matrix
  {
    auto out_file_path = out_path / "curl_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(curl, out_file);
    out << "  " << out_file_path << "\n";
  }

  // block sizes
  {
    auto out_file_path = out_path / "block_sizes";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    for (unsigned int i = 0; i < solution.n_blocks(); ++i) {
      out_file << solution.block(i).size() << " ";
    }
  }
  out << "\n";
}

double convergence_cavity_TE::compute_error_L2(const double current_time) {
  TimerOutput::Scope timer_section(computing_timer, "L2-error");

  ex_solution.set_time(current_time);

  dealii::Vector<double> local_errors(triangulation.n_active_cells());
  const dealii::QGauss<2> quad(6);

  dealii::VectorTools::integrate_difference(
	  mapping,
	  dof_handler,
	  solution,
	  ex_solution,
	  local_errors,
	  quad,
	  dealii::VectorTools::L2_norm);

  const double L2_error =
	  dealii::VectorTools::compute_global_error(triangulation,
												local_errors,
												dealii::VectorTools::L2_norm);
  return L2_error;
}

convergence_cavity_TE::convergence_cavity_TE(
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
						 ex_solution(MU, EPS, start_time),
             rhs(MU, EPS, start_time),
             computing_timer(std::cout, TimerOutput::never, TimerOutput::cpu_and_wall_times_grouped) {}

void convergence_cavity_TE::setup_triangulation(const unsigned int global_refinements, const unsigned int inner_refinements) {
  TimerOutput::Scope timer_section(computing_timer, "Setup triangulation");

  dealii::Point<2> p0(0, 0);
  dealii::Point<2> p1(a_x, a_y);
  GridGenerator::subdivided_hyper_rectangle(triangulation, {3, 3}, p0, p1);
  triangulation.refine_global(global_refinements);

  double explicit_diam = dealii::GridTools::minimal_cell_diameter(triangulation);

	for (unsigned int i = 0; i < inner_refinements; i++) {
    const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);

    for (const auto &cell : triangulation.active_cell_iterators()) {
      const auto center = cell->center();
      double diff_0 = std::abs(center[0] - 0.5 * a_x);
      double diff_1 = std::abs(center[1] - 0.5 * a_y);
      double diff = std::sqrt(diff_0*diff_0 + diff_1*diff_1);
      if (diff < min_cell_diam / std::sqrt(2)) {
        cell->set_refine_flag();
      }
    }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }
  dealii::GridTools::distort_random(0.1, triangulation);

	MaxwellProblem::LocallyImplicit::set_cells_explicit_implicit(triangulation, explicit_diam/global_local_threshold);

  #ifdef OUTPUT_MESH
  std::stringstream ss;
  ss << "grid_globalref" << global_refinements << "_localref" << inner_refinements << ".vtk"; 
  std::string out_str = ss.str();

  std::ofstream out(out_str);
  dealii::GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
  #endif
}

void convergence_cavity_TE::setup_system() {

  TimerOutput::Scope timer_section(computing_timer, "Setup system");

  // dof handling
  dof_handler.distribute_dofs(fe);

  MaxwellProblem::LocallyImplicit::renumbering_dofs_explicit_to_implicit(dof_handler);

  std::vector<unsigned int> block_components = {0, 1, 1};
  dealii::DoFRenumbering::component_wise(dof_handler, block_components);

  // setup matrices
  assembler.generate_mass_pattern_locally_implicit(mass, mass_pattern);
  assembler.generate_mass_pattern_locally_implicit(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern_locally_implicit(curl, curl_pattern);

  const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);
	const auto max_cell_diam = dealii::GridTools::maximal_cell_diameter(triangulation);
    
  out.precision(4);
  out << "Setting up system:\n";
  out << "  max cell diam: " << max_cell_diam << "\n";
  out << "  min cell diam: " << min_cell_diam << "\n";
  out << "  DoFs: " << dof_handler.n_dofs() << "\n";

  // setup vectors
  {
    std::vector<dealii::types::global_dof_index> imp_exp_dofs(8,0);
    MaxwellProblem::LocallyImplicit::get_number_dofs_exp_imp(dof_handler, imp_exp_dofs);

    solution.reinit(imp_exp_dofs);
    j_current.reinit(imp_exp_dofs);

    out << "  leapfrog DoFs: " << solution.block(0).size() + solution.block(4).size() << "\n";
    out << "  leapfrog with modified neighbor DoFs: " << solution.block(1).size() + solution.block(5).size() << "\n";
    out << "  modified coarse DoFs: " << solution.block(2).size() + solution.block(6).size() << "\n";
    out << "  modified fine DoFs: " << solution.block(3).size() + solution.block(7).size() << "\n\n";
  }
}

void convergence_cavity_TE::assemble_system() {

  TimerOutput::Scope timer_section(computing_timer, "Assemble system");

  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);
}

double convergence_cavity_TE::time_integration(const double time_step_width, const unsigned int lfc_degree) {
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;
  double current_time = start_time;
  unsigned int current_time_step = 0;

  out.precision(4);
  out << "Time Integration:" << std::endl;
  out << "  with time step size: " << time_step_width << "\n";
  out << "  total time steps: " << total_time_steps << "\n";
  out << '\n';

  // setup integrator
  MaxwellProblem::TimeIntegration::LTS_LFC<BlockSparseMatrix<double>, BlockSparseMatrix<double>>
    integrator(
      mass,
      inv_mass,
      curl,
      time_step_width,
      lfc_degree,
      eta);

  // initialize H_0, E_0
  ex_solution.set_time(start_time);

  // L2-projected IV
  {
    TimerOutput::Scope timer_section(computing_timer, "L2-project IV");
    dealii::BlockVector<double> tmp_iv_solution(solution);
    assembler.assemble_rhs(ex_solution, tmp_iv_solution);
    inv_mass.vmult(solution, tmp_iv_solution);
  }

  std::vector<double> L2_errors;
  L2_errors.push_back(compute_error_L2(current_time));

  while (current_time_step < total_time_steps) {
    current_time += time_step_width;
    current_time_step += 1;

    {
      TimerOutput::Scope timer_section(computing_timer, "Assemble RHS");
      rhs.set_time(current_time - time_step_width / 2);
      assembler.assemble_rhs(rhs, j_current);
    }

    {
      TimerOutput::Scope timer_section(computing_timer, "Time integration");
      j_current *= -1.;
      integrator.integrate_step(solution, j_current);
    }
    double L2_error = 0.0;

    try {
      L2_error = compute_error_L2(current_time);
    } catch (dealii::ExcNumberNotFinite const&) {
      L2_error = 100;
      out << "  Unstable at time t=" << current_time << "\n\n";
    }
    
    L2_errors.push_back(L2_error);
  }
  double max_L2_error = *std::max_element(L2_errors.begin(), L2_errors.end());

  // cut-off unstable values
  if (max_L2_error > 100) max_L2_error = 100;

  out << "  Max L2 Error: " << max_L2_error << "\n";
  out << "\n";

  return max_L2_error;
}

void convergence_cavity_TE::run() {
    computing_timer.reset();

    dealii::ConvergenceTable conv_table;
    
    setup_triangulation(GLOBAL_REFINEMENTS, INNER_REFINEMENTS);
    setup_system();
    assemble_system();

    #ifdef OUTPUT_MATRICES
    output_matrices(GLOBAL_REFINEMENTS, INNER_REFINEMENTS);
    #endif

    std::vector<unsigned int> lfc_degrees = {1, 2, 4, 8, 9};

    auto time_step_sizes = MaxwellProblem::Tools::log2_spaced(
      START_TIME_STEP_WIDTH, 
      END_TIME_STEP_WIDTH, 
      NUMBER_OF_RUNS);

    #ifdef CFLS
    // some additional time stepsizes to hit the CFLs
    std::vector<double> lf_lfc_cfl_time_step_sizes;

    if (ETA == 1.0) {
      lf_lfc_cfl_time_step_sizes = {0.000471, 0.000933, 0.001862, 0.002094, 0.002018, 0.000266};
    } else {
      lf_lfc_cfl_time_step_sizes = {0.002018, 0.000266};
    }

    time_step_sizes.insert(time_step_sizes.begin(), lf_lfc_cfl_time_step_sizes.begin(), lf_lfc_cfl_time_step_sizes.end());

    std::sort(time_step_sizes.begin(), time_step_sizes.end(), std::greater<double>());
    #endif
    
    for (unsigned int lfc_degree : lfc_degrees) {
      out << "LTS-LFC with degree p = " << lfc_degree << "\n\n";
      for (const auto &step_size : time_step_sizes) {
        conv_table.add_value("lfc degree", lfc_degree);
        conv_table.add_value("time step width", step_size);
        conv_table.add_value("inverse time step width", 1. / step_size);
        conv_table.add_value("max l2 error", time_integration(step_size, lfc_degree));
      }
    }

    conv_table.set_precision("time step width", 8);
    conv_table.set_precision("inverse time step width", 8);
    conv_table.set_precision("max l2 error", 8);

    conv_table.evaluate_convergence_rates("max l2 error", "inverse time step width",dealii::ConvergenceTable::RateMode::reduction_rate_log2, 1);
    conv_table.write_text(std::cout, dealii::ConvergenceTable::TextOutputFormat::org_mode_table);

    #ifdef OUTPUT_TABLE
    std::stringstream ss;
    ss << "errors_time_cavity_LTS_LFC_eta" << ETA
       << "_dg" << FE_DEGREE << "_globalref" << GLOBAL_REFINEMENTS
       << "_localref" << INNER_REFINEMENTS << "_threshold" << GLOBAL_LOCAL_THRESHOLD;

    std::string out_str = ss.str();
    std::replace(out_str.begin(), out_str.end(), '.', ',');

    std::ofstream out_file(out_str + ".txt");

    conv_table.write_text(out_file, dealii::ConvergenceTable::TextOutputFormat::org_mode_table);
    out_file.close();
    #endif

    computing_timer.print_summary();
    std::cout << std::endl;
}

int main() {
  try {
    convergence_cavity_TE te_lfc;
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
