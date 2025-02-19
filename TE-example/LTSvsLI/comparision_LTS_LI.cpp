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
#include "Leapfrog.h"
#include "Leapfrog.hh"
#include "LTS_LFC.h"
#include "LTS_LFC.hh"
#include "LocallyImplicit.h"
#include "LocallyImplicit.hh"

#include "NumberSpaces.h"
#include "OutputMatrix.h"

// Setup
#define GEOMETRY_X 4
#define GEOMETRY_Y 4

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 2
#define GLOBAL_REFINEMENTS 7
#define INNER_REFINEMENTS 2 // must be at least 1 for LI/LTS
#define GLOBAL_LOCAL_THRESHOLD 1.0

#define LFC_DEGREE 4
#define ETA 0.5

#define TIME_STEP_WIDTH_LEAPFROG 0.00068
#define TIME_STEP_WIDTH_LTS_LI 0.0022

// #define OUTPUT_MATRICES

class comp_LTS_LI_TE {
 private:
  const double a_x = GEOMETRY_X;
  const double a_y = GEOMETRY_Y;

  const double global_local_threshold = GLOBAL_LOCAL_THRESHOLD;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;
  
  const double eta = ETA;

  const std::string integrator_str;
  const double radius;

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

  double compute_error_L2(const double current_time);
  void setup_triangulation(const unsigned int global_refinements, const unsigned int inner_refinements);
  void setup_system();
  void assemble_system();
  void output_matrices(const unsigned int global_ref, const unsigned int local_ref);
  void ti_leapfrog(const double time_step_width);
  void ti_lts(const double time_step_width);
  void ti_li(const double time_step_width);

 public:
  comp_LTS_LI_TE(
    const std::string &integrator_str,
    const double radius,
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void run();
};

void comp_LTS_LI_TE::output_matrices(const unsigned int global_ref, const unsigned int local_ref) {

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
}

double comp_LTS_LI_TE::compute_error_L2(const double current_time) {
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

comp_LTS_LI_TE::comp_LTS_LI_TE(
  const std::string &integrator_str,
  const double radius,
	const std::filesystem::path &path,
	std::ostream &out) : 
              integrator_str(integrator_str),
              radius(radius),
              path(path),
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

void comp_LTS_LI_TE::setup_triangulation(const unsigned int global_refinements, const unsigned int inner_refinements) {
  TimerOutput::Scope timer_section(computing_timer, "Setup triangulation");

  dealii::Point<2> p0(0, 0);
  dealii::Point<2> p1(a_x, a_y);
  GridGenerator::subdivided_hyper_rectangle(triangulation, {2, 2}, p0, p1);

  triangulation.refine_global(global_refinements);

  double explicit_diam = dealii::GridTools::minimal_cell_diameter(triangulation);

	for (unsigned int i = 0; i < inner_refinements; i++) {
    for (const auto &cell : triangulation.active_cell_iterators()) {
      const auto center = cell->center();
      if (center.norm() < radius) {
        cell->set_refine_flag();
      }
    }
    triangulation.prepare_coarsening_and_refinement();
    triangulation.execute_coarsening_and_refinement();
  }

	MaxwellProblem::LocallyImplicit::set_cells_explicit_implicit(triangulation, explicit_diam/global_local_threshold);

  std::stringstream ss;
  ss << "grid_globalref" << global_refinements << "_localref" << inner_refinements << ".vtk"; 
  std::string out_str = ss.str();

  std::ofstream out(out_str);
  dealii::GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
}

void comp_LTS_LI_TE::setup_system() {

  TimerOutput::Scope timer_section(computing_timer, "Setup system");

  // dof handling
  dof_handler.distribute_dofs(fe);

  MaxwellProblem::LocallyImplicit::renumbering_dofs_explicit_to_implicit(dof_handler);

  std::vector<unsigned int> block_components = {0, 1, 1};
  dealii::DoFRenumbering::component_wise(dof_handler, block_components);

  // setup matrices
  if(integrator_str == "lts_lfc" || integrator_str == "li") {
    assembler.generate_mass_pattern_locally_implicit(mass, mass_pattern);
    assembler.generate_mass_pattern_locally_implicit(inv_mass, inv_mass_pattern);
    assembler.generate_curl_pattern_locally_implicit(curl, curl_pattern);
  } else {
    assembler.generate_mass_pattern(mass, mass_pattern);
    assembler.generate_mass_pattern(inv_mass, inv_mass_pattern);
    assembler.generate_curl_pattern(curl, curl_pattern);
  }

  const auto min_cell_diam = dealii::GridTools::minimal_cell_diameter(triangulation);
	const auto max_cell_diam = dealii::GridTools::maximal_cell_diameter(triangulation);
    
  out.precision(4);
  out << "Setting up system:\n";
  out << "  max cell diam: " << max_cell_diam << "\n";
  out << "  min cell diam: " << min_cell_diam << "\n";
  out << "  DoFs: " << dof_handler.n_dofs() << "\n";

  // setup vectors
  if(integrator_str == "lts_lfc" || integrator_str == "li") {
    std::vector<dealii::types::global_dof_index> imp_exp_dofs(8,0);
    MaxwellProblem::LocallyImplicit::get_number_dofs_exp_imp(dof_handler, imp_exp_dofs);

    solution.reinit(imp_exp_dofs);
    j_current.reinit(imp_exp_dofs);

    out << "  leapfrog DoFs: " << solution.block(0).size() + solution.block(4).size() << "\n";
    out << "  leapfrog with modified neighbor DoFs: " << solution.block(1).size() + solution.block(5).size() << "\n";
    out << "  modified coarse DoFs: " << solution.block(2).size() + solution.block(6).size() << "\n";
    out << "  modified fine DoFs: " << solution.block(3).size() + solution.block(7).size() << "\n\n";
  } else {
    std::vector<dealii::types::global_dof_index> dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});

    j_current.reinit(dofs_per_block);
    solution.reinit(dofs_per_block);
  }
}

void comp_LTS_LI_TE::assemble_system() {

  TimerOutput::Scope timer_section(computing_timer, "Assemble system");

  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);

  if(integrator_str == "leapfrog") curl.block(0, 1).operator*=(-1.0);
}

void comp_LTS_LI_TE::ti_leapfrog(const double time_step_width) {
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;
  double current_time = start_time;
  unsigned int current_time_step = 0;

  out.precision(4);
  out << "Time Integration:" << std::endl;
  out << "  with time step size: " << time_step_width << "\n";
  out << "  total time steps: " << total_time_steps << "\n";
  out << '\n';

  // setup integrator
    MaxwellProblem::TimeIntegration::Leapfrog<SparseMatrix<double>, SparseMatrix<double>>
      integrator(
        inv_mass.block(0, 0),
        inv_mass.block(1, 1),
        curl.block(1, 0),
        curl.block(0, 1),
        time_step_width);

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
      TimerOutput::Scope timer_section(computing_timer, "Time integration LF");
      integrator.integrate_step(solution.block(0), solution.block(1), j_current.block(1));
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
}

void comp_LTS_LI_TE::ti_lts(const double time_step_width) {
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
      LFC_DEGREE,
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
      j_current *= -1.;
    }

    {
      TimerOutput::Scope timer_section(computing_timer, "Time integration LTS");
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
}

void comp_LTS_LI_TE::ti_li(const double time_step_width) {
  const unsigned int total_time_steps = (end_time - start_time) / time_step_width;
  double current_time = start_time;
  unsigned int current_time_step = 0;

  out.precision(4);
  out << "Time Integration:" << std::endl;
  out << "  with time step size: " << time_step_width << "\n";
  out << "  total time steps: " << total_time_steps << "\n";
  out << '\n';

  // setup integrator
  MaxwellProblem::TimeIntegration::LocallyImplicit<BlockSparseMatrix<double>, BlockSparseMatrix<double>>
    integrator(
      mass,
      inv_mass,
      curl,
      time_step_width,
      1e-6,
      1000);

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

  dealii::BlockVector<double> j_current_E; 
  j_current_E.reinit(
    {solution.block(4).size(), 
    solution.block(5).size(), 
    solution.block(6).size(), 
    solution.block(7).size()});

  while (current_time_step < total_time_steps) {
    current_time += time_step_width;
    current_time_step += 1;

    {
      TimerOutput::Scope timer_section(computing_timer, "Assemble RHS");
      rhs.set_time(current_time - time_step_width / 2);
      assembler.assemble_rhs(rhs, j_current);
      j_current_E.block(0).swap(j_current.block(4));
      j_current_E.block(1).swap(j_current.block(5));
      j_current_E.block(2).swap(j_current.block(6));
      j_current_E.block(3).swap(j_current.block(7));
    }

    {
      TimerOutput::Scope timer_section(computing_timer, "Time integration LI");
      integrator.integrate_step(solution, j_current_E);
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
}

void comp_LTS_LI_TE::run() {
    computing_timer.reset();

    out << "Integrator: " << integrator_str << "\n\n";
    
    setup_triangulation(GLOBAL_REFINEMENTS, INNER_REFINEMENTS);
    setup_system();
    assemble_system();

    #ifdef OUTPUT_MATRICES
    output_matrices(GLOBAL_REFINEMENTS, INNER_REFINEMENTS);
    #endif
    
    if(integrator_str == "leapfrog") {
      ti_leapfrog(TIME_STEP_WIDTH_LEAPFROG);
    } else if(integrator_str == "lts_lfc") {
      ti_lts(TIME_STEP_WIDTH_LTS_LI);
    } else if(integrator_str == "li") {
      ti_li(TIME_STEP_WIDTH_LTS_LI);
    } else {
      throw dealii::ExcMessage("No valid time-integrator. Choose 'leapfrog', 'lts_lfc' or 'li'.");
    }

    computing_timer.print_summary();
    std::cout << std::endl;
}

int main() {
  try {
    {
    comp_LTS_LI_TE te_leapfrog("leapfrog", 0.5);
    te_leapfrog.run();
    comp_LTS_LI_TE te_lts_lfc("lts_lfc", 0.5);
    te_lts_lfc.run();
    comp_LTS_LI_TE te_li("li", 0.5);
    te_li.run();
    }

    {
    comp_LTS_LI_TE te_leapfrog("leapfrog", 0.1);
    te_leapfrog.run();
    comp_LTS_LI_TE te_lts_lfc("lts_lfc", 0.1);
    te_lts_lfc.run();
    comp_LTS_LI_TE te_li("li", 0.1);
    te_li.run();
    }

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
