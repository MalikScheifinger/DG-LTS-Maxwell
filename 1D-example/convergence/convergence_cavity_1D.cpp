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

#include "Assembler1D.h"
#include "IsotropicConstant.h"
#include "Solutions1D.h"
#include "LocallyImplicitReordering.h"
#include "LTS_LFC.h"
#include "LTS_LFC.hh"

#include "NumberSpaces.h"
#include "OutputMatrix.h"

// Setup
#define GEOMETRY_X 1

#define START_TIME 0.0
#define END_TIME 1.0

#define MU 1.0
#define EPS 1.0

#define FE_DEGREE 2
#define COARSE_CELLS 100
#define FINE_MESH_FACTOR 4

#define START_TIME_STEP_WIDTH 0.003
#define END_TIME_STEP_WIDTH 0.001
#define NUMBER_OF_RUNS 140

#define OUTPUT_TABLE
// #define OUTPUT_MESH
// #define OUTPUT_MATRICES

class convergence_cavity_1D {
 private:
  const double a_x = GEOMETRY_X;

  const double start_time = START_TIME;
  const double end_time = END_TIME;

  const unsigned int degree = FE_DEGREE;

  const std::filesystem::path path;
  std::ostream &out;

  // dealii objects
  dealii::Triangulation<1> triangulation;
  dealii::DoFHandler<1> dof_handler;
  dealii::FESystem<1> fe;
  dealii::MappingQ1<1> mapping;
  const dealii::QGauss<1> cell_quadrature;
  const dealii::QGauss<0> face_quadrature;

  MaxwellProblem::Data::IsotropicConstant<1> mu;
  MaxwellProblem::Data::IsotropicConstant<1> eps;
  MaxwellProblem1D::Assembling::Assembler1D assembler;
  MaxwellProblem1D::Data::CavitySolution1D ex_solution;

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
  void setup_triangulation();
  void setup_system();
  void assemble_system();
  void output_matrices();
  double time_integration(const double time_step_width, const unsigned int lfc_degree, const double eta);

 public:
  convergence_cavity_1D(
	  const std::filesystem::path &path = std::filesystem::current_path(),
	  std::ostream &out = std::cout);
  void run(const double eta);
};

void convergence_cavity_1D::output_matrices() {

  TimerOutput::Scope timer_section(computing_timer, "Output matrices");

  out << "Writing matrices: \n";
  const std::filesystem::path out_path = path / "matrices";
  std::filesystem::create_directories(out_path);

  std::stringstream ss;
  ss << "_dg" << FE_DEGREE << "_coarse-fine-ratio" << FINE_MESH_FACTOR << ".dat";
  std::string data_str = ss.str();

  // mass matrix
  {
    auto out_file_path = out_path / "mass_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(mass, out_file);
    out << "\t" << out_file_path << '\n';
  }

  // inv mass matrix
  {
    auto out_file_path = out_path / "inv_mass_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(inv_mass, out_file);
    out << "\t" << out_file_path << '\n';
  }

  // curl matrix
  {
    auto out_file_path = out_path / "curl_matrix";
    out_file_path += data_str;
    std::ofstream out_file(out_file_path);
    MaxwellProblem::Tools::output_matrix(curl, out_file);
    out << "\t" << out_file_path << '\n';
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

double convergence_cavity_1D::compute_error_L2(const double current_time) {
  TimerOutput::Scope timer_section(computing_timer, "L2-error");

  ex_solution.set_time(current_time);

  dealii::Vector<double> local_errors(triangulation.n_active_cells());
  const dealii::QGauss<1> quad(4);

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

convergence_cavity_1D::convergence_cavity_1D(
	const std::filesystem::path &path,
	std::ostream &out) : path(path),
						 out(out),
						 dof_handler(triangulation),
						 fe(dealii::FESystem<1>(FE_DGQ<1>(degree), 1), 1,
							dealii::FESystem<1>(FE_DGQ<1>(degree), 1), 1),
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
             computing_timer(std::cout, TimerOutput::never, TimerOutput::cpu_and_wall_times_grouped) {}

void convergence_cavity_1D::setup_triangulation() {
  TimerOutput::Scope timer_section(computing_timer, "Setup triangulation");

  {
    const unsigned int num_cells = COARSE_CELLS;
    const double coarse_celldiam = (a_x - (a_x / num_cells) / FINE_MESH_FACTOR) / num_cells;

    std::vector<dealii::Point<1>> vertices;
    vertices.reserve(num_cells + 2);

    double value = 0.0;
    vertices.emplace_back(dealii::Point<1>{value});
    for (unsigned int i = 0; i < num_cells; ++i) {
      if (i == (unsigned int)std::floor(num_cells / 2)) {
        value += (a_x / num_cells) / FINE_MESH_FACTOR;
      } else {
        value += coarse_celldiam;
      }
      vertices.emplace_back(dealii::Point<1>{value});
    }
    vertices.emplace_back(dealii::Point<1>{a_x});

    std::vector<std::array<unsigned int, GeometryInfo<1>::vertices_per_cell>> cell_vertices;
    cell_vertices.reserve(vertices.size() - 1);

    for (unsigned int i = 0; i < vertices.size() - 1; ++i) {
      cell_vertices.push_back({{i, i + 1}});
    }
    
    const unsigned int n_cells = cell_vertices.size();

    std::vector<CellData<1>> cells(n_cells, CellData<1>());
    for (unsigned int i = 0; i < n_cells; ++i) {
      for (unsigned int j = 0; j < cell_vertices[i].size(); ++j) {
        cells[i].vertices[j] = cell_vertices[i][j];
      }
      cells[i].material_id = 0;
    }
    GridTools::consistently_order_cells(cells);
    triangulation.create_triangulation(vertices, cells, SubCellData());
  }

  double explicit_diam = dealii::GridTools::maximal_cell_diameter(triangulation);

	MaxwellProblem::LocallyImplicit::set_cells_explicit_implicit(triangulation, explicit_diam/2);

  #ifdef OUTPUT_MESH
  std::stringstream ss;
  ss << "grid.vtk"; 
  std::string out_str = ss.str();

  std::ofstream out(out_str);
  dealii::GridOut grid_out;
  grid_out.write_vtk(triangulation, out);
  #endif
}

void convergence_cavity_1D::setup_system() {

  TimerOutput::Scope timer_section(computing_timer, "Setup system");

  // dof handling
  dof_handler.distribute_dofs(fe);

  MaxwellProblem::LocallyImplicit::renumbering_dofs_explicit_to_implicit(dof_handler);

  std::vector<unsigned int> block_components = {0, 1};
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

    for (const auto &cell : dof_handler.active_cell_iterators()) {
      imp_exp_dofs[cell->material_id()]++;
      imp_exp_dofs[4 + cell->material_id()]++;
    }

    std::cout << "\n  Number of \n"
        "    implicit fine cells: "
    << imp_exp_dofs[3] << "\n"
        "    implicit coarse cells: "
    << imp_exp_dofs[2] << "\n"
        "    explicit cells with implicit neighbor cells: "
    << imp_exp_dofs[1] << "\n"
        "    explicit cells without implicit neighbor cells: "
    << imp_exp_dofs[0] << std::endl
    << std::endl;

    for(int i = 0; i<4; i++) {
      imp_exp_dofs[i] *= dof_handler.get_fe().get_sub_fe(0,1).dofs_per_cell;
      imp_exp_dofs[4+i] *= dof_handler.get_fe().get_sub_fe(1,1).dofs_per_cell;
    }

    solution.reinit(imp_exp_dofs);
    j_current.reinit(imp_exp_dofs);

    out << "  leapfrog DoFs: " << solution.block(0).size() + solution.block(4).size() << "\n";
    out << "  leapfrog with modified neighbor DoFs: " << solution.block(1).size() + solution.block(5).size() << "\n";
    out << "  modified coarse DoFs: " << solution.block(2).size() + solution.block(6).size() << "\n";
    out << "  modified fine DoFs: " << solution.block(3).size() + solution.block(7).size() << "\n\n";
  }
}

void convergence_cavity_1D::assemble_system() {

  TimerOutput::Scope timer_section(computing_timer, "Assemble system");

  assembler.assemble_mass_matrix(mass, inv_mass);
  assembler.assemble_curl_matrix(curl);
}

double convergence_cavity_1D::time_integration(const double time_step_width, const unsigned int lfc_degree, const double eta) {
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

  dealii::VectorTools::interpolate(
    dof_handler, ex_solution, solution);

  std::vector<double> L2_errors;
  L2_errors.push_back(compute_error_L2(current_time));

  while (current_time_step < total_time_steps) {
    current_time += time_step_width;
    current_time_step += 1;

    {
      TimerOutput::Scope timer_section(computing_timer, "Time integration");
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

void convergence_cavity_1D::run(const double eta) {
    computing_timer.reset();

    out << "LFC-LTS with eta= " << eta << "\n\n";

    dealii::ConvergenceTable conv_table;
    
    setup_triangulation();
    setup_system();
    assemble_system();

    #ifdef OUTPUT_MATRICES
    output_matrices();
    #endif

    std::vector<unsigned int> lfc_degrees = {3, 4, 5};

    auto time_step_sizes = MaxwellProblem::Tools::log2_spaced(
      START_TIME_STEP_WIDTH, 
      END_TIME_STEP_WIDTH, 
      NUMBER_OF_RUNS);
    
    for (unsigned int lfc_degree : lfc_degrees) {
      for (const auto &step_size : time_step_sizes) {
        conv_table.add_value("lfc degree", lfc_degree);
        conv_table.add_value("time step width", step_size);
        conv_table.add_value("inverse time step width", 1. / step_size);
        conv_table.add_value("max l2 error", time_integration(step_size, lfc_degree, eta));
      }
    }

    conv_table.set_precision("time step width", 8);
    conv_table.set_precision("inverse time step width", 8);
    conv_table.set_precision("max l2 error", 8);

    conv_table.evaluate_convergence_rates("max l2 error", "inverse time step width",dealii::ConvergenceTable::RateMode::reduction_rate_log2, 1);
    conv_table.write_text(std::cout, dealii::ConvergenceTable::TextOutputFormat::org_mode_table);

    #ifdef OUTPUT_TABLE
    std::stringstream ss;
    ss << "errors_cavity_LTS_LFC_eta" << eta;

    ss << "_dg" << FE_DEGREE << "_coarse-fine-ratio" << FINE_MESH_FACTOR;

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
    convergence_cavity_1D te_lfc;
    te_lfc.run(0.1);
    te_lfc.run(0.0);

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
