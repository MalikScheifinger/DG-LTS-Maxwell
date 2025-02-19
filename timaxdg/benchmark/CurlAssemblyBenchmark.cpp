
#include <vector>

#include <benchmark/benchmark.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "Assembler3D.h"
#include "IsotropicConstant.h"

#define SUBDIVISIONS_START 1
#define SUBDIVISIONS_END 15
#define SUBDIVISIONS_STEP 1
#define ITERATIONS 10
#define DEGREE 2

static void CurlMatrixAssembly(benchmark::State &state) {

  // setup finite elements

  dealii::Triangulation<3> triangulation;
  dealii::Point<3> p0({0, 0, 0});
  dealii::Point<3> p1({1, 1, 1});
  unsigned int one_rep = static_cast<unsigned int>(state.range(0));
  std::vector<unsigned int> reps = {one_rep, one_rep, one_rep};
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, reps, p0, p1);

  dealii::DoFHandler<3> dof_handler(triangulation);
  dealii::MappingQ1<3> mapping;
  dealii::QGauss<3> quadrature(DEGREE + 2);
  dealii::QGauss<2> face_quadrature(DEGREE + 1);

  dealii::FESystem<3> fe(
	  dealii::FESystem<3>(dealii::FE_DGQ<3>(DEGREE), 3), 1,
	  dealii::FESystem<3>(dealii::FE_DGQ<3>(DEGREE), 3), 1);

  dof_handler.distribute_dofs(fe);
  dealii::DoFRenumbering::component_wise(dof_handler, {0, 0, 0, 1, 1, 1});

  MaxwellProblem::Data::IsotropicConstant<3> mu(1);
  MaxwellProblem::Data::IsotropicConstant<3> eps(1);

  MaxwellProblem::Assembling::Assembler3D assembler(fe,
													mapping,
													quadrature,
													face_quadrature,
													dof_handler,
													mu,
													eps);

  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparsityPattern pattern;

  assembler.generate_curl_pattern(curl_matrix, pattern);

  for (auto _ : state) {
	assembler.assemble_curl_matrix(curl_matrix);
  }

  state.counters["DoF"] = dof_handler.n_dofs();
  state.counters["DoF/s"] = benchmark::Counter(dof_handler.n_dofs() * ITERATIONS, benchmark::Counter::kIsRate);
}
BENCHMARK(CurlMatrixAssembly)
	->DenseRange(SUBDIVISIONS_START, SUBDIVISIONS_END, SUBDIVISIONS_STEP)
	->Iterations(ITERATIONS)
	->Unit(benchmark::kSecond);

static void CurlMatrixAssemblyParallel(benchmark::State &state) {

  // setup finite elements

  dealii::Triangulation<3> triangulation;
  dealii::Point<3> p0({0, 0, 0});
  dealii::Point<3> p1({1, 1, 1});
  unsigned int one_rep = static_cast<unsigned int>(state.range(0));
  std::vector<unsigned int> reps = {one_rep, one_rep, one_rep};
  dealii::GridGenerator::subdivided_hyper_rectangle(triangulation, reps, p0, p1);

  dealii::DoFHandler<3> dof_handler(triangulation);
  dealii::MappingQ1<3> mapping;
  dealii::QGauss<3> quadrature(DEGREE + 2);
  dealii::QGauss<2> face_quadrature(DEGREE + 1);

  dealii::FESystem<3> fe(
	  dealii::FESystem<3>(dealii::FE_DGQ<3>(DEGREE), 3), 1,
	  dealii::FESystem<3>(dealii::FE_DGQ<3>(DEGREE), 3), 1);

  dof_handler.distribute_dofs(fe);
  dealii::DoFRenumbering::component_wise(dof_handler, {0, 0, 0, 1, 1, 1});

  MaxwellProblem::Data::IsotropicConstant<3> mu(1);
  MaxwellProblem::Data::IsotropicConstant<3> eps(1);

  MaxwellProblem::Assembling::Assembler3D assembler(fe,
													mapping,
													quadrature,
													face_quadrature,
													dof_handler,
													mu,
													eps);

  dealii::BlockSparseMatrix<double> curl_matrix;
  dealii::BlockSparsityPattern pattern;

  assembler.generate_curl_pattern(curl_matrix, pattern);

  for (auto _ : state) {
	assembler.assemble_curl_matrix_parallel(curl_matrix);
  }

  state.counters["DoF"] = dof_handler.n_dofs();
  state.counters["DoF/s"] = benchmark::Counter(dof_handler.n_dofs() * ITERATIONS, benchmark::Counter::kIsRate);
}
BENCHMARK(CurlMatrixAssemblyParallel)
	->DenseRange(SUBDIVISIONS_START, SUBDIVISIONS_END, SUBDIVISIONS_STEP)
	->Iterations(ITERATIONS)
	->Unit(benchmark::kSecond);

BENCHMARK_MAIN();