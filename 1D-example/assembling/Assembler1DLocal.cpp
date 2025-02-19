

#include "Assembler1DLocal.h"

namespace MaxwellProblem1D::Assembling {

void assemble_mass_matrix_cell_H(
	dealii::FullMatrix<double> &cell_matrix,
	const dealii::FEValues<1> &fe_v,
	unsigned int dofs_per_cell_H) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const std::vector<double> &JxW = fe_v.get_JxW_values();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point)
	for (unsigned int i = 0; i < dofs_per_cell_H; ++i) {
	  const auto H_i = fe_v[H].value(i, q_point);
	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);

		cell_matrix(i, j) += H_j * H_i * JxW[q_point];
	  }
	}
}

void assemble_cell_curl(
	dealii::FullMatrix<double> &cell_curl_matrix,
	const dealii::FEValues<1> &fe_v,
	const double dofs_per_cell_E,
	const double dofs_per_cell_H) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Scalar E(1);

  const auto &JxW = fe_v.get_JxW_values();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point)
	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto grad_H_j = fe_v[H].gradient(j, q_point);

		cell_curl_matrix(i, j) -= grad_H_j[0] * E_i * JxW[q_point];
	  }
	}
}

void assemble_face_curl(
	const dealii::FEFaceValuesBase<1> &fe_v,
	const dealii::FEFaceValuesBase<1> &fe_v_neighbor,
	dealii::FullMatrix<double> &cell_curl_matrix,
	dealii::FullMatrix<double> &cell_face_matrix_ext_int,
	dealii::FullMatrix<double> &cell_face_matrix_int_ext,
	dealii::FullMatrix<double> &cell_curl_matrix_neighbor,
	double eps,
	double mu,
	double eps_neighbor,
	double mu_neighbor) {

  const dealii::FEValuesExtractors::Scalar H(0);
  const dealii::FEValuesExtractors::Scalar E(1);

  double w = std::sqrt(eps / mu);
  double w_ext = std::sqrt(eps_neighbor / mu_neighbor);
  double w_ges = w + w_ext;
  w /= w_ges;
  w_ext /= w_ges;

  const std::vector<double> &JxW = fe_v.get_JxW_values();

  const auto dofs_per_cell_E = cell_curl_matrix.n_rows();
  const auto dofs_per_cell_H = cell_curl_matrix.n_cols();

  const auto &normals = fe_v.get_normal_vectors();

  for (unsigned int q_point = 0; q_point < fe_v.n_quadrature_points; ++q_point) {
	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_j = fe_v[H].value(j, q_point);
		const auto n_times_H_j = -normals[q_point]*H_j;

		cell_curl_matrix(i, j) -= w * E_i * n_times_H_j[0] * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_i = fe_v[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {
		const auto H_neighbor_j = fe_v_neighbor[H].value(j, q_point);
		const auto n_times_H_neighbor_j = -normals[q_point]*H_neighbor_j;

		cell_face_matrix_int_ext(i, j) += w * E_i * n_times_H_neighbor_j[0] * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_neighbor_i = fe_v_neighbor[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		const auto H_j = fe_v[H].value(j, q_point);
		const auto n_times_H_j = -normals[q_point]*H_j;

		cell_face_matrix_ext_int(i, j) -= w_ext * E_neighbor_i * n_times_H_j[0] * JxW[q_point];
	  }
	}

	for (unsigned int i = 0; i < dofs_per_cell_E; ++i) {
	  const auto E_neighbor_i = fe_v_neighbor[E].value(i + dofs_per_cell_H, q_point);

	  for (unsigned int j = 0; j < dofs_per_cell_H; ++j) {

		const auto H_neighbor_j = fe_v_neighbor[H].value(j, q_point);
		const auto n_times_H_neighbor_j = -normals[q_point]*H_neighbor_j;

		cell_curl_matrix_neighbor(i, j) += w_ext * E_neighbor_i * n_times_H_neighbor_j[0] * JxW[q_point];
	  }
	}
  }
}

}// namespace MaxwellProblem1D::Assembling