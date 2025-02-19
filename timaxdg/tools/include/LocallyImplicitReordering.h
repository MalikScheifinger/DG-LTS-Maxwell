#ifndef LOCALLY_IMPLICIT_REORDERING_H_
#define LOCALLY_IMPLICIT_REORDERING_H_

#include <algorithm>
#include <vector>
 #include <numeric>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

namespace MaxwellProblem::LocallyImplicit {

/**
   * @brief Enumerates the different sets a cell can belong
   * to in a locally implicit context.
   */
enum LocallyImplicitMeshSets {
  Explicit_Explicit,///< Explicit cell with only explicit neighbors
  Explicit_Implicit,///< Explicit cell with at least one implicit neighbor
  Implicit_Explicit,///< Implicit cell with at least one explicit neighbor
  Implicit_Implicit,///< Implicit cell with only implicit neighbors
};

template<int dim>
void renumbering_dofs_explicit_to_implicit(
	dealii::DoFHandler<dim> &dof_handler) {

  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> ordered_cells;
  ordered_cells.reserve(dof_handler.get_triangulation().n_active_cells());

  for (const auto &cell : dof_handler.active_cell_iterators()) {
	ordered_cells.push_back(cell);
  }

  auto cmp = [](const typename dealii::DoFHandler<dim>::active_cell_iterator &cell1,
				const typename dealii::DoFHandler<dim>::active_cell_iterator &cell2) {
	return cell1->material_id() < cell2->material_id();
  };

  std::stable_sort(ordered_cells.begin(), ordered_cells.end(), cmp);

  dealii::DoFRenumbering::cell_wise(dof_handler, ordered_cells);
}

template<int dim>
void set_cells_explicit_implicit(dealii::Triangulation<dim> &triangulation,
								 const double min_dia_explicit) {
  // implicit cells (fine and coarse)
  typename dealii::Triangulation<dim>::active_cell_iterator
	  cell = triangulation.begin_active(),
	  endc = triangulation.end();
  for (; cell != endc; ++cell) {
	if ((min_dia_explicit - cell->diameter()) > 1e-6) {
	  cell->set_material_id(LocallyImplicitMeshSets::Implicit_Implicit);
	} else {
	  // set cell explicit first; if necessary, it is changed
	  cell->set_material_id(LocallyImplicitMeshSets::Explicit_Explicit);

	  for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
		typename dealii::Triangulation<dim>::face_iterator face = cell->face(face_no);
		if (!(face->at_boundary())) {
		  Assert(cell->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
		  typename dealii::Triangulation<dim>::cell_iterator neighbor = cell->neighbor(face_no);
		  if (face->has_children()) {
			for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no) {
			  typename dealii::Triangulation<dim>::cell_iterator neighbor_child = cell->neighbor_child_on_subface(face_no, subface_no);
			  Assert(!neighbor_child->has_children(), dealii::ExcInternalError());

			  if ((min_dia_explicit - neighbor_child->diameter()) > 1e-6) {
				cell->set_material_id(LocallyImplicitMeshSets::Implicit_Explicit);

				break;
			  }
			}
		  } else if ((min_dia_explicit - neighbor->diameter()) > 1e-6) {
			cell->set_material_id(LocallyImplicitMeshSets::Implicit_Explicit);
		  }

		  if (cell->material_id() == LocallyImplicitMeshSets::Implicit_Implicit)
			break;
		}
	  }
	}
  }

  // explicit cells with implicit neighbors
  typename dealii::Triangulation<dim>::active_cell_iterator
	  cell2 = triangulation.begin_active(),
	  endc2 = triangulation.end();
  for (; cell2 != endc2; ++cell2) {
	if (cell2->material_id() == LocallyImplicitMeshSets::Explicit_Explicit) {
	  for (unsigned int face_no = 0; face_no < dealii::GeometryInfo<dim>::faces_per_cell; ++face_no) {
		typename dealii::Triangulation<dim>::face_iterator face = cell2->face(face_no);
		if (!(face->at_boundary())) {
		  Assert(cell2->neighbor(face_no).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
		  if (face->has_children()) {
			for (unsigned int subface_no = 0; subface_no < face->number_of_children(); ++subface_no) {
			  typename dealii::Triangulation<dim>::cell_iterator neighbor_child = cell2->neighbor_child_on_subface(face_no, subface_no);
			  Assert(!neighbor_child->has_children(), dealii::ExcInternalError());

			  if (neighbor_child->material_id() == LocallyImplicitMeshSets::Implicit_Explicit) {
				cell2->set_material_id(LocallyImplicitMeshSets::Explicit_Implicit);

				break;
			  }
			}
		  } else if (cell2->neighbor(face_no)->material_id() == LocallyImplicitMeshSets::Implicit_Explicit) {
			cell2->set_material_id(LocallyImplicitMeshSets::Explicit_Implicit);
		  }

		  if (cell2->material_id() == LocallyImplicitMeshSets::Explicit_Implicit)
			break;
		}
	  }
	}
  }
}

template<int dim>
void get_number_dofs_exp_imp(
								dealii::DoFHandler<dim> &dof_handler,
							 std::vector<unsigned int> &number_imp_exp_dofs) {
  // Resize vector
  number_imp_exp_dofs.resize(8, 0);


  for (const auto &cell : dof_handler.active_cell_iterators()) {
	number_imp_exp_dofs[cell->material_id()]++;
  number_imp_exp_dofs[4 + cell->material_id()]++;
  }

  /* std::cout << "Number of \n"
			   "implicit fine cells: "
			<< number_imp_exp_dofs[3] << "\n"
										 "implicit coarse cells: "
			<< number_imp_exp_dofs[2] << "\n"
										 "explicit cells with implicit neighbor cells: "
			<< number_imp_exp_dofs[1] << "\n"
										 "explicit cells without implicit neighbor cells: "
			<< number_imp_exp_dofs[0] << std::endl
			<< std::endl; */

  // Change in case of upwind fluxes and different polynomial degrees!
  for(int i = 0; i<4; i++) {
    number_imp_exp_dofs[i] *= dof_handler.get_fe().get_sub_fe(0,1).dofs_per_cell;
    number_imp_exp_dofs[4+i] *= dof_handler.get_fe().get_sub_fe(1,2).dofs_per_cell;
  }
  
  int sum = 0;
  for(int i = 0; i<8; i++) sum += number_imp_exp_dofs[i];
  // std::cout << "Total dofs??: " << dof_handler.n_dofs() << " = sum : " << sum << std::endl;
}

}// namespace MaxwellProblem::LocallyImplicit

#endif//LOCALLY_IMPLICIT_REORDERING_H_