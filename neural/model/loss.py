import torch
from torch import Tensor


def unsupervised_loss(
    soft_indicator_matrix: Tensor,
    adj: Tensor,
    lambda_1: float,
    lambda_2: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Unsupervised Loss from the paper
    Loss = lambda_1 *  sum_i ((sum_j T_i,j) -1)^2
    + lambda_2 * sum_i H_i,i
    + sum_i(sum_j(D_i,j * H_i,j))

    Args:
        soft_indicator_matrix (Tensor): model output
        adj (Tensor): Adjacency matrix
        lambda_1 (float): weight for the row wise constraint
        lambda_2 (float): weight for the no self loop constraint

    Returns:
        tuple: (total_loss, row_constraint_term, self_loop_term, min_distance_term)
    """
    # Heatmap for probability of an edge being in the optimal tour
    heatmap = _soft_indicator_matrix_to_heatmap(soft_indicator_matrix)
    row_wise_constraint = _row_wise_constraint(soft_indicator_matrix)
    no_self_loops_constraint = _no_self_loops_constraint(heatmap)
    min_distance_constraint = _min_distance_constraint(adj, heatmap)

    batch_size = soft_indicator_matrix.shape[0]
    row_term = lambda_1 * row_wise_constraint.sum() / batch_size
    self_loop_term = lambda_2 * no_self_loops_constraint.sum() / batch_size
    dist_term = min_distance_constraint.sum() / batch_size
    total = row_term + self_loop_term + dist_term
    return total, row_term, self_loop_term, dist_term


def _row_wise_constraint(soft_indicator_matrix: Tensor) -> Tensor:
    """
    Term 1:
    sum_i ((sum_j T_i,j) -1)^2

    pushes the sum of a row as close as possible to 1 (for valid probabilities)
    """
    return (soft_indicator_matrix.sum(dim=2) - 1.0).pow(2).sum(dim=1)


def _no_self_loops_constraint(heatmap: Tensor) -> Tensor:
    """
    Term 2
    sum_i H_i,i

    Self loops are not allowed
    """
    diagonal_elements = torch.diagonal(
        heatmap, dim1=-2, dim2=-1
    )  # -> returns (batch, problem_size)
    return diagonal_elements.sum(dim=1)  # because 0=batch


def _min_distance_constraint(adj: Tensor, heatmap: Tensor) -> Tensor:
    """
    Term 3
    sum_i(sum_j(D_i,j * H_i,j))

    The "Expected Cost of a tour" should be as small as possible
    """

    # Sum of the edges weighted by their probability according to heatmap
    # If the heatmap is good, this should be small
    return (heatmap * adj).sum(dim=(1, 2))


def _soft_indicator_matrix_to_heatmap(soft_indicator_matrix: Tensor) -> Tensor:
    """
    Transform T to H via
    H = sum_{t=1}^n-1 p_t p_{t+1}^T + p_n p_1^T

    That is, the sum of outer products of column p_t and p_{t+1}.
    The last part p_n*p_1^T is the wrap around to the start

    Parameters
    ----------
        - soft_indicator_matrix: Output of the model of shape (batch, problem_size, problem_size)
    Returns
    ----------
        - heatmap: Heatmap with probabilities of the edges being part of the optimal tour of shape (batch, problem_size, problem_size)
    """
    shifted_columns_transposed = torch.roll(
        torch.transpose(soft_indicator_matrix, 1, 2), -1, 1
    )
    heatmap = torch.matmul(soft_indicator_matrix, shifted_columns_transposed)
    return heatmap
