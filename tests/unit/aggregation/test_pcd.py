import torch
from pytest import mark, raises
from torch import Tensor
from torch.testing import assert_close
from utils.tensors import ones_, randn_, tensor_, zeros_

from torchjd.aggregation import PCD, PCDWeighting
from torchjd.aggregation._pcd import _solve_qp

from ._asserts import assert_expected_structure, assert_non_differentiable
from ._inputs import scaled_matrices, typical_matrices, typical_matrices_2_plus_rows

scaled_pairs = [(PCD(), m) for m in scaled_matrices]
typical_pairs = [(PCD(), m) for m in typical_matrices]
requires_grad_pairs = [(PCD(), ones_(3, 5, requires_grad=True))]


def _two_objectives_pcd(matrix: Tensor, tau: float, scales: Tensor) -> Tensor:
    """
    Closed-form solution of PCD with one secondary objective (Corollary 4.6 of the paper), used to
    derive expected values independently of the implementation.
    """

    g1, g2 = matrix[0] * scales[0], matrix[1] * scales[1]
    mu = ((tau * (g2 @ g2) - g1 @ g2) / (g2 @ g2)).clamp(min=0.0)
    direction = g1 + mu * g2
    return direction * matrix[0].norm() / direction.norm()


@mark.parametrize(["aggregator", "matrix"], scaled_pairs + typical_pairs)
def test_expected_structure(aggregator: PCD, matrix: Tensor) -> None:
    assert_expected_structure(aggregator, matrix)


@mark.parametrize(["aggregator", "matrix"], requires_grad_pairs)
def test_non_differentiable(aggregator: PCD, matrix: Tensor) -> None:
    assert_non_differentiable(aggregator, matrix)


def test_representations() -> None:
    A = PCD(tau=0.1, beta=0.9, eps=1e-6)
    assert repr(A) == "PCD(tau=0.1, beta=0.9, eps=1e-06)"
    assert str(A) == "PCD"

    W = PCDWeighting(tau=0.1, beta=0.9, eps=1e-6)
    assert repr(W) == "PCDWeighting(tau=0.1, beta=0.9, eps=1e-06)"


def test_zero_rows_returns_zero_vector() -> None:
    out = PCD()(tensor_([]).reshape(0, 3))
    assert_close(out, zeros_(3))


def test_zero_columns_returns_zero_vector() -> None:
    out = PCD()(tensor_([]).reshape(2, 0))
    assert out.shape == (0,)


def test_single_row_returns_it() -> None:
    J = randn_((1, 5))
    assert_close(PCD()(J), J[0])


@mark.parametrize("shape", [(2, 5), (3, 5), (5, 10), (9, 11)])
def test_output_has_the_norm_of_the_primary_gradient(shape: tuple[int, int]) -> None:
    J = randn_(shape) * randn_((shape[0], 1)).exp()
    out = PCD()(J)
    assert_close(out.norm(), J[0].norm())


def test_zero_primary_gradient_returns_zero_vector() -> None:
    J = randn_((3, 5))
    J[0] = 0.0
    assert_close(PCD()(J), zeros_(5))


def test_primary_gradient_is_returned_when_constraints_are_satisfied() -> None:
    J = tensor_([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    assert_close(PCD(tau=0.5)(J), J[0])


def test_primary_gradient_is_returned_when_constraints_are_infeasible() -> None:
    # The two secondary gradients are anti-parallel, so no direction improves both of them.
    J = tensor_([[1.0, 1.0], [0.0, 1.0], [0.0, -2.0]])
    assert_close(PCD()(J), J[0])


def test_primary_objective_is_sacrificed_for_an_opposed_secondary_objective() -> None:
    # The secondary constraint can only be satisfied by moving against the primary gradient, and
    # the output is then rescaled to the norm of the primary gradient.
    J = tensor_([[1.0, 0.0], [-3.0, 0.0]])
    assert_close(PCD(tau=0.02)(J), tensor_([-1.0, 0.0]))


@mark.parametrize("tau", [0.0, 0.02, 0.5, 1.0])
def test_two_objectives_matches_closed_form(tau: float) -> None:
    J = randn_((2, 6))
    scales = 1.0 / J.norm(dim=1)
    assert_close(PCD(tau=tau, eps=0.0)(J), _two_objectives_pcd(J, tau, scales))


def test_second_call_uses_debiased_moving_average() -> None:
    beta = 0.9
    J1 = randn_((2, 6))
    J2 = 10.0 * randn_((2, 6))
    A = PCD(tau=0.3, beta=beta, eps=0.0)
    A(J1)
    out = A(J2)

    ema = beta * (1 - beta) * J1.norm(dim=1) ** 2 + (1 - beta) * J2.norm(dim=1) ** 2
    scales = 1.0 / (ema / (1 - beta**2)).sqrt()
    assert_close(out, _two_objectives_pcd(J2, 0.3, scales))


def test_first_call_is_invariant_to_positive_row_scaling() -> None:
    J = randn_((4, 6))
    c = tensor_([2.0, 0.1, 30.0, 0.5])
    assert_close(PCD(eps=0.0)(c.unsqueeze(1) * J), c[0] * PCD(eps=0.0)(J))


@mark.parametrize("m", [2, 3, 5, 8])
@mark.parametrize("tau", [0.0, 0.02, 0.5])
def test_solve_qp_satisfies_kkt_conditions(m: int, tau: float) -> None:
    """
    Tests that the weights w = [1, mu_2, ..., mu_m] satisfy the KKT conditions of the QP. Since the
    QP is convex, they are sufficient for optimality. Stationarity holds by construction of w.
    """

    J = randn_((m, 10)).to(device="cpu", dtype=torch.float64)
    G = J @ J.T
    taus = torch.full([m - 1], tau, dtype=torch.float64)
    weights = _solve_qp(G, taus)
    mu = weights[1:]
    slacks = G[1:] @ weights - taus * G.diagonal()[1:]

    assert weights[0] == 1.0
    assert (mu >= 0.0).all()
    assert (slacks >= -1e-8).all()
    assert_close(mu * slacks, torch.zeros_like(mu), rtol=0.0, atol=1e-8)


def test_vector_tau_with_equal_values_matches_scalar_tau() -> None:
    J = randn_((4, 6))
    assert_close(PCD(tau=tensor_([0.3, 0.3, 0.3]))(J), PCD(tau=0.3)(J))


def test_vector_tau_with_wrong_length_raises() -> None:
    A = PCD(tau=tensor_([0.1, 0.2]))
    with raises(ValueError, match="tau"):
        A(randn_((4, 6)))


@mark.parametrize("matrix", typical_matrices_2_plus_rows)
def test_reset_restores_first_step_behavior(matrix: Tensor) -> None:
    A = PCD()
    first = A(matrix)
    A(2.0 * matrix + 1.0)
    A.reset()
    assert_close(first, A(matrix))


def test_weighting_reset_restores_first_step_behavior() -> None:
    J = randn_((3, 8))
    G = J @ J.T
    W = PCDWeighting()
    first = W(G)
    W(4.0 * G)
    W.reset()
    assert_close(first, W(G))


def test_changing_m_auto_resets() -> None:
    J = randn_((3, 8))
    A = PCD()
    A(randn_((4, 8)))
    assert_close(A(J), PCD()(J))


def test_aggregator_and_weighting_agree() -> None:
    A = PCD(tau=0.1)
    W = PCDWeighting(tau=0.1)
    for _ in range(3):
        J = randn_((3, 8))
        assert_close(W(J @ J.T) @ J, A(J))


def test_tau_setter_accepts_valid() -> None:
    A = PCD()
    A.tau = 0.0
    assert A.tau == 0.0
    A.tau = 1.0
    assert A.tau == 1.0
    tau = tensor_([0.1, 0.5])
    A.tau = tau
    assert A.tau is tau
    assert A.gramian_weighting.tau is tau


@mark.parametrize("tau", [-0.1, 1.1, float("nan")])
def test_tau_setter_rejects_out_of_range(tau: float) -> None:
    A = PCD()
    with raises(ValueError, match="tau"):
        A.tau = tau
    with raises(ValueError, match="tau"):
        A.tau = tensor_([0.1, tau])


def test_tau_setter_rejects_non_vector_tensor() -> None:
    A = PCD()
    with raises(ValueError, match="tau"):
        A.tau = tensor_([[0.1, 0.2]])


def test_beta_setter_accepts_valid() -> None:
    A = PCD()
    A.beta = 0.0
    assert A.beta == 0.0
    A.beta = 0.5
    assert A.beta == 0.5
    assert A.gramian_weighting.beta == 0.5


@mark.parametrize("beta", [-0.1, 1.0])
def test_beta_setter_rejects_out_of_range(beta: float) -> None:
    A = PCD()
    with raises(ValueError, match="beta"):
        A.beta = beta


def test_eps_setter_accepts_valid() -> None:
    A = PCD()
    A.eps = 0.0
    assert A.eps == 0.0
    A.eps = 1e-6
    assert A.eps == 1e-6
    assert A.gramian_weighting.eps == 1e-6


def test_eps_setter_rejects_negative() -> None:
    A = PCD()
    with raises(ValueError, match="eps"):
        A.eps = -1e-9
