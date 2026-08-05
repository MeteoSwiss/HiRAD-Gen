import pytest
import torch

from hirad.inference.stochastic_sampler import stochastic_sampler
from hirad.utils.patching import GridPatching2D


class DummyNet(torch.nn.Module):
    """Minimal EDM-style preconditioned model stub for the sampler."""

    sigma_min = 0.002
    sigma_max = 800.0

    def __init__(self):
        super().__init__()
        self.calls = 0

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, x_lr, t, class_labels=None, embedding_selector=None):
        self.calls += 1
        return 0.5 * x


def _run_sampler(num_steps=6, seed=0, **sampler_kwargs):
    torch.manual_seed(seed)
    latents = torch.randn(1, 3, 8, 8)
    img_lr = torch.randn(1, 4, 8, 8)
    net = DummyNet()
    out = stochastic_sampler(net, latents, img_lr, num_steps=num_steps, **sampler_kwargs)
    return net, out


############################################################################
#                    stochastic_sampler solver option                       #
############################################################################


class TestStochasticSamplerSolver:
    """Tests for the euler/heun solver option of stochastic_sampler."""

    def test_heun_network_evaluations(self):
        num_steps = 6
        net, out = _run_sampler(num_steps=num_steps, solver="heun")
        # Heun evaluates the network twice per step, except the last step.
        assert net.calls == 2 * num_steps - 1
        assert torch.isfinite(out).all()

    def test_euler_network_evaluations(self):
        num_steps = 6
        net, out = _run_sampler(num_steps=num_steps, solver="euler")
        # Euler evaluates the network once per step.
        assert net.calls == num_steps
        assert torch.isfinite(out).all()

    def test_default_solver_is_heun(self):
        num_steps = 6
        net_default, out_default = _run_sampler(num_steps=num_steps)
        net_heun, out_heun = _run_sampler(num_steps=num_steps, solver="heun")
        assert net_default.calls == net_heun.calls == 2 * num_steps - 1
        assert torch.equal(out_default, out_heun)

    def test_euler_and_heun_differ(self):
        _, out_euler = _run_sampler(solver="euler")
        _, out_heun = _run_sampler(solver="heun")
        assert not torch.equal(out_euler, out_heun)

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            _run_sampler(solver="rk4")

    def test_output_shape_matches_latents(self):
        for solver in ("euler", "heun"):
            _, out = _run_sampler(solver=solver)
            assert out.shape == (1, 3, 8, 8)

    @pytest.mark.parametrize("solver,expected_calls_factor", [("euler", 1), ("heun", 2)])
    def test_solver_with_patching(self, solver, expected_calls_factor):
        num_steps = 4
        torch.manual_seed(0)
        latents = torch.randn(1, 3, 16, 16)
        img_lr = torch.randn(1, 4, 16, 16)
        patching = GridPatching2D(
            img_shape=(16, 16), patch_shape=(8, 8), boundary_pix=0, overlap_pix=0
        )
        net = DummyNet()
        out = stochastic_sampler(
            net, latents, img_lr, patching=patching, num_steps=num_steps, solver=solver
        )
        expected_calls = (
            num_steps if expected_calls_factor == 1 else 2 * num_steps - 1
        )
        assert net.calls == expected_calls
        assert out.shape == (1, 3, 16, 16)
        assert torch.isfinite(out).all()
