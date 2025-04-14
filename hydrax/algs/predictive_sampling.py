from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class PSParams(SamplingParams):
    """Policy parameters for predictive sampling.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


class PredictiveSampling(SamplingBasedController):
    """A simple implementation of https://arxiv.org/abs/2212.00541."""

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        use_noise_ramp: bool = False,
        noise_ramp: bool = 4.0,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control tapes to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls
                If using noise ramp, this is the max value.
            use_noise_ramp: Whether to use a noise ramp for the covariance.
            noise_ramp: The maximum value of the noise ramp.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.use_noise_ramp = use_noise_ramp
        self.noise_ramp = noise_ramp

    def init_params(self, seed: int = 0) -> PSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(seed)
        return PSParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: PSParams) -> Tuple[jax.Array, PSParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )

        # apply noise ramping or simply scaling by the noise_level
        ramp = jnp.linspace(
            1.0,
            self.noise_ramp,
            self.num_knots,
        )[None, :, None]
        noise = jnp.where(
            self.use_noise_ramp,
            self.noise_level * noise * ramp,
            self.noise_level * noise,
        )

        # The original mean of the distribution is included as a sample
        controls = params.mean + noise
        controls = controls.at[0].set(params.mean)
        return controls, params.replace(rng=rng)

    def update_params(self, params: PSParams, rollouts: Trajectory) -> PSParams:
        """Update the policy parameters by choosing the lowest-cost rollout."""
        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        reduced_costs = self.risk_strategy.combine_costs(rollouts.costs)
        costs = jnp.sum(reduced_costs, axis=-1)  # sum over time steps
        best_idx = jnp.argmin(costs)
        mean = rollouts.knots[best_idx]
        return params.replace(mean=mean)
