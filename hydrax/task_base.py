from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import MjData, MjModel, mjx


@dataclass
class CostMetadata:
    """Metadata for the cost function."""


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model)

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        params: Any,
        metadata: CostMetadata,
    ) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.
            params: The policy params.
            metadata: Metadata for the cost function.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """

    @abstractmethod
    def terminal_cost(
        self, state: mjx.Data, params: Any, metadata: CostMetadata
    ) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.
            params: The policy params.
            metadata: Metadata for the cost function.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def compute_cost_metadata(self, state: mjx.Data) -> CostMetadata:
        """Computes cost metadata for the current state.

        For example, this function could compute spline parameters used for
        tracking-based costs.
        """
        return CostMetadata()

    def pre_optimize(self, state: mjx.Data, params: Any) -> None:
        """Hook for pre-optimization processing."""
        return None

    def post_step(self, mj_model: MjModel, mj_data: MjData) -> None:
        """A hook for post-step processing after each simulation step.

        This method is called after each simulation step, and can be used to
        update the task state or perform any other necessary operations.

        By default, does nothing.

        Args:
            mj_model: The MuJoCo model.
            mj_data: The MuJoCo data object.
        """
        return None
