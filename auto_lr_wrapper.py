import torch
import random

class BaseOptimizerWrapper(torch.optim.Optimizer):
    """
    A base wrapper class for PyTorch optimizers, designed to lay the foundation for implementing
    an active learning rate adjustment strategy based on gradient sign continuity across epochs.

    The class encapsulates an optimizer and applies an adjustment rule to the learning rates of
    each parameter, enhancing optimization by considering the dynamic behavior of gradients.
    This strategy aims to increase the learning rate when the gradient sign remains constant and
    decrease it when the gradient sign changes, thereby adapting to the optimization landscape.

    Attributes:
        optimizer (torch.optim.Optimizer): The encapsulated optimizer responsible for the actual
                                           update step based on the computed gradients.
        alpha_high (float): The additive factor by which the learning rate is increased when the
                            gradient sign is consistent.
        alpha_low (float): The multiplicative factor by which the learning rate is decreased when
                           the gradient sign changes.
        cumulative_grad (dict): A state dictionary storing the cumulative gradient for each parameter.
        prev_grad (dict): A state dictionary storing the previous gradient for each parameter.

    The adjustment mechanism for the learning rate is as follows:
        - If the gradient sign of a parameter does not change, indicating a steady descent direction,
          the learning rate is increased additively by alpha_high to encourage faster convergence.
        - If the gradient sign changes, suggesting potential overshooting or oscillation, the learning
          rate is decreased by a factor of alpha_low to ensure stability.

    This adjustment protocol is inspired by adaptive control theory, adapting the learning rate
    based on the observed optimization behavior to achieve desired convergence properties.

    The base class does not implement the adjustment rule; it sets up the structure for subclasses
    to define the specific active adjustment based on the optimization landscape.

    Args:
        optimizer (torch.optim.Optimizer): The original PyTorch optimizer.
        alpha_high (float): The additive increase to the learning rate for consistent gradient signs.
                            Should be a value between 0 and 1.
        alpha_low (float): The multiplicative decrease to the learning rate for changing gradient signs.
                           Should also be between 0 and 1 and complement alpha_high such that their sum is 1.

    Raises:
        ValueError: If alpha_high or alpha_low is not in the range (0, 1), or their sum is not equal to 1.
    """

    def __init__(self, optimizer, alpha_high=0.1, alpha_low=0.9):
        """
        Initializes the BaseOptimizerWrapper instance.

        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer to wrap.
            alpha_high (float): Weight for the high learning rate portion of the update rule.
            alpha_low (float): Weight for the low learning rate portion of the update rule.
        """
        if not (0.0 < alpha_high < 1.0) or not (0.0 < alpha_low < 1.0):
            raise ValueError(
                "alpha_high and alpha_low must be between 0 and 1, exclusive."
            )
        if not (alpha_high + alpha_low == 1.0):
            raise ValueError("The sum of alpha_high and alpha_low must be 1.")

        self.optimizer = optimizer
        self.defaults = optimizer.defaults
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.alpha_high = alpha_high
        self.alpha_low = alpha_low
        self.cumulative_grad = {
            id(p): torch.zeros_like(p.data)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        self.prev_grad = {
            id(p): torch.zeros_like(p.data)
            for group in optimizer.param_groups
            for p in group["params"]
        }

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            The loss returned by the optimizer's step function.
        """
        loss = self.optimizer.step(closure)
        self.update_grads()  # Update the gradients after each step
        return loss

    @torch.no_grad()
    def update_grads(self):
        """Updates the cumulative gradients for each parameter."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    param_id = id(p)
                    self.cumulative_grad[param_id].add_(p.grad.data)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clears the gradients of all optimized parameters.

        Args:
            set_to_none (bool): Instead of setting to zero, set the grads to None.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        """
        Adds a parameter group to the optimizer's param_groups.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group-specific options.
        """
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        """
        Returns the state of the optimizer as well as the custom wrapper state.

        Returns:
            A dict containing both the state of the original optimizer and the additional state from the wrapper.
        """
        base_state = self.optimizer.state_dict()
        custom_state = {
            "cumulative_grad": self.cumulative_grad,
            "prev_grad": self.prev_grad,
        }
        return {"base_state": base_state, "custom_state": custom_state}

    def load_state_dict(self, state_dict):
        """
        Loads the state of the optimizer and the custom wrapper state.

        Args:
            state_dict (dict): State dictionary as returned by `state_dict`.
        """
        self.optimizer.load_state_dict(state_dict["base_state"])
        self.cumulative_grad = state_dict["custom_state"]["cumulative_grad"]
        self.prev_grad = state_dict["custom_state"]["prev_grad"]


class ActiveOptimizerWrapper(BaseOptimizerWrapper):
    def __init__(
        self,
        optimizer,
        alpha_high=0.1,
        alpha_low=0.9,
        initial_stochastic_ratio=0.1,
        dampening_rate=0.01,
        apply_dampening=True,
    ):
        super().__init__(optimizer, alpha_high, alpha_low)
        self.stochastic_ratio = initial_stochastic_ratio
        self.dampening_rate = dampening_rate
        self.apply_dampening = apply_dampening
        self.grad_variance = {
            id(p): 0 for group in optimizer.param_groups for p in group["params"]
        }

    def end_epoch_adjustment(self):
        for group in self.param_groups:
            for p in group["params"]:
                param_id = id(p)
                cum_grad = self.cumulative_grad[param_id]
                prev_grad = self.prev_grad[param_id]
                # Apply stochastic update based on stochastic_ratio
                if random.random() < self.stochastic_ratio:
                    # Adjust learning rate
                    if torch.sign(torch.dot(cum_grad.view(-1), prev_grad.view(-1))) > 0:
                        # Increase learning rate by an absolute increment
                        group["lr"] += self.alpha_high
                    else:
                        # Decrease learning rate by a multiplicative factor
                        group["lr"] *= self.alpha_low
                # Reset gradients for the next epoch
                self.prev_grad[param_id] = cum_grad.clone()
                self.cumulative_grad[param_id].zero_()

        # Conditionally apply dampening to the stochastic_ratio
        if self.apply_dampening:
            self.stochastic_ratio = max(self.stochastic_ratio - self.dampening_rate, 0)