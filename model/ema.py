"""
EMA Target Encoder for CAD-JEPA.
theta_bar <- tau * theta_bar + (1 - tau) * theta   after every optimizer.step()
"""

import copy
import torch
import torch.nn as nn


class EMATargetEncoder:

    # def __init__(self, online_encoder: nn.Module, tau: float = 0.996):
    #     self.tau = tau
    #     self.target = copy.deepcopy(online_encoder)
    #     for p in self.target.parameters():
    #         p.requires_grad_(False)
    def __init__(self, online_encoder: nn.Module, tau: float = 0.996):
        self.tau = tau
        # deepcopy on CPU — avoids CUDA dispatch issues with RotaryEmbedding
        device = next(online_encoder.parameters()).device
        self.target = copy.deepcopy(online_encoder.cpu())
        online_encoder.to(device)
        self.target.to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        """theta_bar <- tau * theta_bar + (1-tau) * theta"""
        for t_p, o_p in zip(self.target.parameters(), online_encoder.parameters()):
            t_p.data.mul_(self.tau).add_(o_p.data, alpha=1.0 - self.tau)

    def set_tau(self, tau: float) -> None:
        self.tau = tau

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def eval(self):
        self.target.eval()
        return self

    def train(self, mode: bool = True):
        self.target.eval()   # always eval — no batchnorm shifts
        return self

    def state_dict(self):
        return {"target": self.target.state_dict(), "tau": self.tau}

    def load_state_dict(self, state: dict):
        self.target.load_state_dict(state["target"])
        self.tau = state.get("tau", self.tau)
