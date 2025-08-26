from typing import Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAutoencoder(nn.Module, ABC):
    """
    Base class for Autoencoders.

    Methods (training, gradients enabled):
        - _encode(x): z
        - _decode(z): x_hat
        - forward(x): (x_hat, z)

    Methods (inference-only, no gradients, optional eval()):
        - encode(x, use_eval=True): z
        - decode(z, use_eval=True): x_hat
    """

    # --- gradient-enabled training APIs (to be implemented by subclasses) ---
    @abstractmethod
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Training encode: returns z (grad-enabled), e.g. shape [B, D_z]."""
        pass

    @abstractmethod
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Training decode: returns x_hat (grad-enabled), e.g. shape [B, ...]."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward: returns (x_hat, z) with gradients."""
        pass

    # --- inference-only convenience wrappers (no grad; optional eval mode) ---
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, use_eval: bool = True) -> torch.Tensor:
        """
        Inference encode (no gradients). If use_eval=True, temporarily sets eval()
        so BN/Dropout behave deterministically, then restores the previous mode.

        Args:
            x (torch.Tensor): Input, shape [B, ...].
            use_eval (bool): If True, run temporarily in eval mode.

        Returns:
            torch.Tensor: Latent code z (e.g., [B, D_z]).
        """
        if not use_eval:
            return self._encode(x)
        prev = self.training
        try:
            self.eval()
            return self._encode(x)
        finally:
            self.train(prev)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, use_eval: bool = True) -> torch.Tensor:
        """
        Inference decode (no gradients). If use_eval=True, temporarily sets eval()
        and restores the previous mode afterward.

        Args:
            z (torch.Tensor): Latent code (e.g., [B, D_z]).

        Returns:
            torch.Tensor: Reconstruction/logits x_hat (e.g., [B, ...]).
        """
        if not use_eval:
            return self._decode(z)
        prev = self.training
        try:
            self.eval()
            return self._decode(z)
        finally:
            self.train(prev)


class BaseVariationalAutoencoder(nn.Module, ABC):
    """
    Base class for Variational Autoencoders.

    Methods (training, gradients enabled):
        - _encode(x, S): (z, mu, log_var)
        - _decode(z): x_hat
        - forward(x, S): (x_hat, z, mu, log_var)

    Methods (inference-only, no gradients, optional eval()):
        - encode(x, use_eval=True): (z, mu, log_var)  # uses S=1 internally
        - decode(z, use_eval=True): x_hat
    """

    # --- gradient-enabled training APIs (to be implemented by subclasses) ---
    @abstractmethod
    def _encode(self, x: torch.Tensor, S: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training encode + sample via the model's sampling layer.

        Args:
            x (torch.Tensor): Input, shape [B, ...].
            S (int): Number of Monte Carlo samples from q(z|x).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z       (torch.Tensor): Samples, shape [B, S, D_z].
                - mu      (torch.Tensor): Mean,    shape [B, D_z].
                - log_var (torch.Tensor): Log-var, shape [B, D_z].
        """
        pass

    @abstractmethod
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Training decode to input space.

        Args:
            z (torch.Tensor): [B, D_z] or [B, S, D_z].

        Returns:
            torch.Tensor: x_hat, [B, ...] if z is [B, D_z], else [B, S, ...].
        """
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, S: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward: reconstruction(s), latent sample(s), and posterior params.

        Args:
            x (torch.Tensor): Input, shape [B, ...].
            S (int): Number of Monte Carlo samples from q(z|x).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_hat   (torch.Tensor): Reconstructions/logits,
                                          [B, ...] if S=1, else [B, S, ...].
                - z       (torch.Tensor): Latent samples, shape [B, S, D_z].
                - mu      (torch.Tensor): Mean of q(z|x), shape [B, D_z].
                - log_var (torch.Tensor): Log-var of q(z|x), shape [B, D_z].

        Notes:
            - When S>1, broadcasting x → [B, S, ...] during loss computation
              allows evaluating log p(x|z_s) for each sample without copying x.
            - For Bernoulli likelihoods, x_hat should be logits.
        """
        pass

    # --- inference-only convenience wrappers (no grad; optional eval mode) ---
    @torch.inference_mode()
    def encode(self, x: torch.Tensor, use_eval: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference encode (no gradients). Calls `_encode(x, S=1)` and returns (z, mu, log_var).
        If use_eval=True, temporarily sets eval() so BN/Dropout behave deterministically, 
        then restores the previous mode.

        Args:
            x (torch.Tensor): Input, shape [B, ...].
            use_eval (bool): If True, run temporarily in eval mode.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z       (torch.Tensor): Latent samples, shape [B, 1, D_z].
                - mu      (torch.Tensor): Mean of q(z|x), shape [B, D_z].
                - log_var (torch.Tensor): Log-var of q(z|x), shape [B, D_z].
        """
        if not use_eval:
            z, mu, log_var = self._encode(x, S=1)
            return z, mu, log_var
        prev = self.training
        try:
            self.eval()
            z, mu, log_var = self._encode(x, S=1)
            return z, mu, log_var
        finally:
            self.train(prev)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, use_eval: bool = True) -> torch.Tensor:
        """
        Inference decode (no gradients). If use_eval=True, temporarily sets eval()
        and restores the previous mode afterward.

        Args:
            z (torch.Tensor): Latent input, [B, D_z] or [B, S, D_z].
            use_eval (bool): If True, run temporarily in eval mode.

        Returns:
            torch.Tensor: Reconstructions/logits x_hat, shape [B, S, ...].
        """
        if not use_eval:
            return self._decode(z)
        prev = self.training
        try:
            self.eval()
            return self._decode(z)
        finally:
            self.train(prev)
