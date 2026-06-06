"""
inference/base_runner.py
Abstract interface. Every model implements generate_one().
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseRunner(ABC):
    """
    To add a new model:
      1. Create inference/runners/my_model.py
      2. Subclass BaseRunner
      3. Implement setup() and generate_one()
      4. Register in generate.py RUNNERS dict
    """

    @abstractmethod
    def setup(self, **kwargs):
        """Load model, weights, tokenizer — whatever the model needs."""
        pass

    @abstractmethod
    def generate_one(self, uid: str, text: str) -> np.ndarray | None:
        """
        Generate a CAD sequence for one (uid, text) pair.

        Returns:
            np.ndarray of shape [N, 17] dtype int64  — the pred vec
            None if generation failed
        """
        pass

    def post_process(self, cmds_raw: list, args_raw: np.ndarray) -> np.ndarray | None:
        """
        Default post-processing: fix empty loops, require SOL+EXT.
        Subclasses can override.
        """
        cmds = self._fix_sequence(cmds_raw)
        if not cmds:
            return None
        rows = [[c] + args_raw[t].tolist() for t, c in enumerate(cmds)]
        return np.array(rows, dtype=np.int64)

    @staticmethod
    def _fix_sequence(cmds: list) -> list:
        if 3 in cmds:
            cmds = cmds[:cmds.index(3)]
        fixed = []
        i = 0
        while i < len(cmds):
            if cmds[i] == 4:
                j = i + 1
                n = 0
                while j < len(cmds) and cmds[j] in (0, 1, 2):
                    n += 1; j += 1
                if n > 0:
                    fixed.extend(cmds[i:j])
                i = j
            else:
                fixed.append(cmds[i]); i += 1
        if not fixed or fixed[0] != 4: return []
        if 5 not in fixed: return []
        return fixed