from abc import ABC, abstractmethod
import numpy as np

from ..hardening import HardeningModel, VoceMod

# from ..models.hardening import HardeningModel, VoceMod

class HandlerHardening(ABC):
    @abstractmethod
    def __call__(self, param: dict[str, float], ep: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def model(self, param: dict[str, float]) -> HardeningModel:
        pass

class HandlerVoceMod(HandlerHardening):
    sig0: str
    k: str
    q: str
    n: str

    def __init__(self, sig0: str, k: str, q: str, n: str):
        self.sig0 = sig0
        self.k = k
        self.q = q
        self.n = n

    def __call__(self, param: dict[str, float], ep: np.ndarray):
        model = self.model(param)
        return model(ep)
    
    def model(self, param: dict[str, float]):
        sig0 = param[self.sig0]
        k = param[self.k]
        q = param[self.q]
        n = param[self.n]
        model = VoceMod(sig0=sig0, k=k, q=q, n=n)
        return model
