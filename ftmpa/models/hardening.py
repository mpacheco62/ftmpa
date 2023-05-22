#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Hardenings models                                            ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************

from numpy import exp
import numpy as np
from abc import ABC, abstractmethod


class HardeningModel(ABC):
    @abstractmethod
    def __call__(self, ep: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def der(self, ep: np.ndarray) -> np.ndarray:
        pass



# sig = sig0 + k*ep + q*(1-exp(-n*ep))
class VoceMod(HardeningModel):
    sig0: float
    k: float
    q: float
    n: float

    def __init__(self, sig0: float, k: float, q: float, n: float):
        self.sig0 = sig0
        self.k = k
        self.q = q
        self.n = n

    def __call__(self, ep: np.ndarray) -> np.ndarray:
        sig0 = self.sig0
        k = self.k
        q = self.q
        n = self.n

        try:
            stress = sig0 + k*ep + q*(1-exp(-n*ep))
        except OverflowError:
            ep = 1.0e5
            stress = sig0 + k*ep + q*(1-exp(-n*ep))
        return stress

    def der(self, ep: np.ndarray) -> np.ndarray:
        k = self.k
        q = self.q
        n = self.n

        try:
            dstress = k + q*n*exp(-n*ep)
        except OverflowError:
            ep = 1.0e5
            dstress = k + q*n*exp(-n*ep)
        return dstress
