#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Handler yields models                                        ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************


from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

from ..yields import Cazacu, VonMises, YieldModel

class HandlerYield(ABC):
    @abstractmethod
    def __call__(self, param: dict[str, float], stress: np.ndarray, ep: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def model(self, param: dict[str, float]) -> YieldModel:
        pass


class HandlerCazacuVar(HandlerYield):
    # {"C1": {"C1_1": [1.0, 0.0]}}
    C11: list[tuple[str, float]]
    C22: list[tuple[str, float]]
    C33: list[tuple[str, float]]
    C44: list[tuple[str, float]]
    C55: list[tuple[str, float]]
    C66: list[tuple[str, float]]
    C12: list[tuple[str, float]]
    C13: list[tuple[str, float]]
    C23: list[tuple[str, float]]
    k: list[tuple[str, float]]
    a: str

    def __init__(self,
                 C11: list[tuple[str, float]],
                 C22: list[tuple[str, float]],
                 C33: list[tuple[str, float]],
                 C44: list[tuple[str, float]],
                 C55: list[tuple[str, float]],
                 C66: list[tuple[str, float]],
                 C12: list[tuple[str, float]],
                 C13: list[tuple[str, float]],
                 C23: list[tuple[str, float]],
                 k: list[tuple[str, float]],
                 a: str):
        self.C11 = C11
        self.C22 = C22
        self.C33 = C33
        self.C44 = C44
        self.C55 = C55
        self.C66 = C66
        self.C12 = C12
        self.C13 = C13
        self.C23 = C23
        self.k = k
        self.a = a

    def _get_constants(self, param: dict[str, float], var: list[tuple[str, float]]):
        temp_const = []
        temp_ep = []
        for ivar, ep in var:
            temp_ep.append(ep)
            temp_const.append(param[ivar])
        return temp_ep, temp_const
    
    def model(self, param: dict[str, float]):
        C11_data = self._get_constants(param, self.C11)
        C11 = interp1d(x=C11_data[0], y=C11_data[1], fill_value='extrapolate')

        C22_data = self._get_constants(param, self.C22)
        C22 = interp1d(x=C22_data[0], y=C22_data[1], fill_value='extrapolate')

        C33_data = self._get_constants(param, self.C33)
        C33 = interp1d(x=C33_data[0], y=C33_data[1], fill_value='extrapolate')

        C44_data = self._get_constants(param, self.C44)
        C44 = interp1d(x=C44_data[0], y=C44_data[1], fill_value='extrapolate')

        C55_data = self._get_constants(param, self.C55)
        C55 = interp1d(x=C55_data[0], y=C55_data[1], fill_value='extrapolate')

        C66_data = self._get_constants(param, self.C66)
        C66 = interp1d(x=C66_data[0], y=C66_data[1], fill_value='extrapolate')

        C12_data = self._get_constants(param, self.C12)
        C12 = interp1d(x=C12_data[0], y=C12_data[1], fill_value='extrapolate')

        C13_data = self._get_constants(param, self.C13)
        C13 = interp1d(x=C13_data[0], y=C13_data[1], fill_value='extrapolate')

        C23_data = self._get_constants(param, self.C23)
        C23 = interp1d(x=C23_data[0], y=C23_data[1], fill_value='extrapolate')

        k_data = self._get_constants(param, self.k)
        k = interp1d(x=k_data[0], y=k_data[1], fill_value='extrapolate')

        model = Cazacu(C11=C11, C22=C22, C33=C33, C44=C44, C55=C55, C66=C66,
                       C12=C12, C13=C13, C23=C23, k=k, a=param[self.a])

        return model 
    
    def __call__(self, param: dict[str, float], stress: np.ndarray, ep: np.ndarray):
        model = self.model(param)
        return model(stress, ep)


class HandlerCazacu(HandlerYield):
    # {"C1": {"C1_1": [1.0, 0.0]}}
    C11: str
    C22: str
    C33: str
    C44: str
    C55: str
    C66: str
    C12: str
    C13: str
    C23: str
    k: str
    a: str

    def __init__(self,
                 C11: str,
                 C22: str,
                 C33: str,
                 C44: str,
                 C55: str,
                 C66: str,
                 C12: str,
                 C13: str,
                 C23: str,
                 k: str,
                 a: int):
        self.C11 = C11
        self.C22 = C22
        self.C33 = C33
        self.C44 = C44
        self.C55 = C55
        self.C66 = C66
        self.C12 = C12
        self.C13 = C13
        self.C23 = C23
        self.k = k
        self.a = a
    
    def model(self, param: dict[str, float]):
        C11 = lambda _: param[self.C11]
        C22 = lambda _: param[self.C22]
        C33 = lambda _: param[self.C33]
        C44 = lambda _: param[self.C44]
        C55 = lambda _: param[self.C55]
        C66 = lambda _: param[self.C66]
        C23 = lambda _: param[self.C23]
        C13 = lambda _: param[self.C13]
        C12 = lambda _: param[self.C12]
        k = lambda _: param[self.k]
        
        model = Cazacu(C11=C11, C22=C22, C33=C33, C44=C44, C55=C55, C66=C66,
                       C12=C12, C13=C13, C23=C23, k=k, a=param[self.a])

        return model 
    
    def __call__(self, param: dict[str, float], stress: np.ndarray, ep: np.ndarray):
        model = self.model(param)
        return model(stress, ep)


class HandlerVonMises(HandlerYield):
    def model(self, param: dict[str, float]):
        model = VonMises()
        return model 
    
    def __call__(self, param: dict[str, float], stress: np.ndarray, ep: np.ndarray):
        model = self.model(param)
        return model(stress, ep)
