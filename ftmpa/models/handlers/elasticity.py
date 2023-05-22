#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Handler elasticity model                                     ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************

from abc import ABC, abstractmethod
import numpy as np

from ..elasticity import ElasticityModel, Hooke

class HandlerElasticity(ABC):
    @abstractmethod
    def stress(self, param: dict[str, float], strain: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def strain(self, param: dict[str, float], stress: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def model(self, param: dict[str, float]) -> ElasticityModel:
        pass


class HandlerHooke(HandlerElasticity):
    E: str
    nu: str

    def __init__(self, E: str, nu: str):
        self.E = E
        self.nu = nu

    def model(self, param: dict[str, float]):
        E = param[self.E]
        nu = param[self.nu]
        model = Hooke(E=E, nu=nu)
        return model

    def stress(self, param: dict[str, float], strain: np.ndarray):
        model = self.model(param)
        return model.stress(strain)

    def strain(self, param: dict[str, float], stress: np.ndarray):
        model = self.model(param)
        return model.strain(stress)
