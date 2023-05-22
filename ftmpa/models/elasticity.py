#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Elasticity models                                            ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************

import numpy as np
from abc import ABC, abstractmethod

class ElasticityModel(ABC):
    @abstractmethod
    def stress(self, strain: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def strain(self, stress: np.ndarray) -> np.ndarray:
        pass

class Hooke(ElasticityModel):
    def __init__(self, E, nu):
        # Cxx son las funciones de las constantes, en base
        # a la def plastica efectiva
        self.E = E
        self.nu = nu

    def stress(self, strain: np.ndarray) -> np.ndarray:
        # Esta funcion calcula la ley de hooke a una deformacion elastica
        # La entrada es el tensor de orden 2 de deformaciones:
        E = self.E
        nu = self.nu
        I = np.eye(3, dtype=float)

        stress = strain.copy()
        stress += nu/(1-2*nu)*np.einsum('...ij,kk->...ij', I, strain)
        stress *= E/(1+nu)

        return stress
    
    def strain(self, stress: np.ndarray) -> np.ndarray:
        # Esta funcion calcula la ley de hooke a una deformacion elastica
        # La entrada es el tensor de orden 2 de deformaciones:
        E = self.E
        nu = self.nu
        I = np.eye(3, dtype=float)

        strain = (1+nu)*stress.copy()
        strain -= nu*np.einsum('ij,...kk->...ij', I, stress)
        strain *= 1/E
        
        return strain