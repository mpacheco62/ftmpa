#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Yield models                                                 ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************

from abc import ABC, abstractmethod
from ..common import from_voigt, to_voigt
import numpy as np
from typing import Callable
# ***********************************************************************

class YieldModel(ABC):
    @abstractmethod
    def __call__(self, stress: np.ndarray, ep: np.ndarray) -> np.ndarray:
        pass

    def der(self, stress: np.ndarray, ep: np.ndarray, eps=1.0e-6) -> np.ndarray:
        stress = to_voigt(stress)
        eps1 = np.zeros_like(stress)
        eps2 = np.zeros_like(stress)

        for i in range(6):
            stress_ii = stress.copy()
            stress_ii[..., i] += eps
            if i>=3: stress_ii[..., i] -= eps/2
            stress_ii = from_voigt(stress_ii)
            eps1[..., i] = self(stress_ii, ep)

        for i in range(6):
            stress_ii = stress.copy()
            stress_ii[..., i] -= eps
            if i>=3: stress_ii[..., i] += eps/2
            stress_ii = from_voigt(stress_ii)
            eps2[..., i] = self(stress_ii, ep)
        
        der = (eps1 - eps2) / (2*eps)
        der = from_voigt(der)
        return der

class Cazacu(YieldModel):
    C11: Callable[[np.ndarray], np.ndarray]
    C22: Callable[[np.ndarray], np.ndarray]
    C33: Callable[[np.ndarray], np.ndarray]
    C44: Callable[[np.ndarray], np.ndarray]
    C55: Callable[[np.ndarray], np.ndarray]
    C66: Callable[[np.ndarray], np.ndarray]
    C12: Callable[[np.ndarray], np.ndarray]
    C13: Callable[[np.ndarray], np.ndarray]
    C23: Callable[[np.ndarray], np.ndarray]
    k: Callable[[np.ndarray], np.ndarray]
    a: int

    def __init__(self, 
                 C11: Callable[[np.ndarray], np.ndarray],
                 C22: Callable[[np.ndarray], np.ndarray],
                 C33: Callable[[np.ndarray], np.ndarray],
                 C44: Callable[[np.ndarray], np.ndarray],
                 C55: Callable[[np.ndarray], np.ndarray],
                 C66: Callable[[np.ndarray], np.ndarray],
                 C12: Callable[[np.ndarray], np.ndarray],
                 C13: Callable[[np.ndarray], np.ndarray],
                 C23: Callable[[np.ndarray], np.ndarray],
                 k: Callable[[np.ndarray], np.ndarray],
                 a: int):
        # Cxx son las funciones de las constantes, en base
        # a la def plastica efectiva
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


    def __call__(self, stress: np.ndarray, ep: np.ndarray) -> np.ndarray:
        # Esta funcion calcula la funcion de cazacu para un sigma a un ep
        # La entrada esta en notacion tensorial
        stress = np.array(stress); ep = np.array(ep)
        stress = to_voigt(stress)

        # Obtiene los parametros
        C11 = self.C11(ep)
        C22 = self.C22(ep)
        C33 = self.C33(ep)
        C44 = self.C44(ep)
        C55 = self.C55(ep)
        C66 = self.C66(ep)
        C12 = self.C12(ep)
        C13 = self.C13(ep)
        C23 = self.C23(ep)
        k = self.k(ep)
        # print("K:",k)
        a = self.a

        # Calcula el desviador
        mean = (stress[..., 0] + stress[..., 1] + stress[..., 2]).copy()/3.0
        dev = stress
        dev[..., :3] -=  mean[..., np.newaxis]

        devShape = list(dev.shape)
        devShape[-1] = 6; devShape.append(6)
        
        C = np.zeros(devShape)

        C[..., 0, 0] = C11
        C[..., 0, 1] = C12
        C[..., 0, 2] = C13

        C[..., 1, 0] = C12
        C[..., 1, 1] = C22
        C[..., 1, 2] = C23

        C[..., 2, 0] = C13
        C[..., 2, 1] = C23
        C[..., 2, 2] = C33

        C[..., 3, 3] = C44
        C[..., 4, 4] = C55
        C[..., 5, 5] = C66

        # Calcula el nuevo esfuerzo \Sigma
        # SIG = C @ dev
        # print('dev', dev)
        SIG = np.einsum('...jk,...k->...j', C, dev)
        SIG = from_voigt(SIG)

        # Calcula los autovalores
        # print('SIG', SIG)
        try:
            SIG_P = np.linalg.eigvals(SIG)

            # Calcula la funcion de cazacu
            # print("a:", a)
            # print("phi: ", abs(SIG_P[..., 0])-k*SIG_P[..., 0], (abs(SIG_P[..., 0])-k*SIG_P[..., 0]).shape)
            # print("phi: ", (abs(SIG_P[..., 0])-k*SIG_P[..., 0])**a)
            phi_s = ((abs(SIG_P[..., 0])-k*SIG_P[..., 0])**a +
                    (abs(SIG_P[..., 1])-k*SIG_P[..., 1])**a +
                    (abs(SIG_P[..., 2])-k*SIG_P[..., 2])**a)**(1.0/a)

            # Calcula el "normalizado" para el esfuerzo
            phi1 = 2.0/3.0*C11 - 1.0/3.0*C12 - 1.0/3.0*C13
            phi2 = 2.0/3.0*C12 - 1.0/3.0*C22 - 1.0/3.0*C23
            phi3 = 2.0/3.0*C13 - 1.0/3.0*C23 - 1.0/3.0*C33
            Bvar = (1.0/((abs(phi1)-k*phi1)**a + (abs(phi2)-k*phi2)**a + (abs(phi3)-k*phi3)**a))**(1.0/a)

            phi_s = Bvar*phi_s

            # phi = 1/phi_s
            # Devuelve el nuevo equivalente
        except np.linalg.LinAlgError as err:
            print(dev)
            print(SIG)
            # if 'Array must not contain infs or NaNs' in str(err):
            #     phi_s = np.nan
            # else:
            raise err
        return phi_s


class VonMises(YieldModel):
    def __call__(self, stress: np.ndarray, ep: np.ndarray) -> np.ndarray:
        # Esta funcion calcula la funcion de vonmises para un sigma a un ep
        # La entrada esta en notacion tensorial:

        # Calcula el desviador
        stress = to_voigt(stress)

        mean = (stress[..., 0] + stress[..., 1] + stress[..., 2]).copy()/3.0
        dev = stress.copy()
        dev[..., :3] -=  mean[..., np.newaxis]

        dev_T = from_voigt(dev)

        eq = np.einsum('...ij, ...ij', dev_T, dev_T)
        eq = (3.0/2.0*eq)**0.5

        return eq