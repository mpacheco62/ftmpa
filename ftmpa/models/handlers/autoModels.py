#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************************
# ******    Anisotropic model complete                                   ******
# ******    AUTH: Matías Pacheco                                         ******
# *****************************************************************************

from abc import ABC, abstractmethod
from ...common import from_voigt, to_voigt
from .elasticity import HandlerElasticity
from .hardening import HandlerHardening
from .yields import HandlerYield
import numpy as np
from scipy.integrate import cumulative_trapezoid

class AutoModels(ABC):
    @abstractmethod
    def calc(self, param: dict[str, float], stress_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class AutoElasticity(AutoModels):
    """Automatic generation of pair stress-strain states for elastic models 

    It should contain a handler elasticity model

    """
    h_elasticity: HandlerElasticity

    def __init__(self,
                 h_elasticity: HandlerElasticity,
                 npoints: int = 5,
                 stress_max: float = 1e9
                 ):
        self.h_elasticity = h_elasticity
        self.npoints = npoints
        self.stress_max = stress_max

    def calc(self, param: dict[str, float], stress_ref: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
        """! Calc the stress-strain curve for elastic model
        """

        stress_ref = np.repeat(stress_ref[..., np.newaxis, :, :], repeats=self.npoints, axis=-3)
        multiplier = np.linspace(start=-self.stress_max, stop=self.stress_max, num=self.npoints)
        stress = np.einsum("i,...ijk->...ijk", multiplier, stress_ref)

        elastic_model = self.h_elasticity.model(param)
        strain = elastic_model.strain(stress)
        
        return stress, strain
    


class AutoElastoPlasticAsociatedStrainDependent(AutoModels):
    """Automatic generation of pair stress-strain states for anisotropic model  

    It should contain a handler for elasticity model, hardening model and a yield surface.
    Also the definition of a maximun effective plastic strain for the making of the stress-strain curves. 
    """

    h_yield: HandlerYield
    h_hardening: HandlerHardening
    h_elasticity: HandlerElasticity
    strain_plas_max: float
    npoints: int

    def __init__(self,
                 h_yield: HandlerYield,
                 h_hardening: HandlerHardening,
                 h_elasticity: HandlerElasticity,
                 strain_plas_max: float = 0.5,
                 npoints: int = 10000):
        self.h_yield = h_yield
        self.h_hardening = h_hardening
        self.h_elasticity = h_elasticity
        self.strain_plas_max = strain_plas_max
        self.npoints = npoints

    def calc_one_direction(self, param: dict[str, float], stress_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """! Calc the stress-strain curve for a reference stress stress_ref
        The program assumes an range of effective plastic deformation \f$\bar\epsilon\f$, 
        and also assumes that the stress \f$\stress\f$ at any given time is proportional
        to a reference stress \f$\sigma_{ref}\f$.
        
        \f[\sigma = \beta \sigma \f]
        
        Additionally, it is assumed that the yield function \f$\sigma_{eq}(\sigma)\f$
        is homogeneous of degree 1, and that the hardening function is \f$Y(\bar\epsilon)\f$.

        \f[ Y(\bar\epsilon) = \sigma_{eq}(\sigma) = \sigma_{eq}(\beta\sigma_{ref}) = \beta\sigma_{eq}(\sigma_{ref})\f]
        \f[ \beta = \frac{Y(\bar\epsilon)}{\sigma_{eq}(\sigma_{ref})} \f]
        """

        stress_ref = np.repeat(stress_ref[..., np.newaxis, :, :], repeats=self.npoints, axis=-3)

        #  Range of equivalent plastic strain
        ep_eq = np.linspace(0.0, self.strain_plas_max, self.npoints)

        #  Defining the models with parameters
        yield_model = self.h_yield.model(param)
        hardening_model = self.h_hardening.model(param)
        elastic_model = self.h_elasticity.model(param)
        
        #  Calculate the stress state based on beta
        stress_eq = hardening_model(ep_eq)
        stress_eq_unit = yield_model(stress_ref, ep_eq)
        beta = stress_eq/stress_eq_unit
        stress = np.einsum("i,...ijk->...ijk", beta, stress_ref)

        #  Calculate the plastic strain
        der = yield_model.der(stress_ref, ep_eq)
        der = to_voigt(der)
        # to_integrate = delta_ep_eq*der
        to_integrate = der
        ep = cumulative_trapezoid(y=to_integrate, x=ep_eq, axis=0, initial=0)
        ep = from_voigt(ep)

        strain_elastic = elastic_model.strain(stress)
        # strain = ep
        strain = ep + strain_elastic
        zero = np.zeros((3,3))
       
        strain = np.insert(strain, 0, zero, axis=-3)
        stress = np.insert(stress, 0, zero, axis=-3)

        return stress, strain
    

    def calc(self, param: dict[str, float], stress_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for key, item in param.items():
            if np.isnan(item):
                return np.nan, np.nan
        stress_t, strain_t = self.calc_one_direction(param, stress_ref)
        stress_c, strain_c = self.calc_one_direction(param, -stress_ref)
        stress_c = stress_c[...,::-1, :, :]
        strain_c = strain_c[...,::-1, :, :]
        stress = np.concatenate((stress_c, stress_t), axis=-3)
        strain = np.concatenate((strain_c, strain_t), axis=-3)
        return stress, strain
    

class AutoElastoPlasticAsociated(AutoModels):
    """Automatic generation of pair stress-strain states for anisotropic model  

    It should contain a handler for elasticity model, hardening model and a yield surface.
    Also the definition of a maximun effective plastic strain for the making of the stress-strain curves. 
    """

    h_yield: HandlerYield
    h_hardening: HandlerHardening
    h_elasticity: HandlerElasticity
    strain_plas_max: float
    npoints: int

    def __init__(self,
                 h_yield: HandlerYield,
                 h_hardening: HandlerHardening,
                 h_elasticity: HandlerElasticity,
                 strain_plas_max: float = 0.5,
                 npoints: int = 1000):
        self.h_yield = h_yield
        self.h_hardening = h_hardening
        self.h_elasticity = h_elasticity
        self.strain_plas_max = strain_plas_max
        self.npoints = npoints

    def calc_one_direction(self, param: dict[str, float], stress_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """! Calc the stress-strain curve for a reference stress stress_ref
        The program assumes an range of effective plastic deformation \f$\bar\epsilon\f$, 
        and also assumes that the stress \f$\stress\f$ at any given time is proportional
        to a reference stress \f$\sigma_{ref}\f$.
        
        \f[\sigma = \beta \sigma \f]
        
        Additionally, it is assumed that the yield function \f$\sigma_{eq}(\sigma)\f$
        is homogeneous of degree 1, and that the hardening function is \f$Y(\bar\epsilon)\f$.

        \f[ Y(\bar\epsilon) = \sigma_{eq}(\sigma) = \sigma_{eq}(\beta\sigma_{ref}) = \beta\sigma_{eq}(\sigma_{ref})\f]
        \f[ \beta = \frac{Y(\bar\epsilon)}{\sigma_{eq}(\sigma_{ref})} \f]
        """

        stress_ref_original = stress_ref.copy()

        stress_ref = np.repeat(stress_ref[..., np.newaxis, :, :], repeats=self.npoints, axis=-3)

        #  Range of equivalent plastic strain
        ep_eq = np.linspace(0.0, self.strain_plas_max, self.npoints)

        #  Defining the models with parameters
        yield_model = self.h_yield.model(param)
        hardening_model = self.h_hardening.model(param)
        elastic_model = self.h_elasticity.model(param)
        
        #  Calculate the stress state based on beta
        stress_eq = hardening_model(ep_eq)
        stress_eq_unit = yield_model(stress_ref, ep_eq)
        beta = stress_eq/stress_eq_unit
        stress = np.einsum("i,...ijk->...ijk", beta, stress_ref)

        #  Calculate the plastic strain
        der = yield_model.der(stress_ref_original, 0)
        ep = np.einsum('i, jk -> ijk', ep_eq, der)

        strain_elastic = elastic_model.strain(stress)
        # strain = ep
        strain = ep + strain_elastic
        zero = np.zeros((3,3))
       
        strain = np.insert(strain, 0, zero, axis=-3)
        stress = np.insert(stress, 0, zero, axis=-3)

        return stress, strain
    

    def calc(self, param: dict[str, float], stress_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for key, item in param.items():
            if np.isnan(item):
                return np.nan, np.nan
        stress_t, strain_t = self.calc_one_direction(param, stress_ref)
        stress_c, strain_c = self.calc_one_direction(param, -stress_ref)
        stress_c = stress_c[...,::-1, :, :]
        strain_c = strain_c[...,::-1, :, :]
        stress = np.concatenate((stress_c, stress_t), axis=-3)
        strain = np.concatenate((strain_c, strain_t), axis=-3)
        return stress, strain