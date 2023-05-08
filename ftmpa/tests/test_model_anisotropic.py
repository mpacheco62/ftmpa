from unittest import TestCase
import unittest
from pathlib import Path

# from ..handlers.elasticity import HandlerHooke
# from ..handlers.hardening import HandlerVoceMod
# from ..handlers.yields import HandlerCazacuVar, HandlerVonMises
# from ..handlers.anisotropic import AutoAnisotropicAsociated

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.linalg import expm
from scipy.interpolate import interp1d

from ..models.handlers.autoModels import AutoAnisotropicAsociated
from ..models.handlers.elasticity import HandlerHooke
from ..models.handlers.hardening import HandlerVoceMod
from ..models.handlers.yields import HandlerCazacuVar, HandlerVonMises

# from ..handlers.handlers import HandlerCazacuVar, HandlerHardTest, HandlerHooke, HandlerVoceMod, HandlerVonMises


DATA_COPPER_VULCAN = Path(__file__).parent.joinpath("data", "copper_iso_tc.txt")
DATA_VONMISES_T_VULCAN = Path(__file__).parent.joinpath("data", "VonMises_t.txt")
DATA_VONMISES_C_VULCAN = Path(__file__).parent.joinpath("data", "VonMises_c.txt")
DATA_MG_AZ21_RDt_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_RDt.txt")
DATA_MG_AZ21_RDc_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_RDc.txt")
DATA_MG_AZ21_TDt_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_TDt.txt")
DATA_MG_AZ21_TDc_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_TDc.txt")
DATA_MG_AZ21_NDt_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_NDt.txt")
DATA_MG_AZ21_NDc_VULCAN = Path(__file__).parent.joinpath("data", "AZ31Mg_NDc.txt")


class TestHandlerAnisotropicAsociated_VonMises(TestCase):
    @classmethod
    def setUpClass(cls):
        data_t = np.loadtxt(DATA_VONMISES_T_VULCAN)
        cls.stress_t = data_t[:, 0]
        cls.strain_t = data_t[:, 1:]

        data_c = np.loadtxt(DATA_VONMISES_C_VULCAN)
        cls.stress_c = data_c[:, 0]
        cls.strain_c = data_c[:, 1:]

    def setUp(self):
        self.param = {"sig0": 100.0, "k": 0.0, "q":0.0, "n": 1.0,
                 "E":10000.0, "nu": 0.25}
        self.hhard = HandlerVoceMod("sig0", "k", "q", "n")
        self.hyield = HandlerVonMises()
        self.helasticity = HandlerHooke("E", "nu")
        self.ani_model = AutoAnisotropicAsociated(self.hyield, self.hhard, self.helasticity)
        
    def test_vonmises_curve_0(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        assert_almost_equal(strain[-1], [[0.51, 0, 0], 
                                         [0, -0.2525, 0.0],
                                         [0, 0, -0.2525]])
        strain_p = strain - ani_model.h_elasticity.model(param).strain(stress)
        
        fun_stress = interp1d(strain[:,0,0], stress[:,0,0])
        fun_strain = interp1d(strain[:,0,0], strain, axis=0)

        dStress = self.stress_t - fun_stress(self.strain_t[:, 0]); dStress_max = np.amax(abs(dStress))
        dStrain_x = self.strain_t[:, 0] - fun_strain(self.strain_t[:, 0])[:, 0, 0]; dStrain_x_max = np.amax(abs(dStrain_x))
        dStrain_y = self.strain_t[:, 1] - fun_strain(self.strain_t[:, 0])[:, 1, 1]; dStrain_y_max = np.amax(abs(dStrain_y))
        dStrain_z = self.strain_t[:, 2] - fun_strain(self.strain_t[:, 0])[:, 2, 2]; dStrain_z_max = np.amax(abs(dStrain_z))

        self.assertLess(dStress_max, 2.0); self.assertLess(dStrain_x_max, 0.005); self.assertLess(dStrain_y_max, 0.005); self.assertLess(dStrain_z_max, 0.005)
        # print(self.strain_t[-1, 0], self.strain_t[-1, 1], self.strain_t[-1, 2])
        # print(strain[-1])
        # print("VonMises Maxs:", np.amax(abs(dStress)),
        #       np.amax(abs(dStrain_x)), np.amax(abs(dStrain_y)), np.amax(abs(dStrain_z)))
        # print("VonMises plastic Disp")
        # print(expm(strain_p[-1])- np.eye(3))


    def test_vonmises_curve_90(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        assert_almost_equal(strain[-1], [[-0.2525, 0, 0], 
                                         [0, 0.51, 0.0],
                                         [0, 0, -0.2525]])
        # print(expm(strain[-1])- np.eye(3))


    def test_vonmises_curve_45(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[0.5, 0.5, 0],
                            [0.5, 0.5, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        assert_almost_equal(strain[-1], [[0.12875, 0.38125, 0], 
                                         [0.38125, 0.12875, 0.0],
                                         [0, 0, -0.2525]])
        # print(expm(strain[-1])- np.eye(3))


class TestHandlerAnisotropicAsociated_Cazacu_Copper(TestCase):
    #  Correlation between swift effects and tension–compression
    #  asymmetry in various polycrystalline materials
    #  https://doi-org.ezproxy.usach.cl/10.1016/j.jmps.2014.05.012
    @classmethod
    def setUpClass(cls):
        data = np.loadtxt(DATA_COPPER_VULCAN)
        cls.e_t = data[:,0]
        cls.s_t = data[:,1]
        cls.e_c = data[:,2]
        cls.s_c = data[:,3]

    def setUp(self):
        self.param = {"sig0": 29.5, "k": 0.0, "q":224.3, "n": 10.92,
                      "E":125.0E3, "nu": 0.34,
                      "C1":1.0, "C2":0.0, "a": 2.0,
                      "k_00": 0.0, "k_03": -0.084, "k_06": -0.127,
                      "k_09": -0.137, "k_12": -0.137,
                     }
        self.hhard = HandlerVoceMod("sig0", "k", "q", "n")
        self.hyield = HandlerCazacuVar(C11 = [("C1", 0.0), ("C1", 1.0)],
                                       C22 = [("C1", 0.0), ("C1", 1.0)],
                                       C33 = [("C1", 0.0), ("C1", 1.0)],
                                       C44 = [("C1", 0.0), ("C1", 1.0)],
                                       C55 = [("C1", 0.0), ("C1", 1.0)],
                                       C66 = [("C1", 0.0), ("C1", 1.0)],
                                       C12 = [("C2", 0.0), ("C2", 1.0)],
                                       C13 = [("C2", 0.0), ("C2", 1.0)],
                                       C23 = [("C2", 0.0), ("C2", 1.0)],
                                       k = [("k_00", 0.0), ("k_03", 0.03),
                                            ("k_06", 0.06), ("k_09", 0.09),
                                            ("k_12", 0.12),
                                            ],
                                       a = "a"
                                       )
        self.helasticity = HandlerHooke("E", "nu")
        self.ani_model = AutoAnisotropicAsociated(self.hyield,
                                                     self.hhard,
                                                     self.helasticity,
                                                     strain_plas_max=0.3)
        
    def test_cazacu_isotropic_asymmetry_copper(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        # m_elas = self.helasticity.model(self.param)
        # strain = strain - m_elas.strain(sig)
        fun_t = interp1d(strain[:,0,0], stress[:,0,0])
        dStress = fun_t(self.e_t) - self.s_t
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, 1.0)

    def test_cazacu_isotropic_asymmetry_copper2(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[-1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)
        fun_t = interp1d(strain[:,0,0], stress[:,0,0])
        dStress = fun_t(self.e_c) - self.s_c
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, 1.0)



class TestHandlerAnisotropicAsociated_Cazacu_MG_AZ31(TestCase):
    #  Correlation between swift effects and tension–compression
    #  asymmetry in various polycrystalline materials
    #  https://doi-org.ezproxy.usach.cl/10.1016/j.jmps.2014.05.012
    @classmethod
    def setUpClass(cls):
        dataRDt = np.loadtxt(DATA_MG_AZ21_RDt_VULCAN)
        cls.stressRDt = dataRDt[:, 0]
        cls.strainRDt = dataRDt[:, 1:]

        dataRDc = np.loadtxt(DATA_MG_AZ21_RDc_VULCAN)
        cls.stressRDc = dataRDc[:, 0]
        cls.strainRDc = dataRDc[:, 1:]
        
        dataTDt = np.loadtxt(DATA_MG_AZ21_TDt_VULCAN)
        cls.stressTDt = dataTDt[:, 0]
        cls.strainTDt = dataTDt[:, 1:]
        
        dataTDc = np.loadtxt(DATA_MG_AZ21_TDc_VULCAN)
        cls.stressTDc = dataTDc[:, 0]
        cls.strainTDc = dataTDc[:, 1:]
        
        dataNDt = np.loadtxt(DATA_MG_AZ21_NDt_VULCAN)
        cls.stressNDt = dataNDt[:, 0]
        cls.strainNDt = dataNDt[:, 1:]
        
        dataNDc = np.loadtxt(DATA_MG_AZ21_NDc_VULCAN)
        cls.stressNDc = dataNDc[:, 0]
        cls.strainNDc = dataNDc[:, 1:]

        cls.strain_criteria = 0.01
        cls.stress_criteria = 4.0


    def setUp(self):
        self.param = {"sig0": 174.8, "k": 0.0, "q": 140.6, "n": 16.3,
                      "E":45.0E3, "nu": 0.3,
                      "1":1.0,
                      "L22_05": 1.090, "L22_06": 1.072, "L22_08": 1.099, "L22_10": 1.082,
                      "L33_05": 3.342, "L33_06": 2.905, "L33_08": 1.439, "L33_10": 0.885,
                      "L12_05":-0.168, "L12_06":-0.595, "L12_08":-0.817, "L12_10":-0.762,
                      "L13_05": 0.098, "L13_06":-0.279, "L13_08":-0.516, "L13_10":-0.657,
                      "L23_05": 0.243, "L23_06":-0.096, "L23_08":-0.350, "L23_10":-0.509,
                      "L44_05": 0.730, "L44_06": 1.039, "L44_08": 1.128, "L44_10": 1.058,
                      "L55_05": 7.300, "L55_06": 10.20, "L55_08": 11.21, "L55_10": 10.12,
                      "L66_05": 7.740, "L66_06": 11.02, "L66_08": 11.95, "L66_10": 11.21,
                      "k_05": -0.625, "k_06": -0.520, "k_08": -0.215, "k_10": -0.169,
                      "a": 2.0,
                     }
        self.hhard = HandlerVoceMod("sig0", "k", "q", "n")
        self.hyield = HandlerCazacuVar(C11 = [("1", 0.0), ("1", 1.0)],
                                       C22 = [("L22_05", 0.00), ("L22_05", 0.05), ("L22_06", 0.06), ("L22_08", 0.08), ("L22_10", 0.10), ("L22_10", 0.50)],
                                       C33 = [("L33_05", 0.00), ("L33_05", 0.05), ("L33_06", 0.06), ("L33_08", 0.08), ("L33_10", 0.10), ("L33_10", 0.50)],
                                       C44 = [("L55_05", 0.00), ("L55_05", 0.05), ("L55_06", 0.06), ("L55_08", 0.08), ("L55_10", 0.10), ("L55_10", 0.50)],
                                       C55 = [("L66_05", 0.00), ("L66_05", 0.05), ("L66_06", 0.06), ("L66_08", 0.08), ("L66_10", 0.10), ("L66_10", 0.50)],
                                       C66 = [("L44_05", 0.00), ("L44_05", 0.05), ("L44_06", 0.06), ("L44_08", 0.08), ("L44_10", 0.10), ("L44_10", 0.50)],
                                       C12 = [("L12_05", 0.00), ("L12_05", 0.05), ("L12_06", 0.06), ("L12_08", 0.08), ("L12_10", 0.10), ("L12_10", 0.50)],
                                       C13 = [("L13_05", 0.00), ("L13_05", 0.05), ("L13_06", 0.06), ("L13_08", 0.08), ("L13_10", 0.10), ("L13_10", 0.50)],
                                       C23 = [("L23_05", 0.00), ("L23_05", 0.05), ("L23_06", 0.06), ("L23_08", 0.08), ("L23_10", 0.10), ("L23_10", 0.50)],
                                       k = [("k_05", 0.00), ("k_05", 0.05), ("k_06", 0.06), ("k_08", 0.08), ("k_10", 0.10), ("k_10", 0.50)],
                                       a = "a"
                                       )
        self.helasticity = HandlerHooke("E", "nu")
        self.ani_model = AutoAnisotropicAsociated(self.hyield,
                                                     self.hhard,
                                                     self.helasticity,
                                                     strain_plas_max=0.8,
                                                     npoints=50000)
        
    def test_cazacu_MG_AZ31_RDt(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        # m_elas = self.helasticity.model(self.param)
        # strain = strain - m_elas.strain(sig)
        fun_stress = interp1d(strain[:,0,0], stress[:,0,0])
        fun_strain = interp1d(strain[:,0,0], strain, axis=0)
        

        dStress = fun_stress(self.strainRDt[:, 0]) - self.stressRDt
        dStress_max = np.amax(abs(dStress))
        # print('MG RDt:', dStress_max, fun_stress(self.strainRDt[-1, 0]), self.stressRDt[-1])
        self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainRDt[:, 0])[:, 0, 0] - self.strainRDt[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        # print(dStrain_x_max)
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainRDt[:, 0])[:, 1, 1] - self.strainRDt[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        # print(dStrain_y_max, fun_strain(self.strainRDt[:, 0])[-1, 1], self.strainRDt[-1, 1])
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainRDt[:, 0])[:, 2, 2] - self.strainRDt[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        # print(dStrain_z_max)
        self.assertLess(dStrain_z_max, self.strain_criteria)

    def test_cazacu_MG_AZ31_RDc(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[-1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        fun_stress = interp1d(strain[:,0,0], stress[:,0,0])
        fun_strain = interp1d(strain[:,0,0], strain, axis=0)    

        dStress = fun_stress(self.strainRDc[:, 0]) - self.stressRDc
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainRDc[:, 0])[:, 0, 0] - self.strainRDc[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainRDc[:, 0])[:, 1, 1] - self.strainRDc[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainRDc[:, 0])[:, 2, 2] - self.strainRDc[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        self.assertLess(dStrain_z_max, self.strain_criteria)


    def test_cazacu_MG_AZ31_TDt(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        fun_stress = interp1d(strain[:,1,1], stress[:,1,1])
        fun_strain = interp1d(strain[:,1,1], strain, axis=0)
        

        dStress = fun_stress(self.strainTDt[:, 1]) - self.stressTDt
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainTDt[:, 1])[:, 0, 0] - self.strainTDt[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainTDt[:, 1])[:, 1, 1] - self.strainTDt[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainTDt[:, 1])[:, 2, 2] - self.strainTDt[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        self.assertLess(dStrain_z_max, self.strain_criteria)

    def test_cazacu_MG_AZ31_TDc(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[0, 0, 0],
                            [0,-1, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        fun_stress = interp1d(strain[:,1,1], stress[:,1,1])
        fun_strain = interp1d(strain[:,1,1], strain, axis=0)    

        dStress = fun_stress(self.strainTDc[:, 1]) - self.stressTDc
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainTDc[:, 1])[:, 0, 0] - self.strainTDc[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainTDc[:, 1])[:, 1, 1] - self.strainTDc[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainTDc[:, 1])[:, 2, 2] - self.strainTDc[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        self.assertLess(dStrain_z_max, self.strain_criteria)


    def test_cazacu_MG_AZ31_NDt(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        fun_stress = interp1d(strain[:,2,2], stress[:,2,2])
        fun_strain = interp1d(strain[:,2,2], strain, axis=0)
        

        # print('last stress', self.stressNDt[-1], fun_stress(self.strainNDt[:, 2])[-1])
        dStress = fun_stress(self.strainNDt[:, 2]) - self.stressNDt
        dStress_max = np.amax(abs(dStress))
        # self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainNDt[:, 2])[:, 0, 0] - self.strainNDt[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainNDt[:, 2])[:, 1, 1] - self.strainNDt[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainNDt[:, 2])[:, 2, 2] - self.strainNDt[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        self.assertLess(dStrain_z_max, self.strain_criteria)

    def test_cazacu_MG_AZ31_NDc(self):
        ani_model = self.ani_model
        param = self.param
        stress_ref = np.array([[-1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=float)

        stress, strain = ani_model.calc(param, stress_ref)

        fun_stress = interp1d(strain[:,0,0], stress[:,0,0])
        fun_strain = interp1d(strain[:,0,0], strain, axis=0)    

        dStress = fun_stress(self.strainRDc[:, 0]) - self.stressRDc
        dStress_max = np.amax(abs(dStress))
        self.assertLess(dStress_max, self.stress_criteria)

        dStrain_x = fun_strain(self.strainRDc[:, 0])[:, 0, 0] - self.strainRDc[:, 0]
        dStrain_x_max = np.amax(abs(dStrain_x))
        self.assertLess(dStrain_x_max, self.strain_criteria)

        dStrain_y = fun_strain(self.strainRDc[:, 0])[:, 1, 1] - self.strainRDc[:, 1]
        dStrain_y_max = np.amax(abs(dStrain_y))
        self.assertLess(dStrain_y_max, self.strain_criteria)

        dStrain_z = fun_strain(self.strainRDc[:, 0])[:, 2, 2] - self.strainRDc[:, 2]
        dStrain_z_max = np.amax(abs(dStrain_z))
        self.assertLess(dStrain_z_max, self.strain_criteria)



   



# class TestHandlerAnisotropicAsociated_Cazacu_Copper(TestCase):
#     #  --------
#     #  ---------
#     # ------------
#     def setUp(self):
#         self.param = {"C1_h": 236.0, "C2_h": 245.0, "n_h": 0.539,
#                       "E":110.0E3, "nu": 0.3,
#                       "1":1.0, "0":0.0,
#                       "a": 2.0,
#                       "C33_00": 1.249, "C33_02": 2.023, "C33_04": 2.185, "C33_10": 1.693, "C33_20": 1.167,
#                       "C13_00":-0.050, "C13_02": 0.523, "C13_04": 0.687, "C13_10": 0.200, "C13_20":-0.307,
#                       "C12_00":1.0 + 0.050, "C12_02": -1 - 0.523, "C12_04": -1 - 0.687, "C12_10": -1 - 0.200, "C12_20":-1-0.307,
#                       "k_00": -0.077, "k_02": -0.071, "k_04": -0.070, "k_10": -0.087, "k_20": -0.224,
#                      }
#         self.hhard = HandlerHardTest(C1 = "C1_h", C2 = "C2_h", n = "n_h")
#         self.hyield = HandlerCazacuVar(C11 = [("1", 0.0), ("1", 1.0)],
#                                        C22 = [("1", 0.0), ("1", 1.0)],
#                                        C33 = [("C33_00", 0.0), ("C33_02", 0.02), ("C33_04", 0.04), ("C33_10", 0.10), ("C33_20", 0.20),],
#                                        C44 = [("1", 0.0), ("1", 1.0)],
#                                        C55 = [("1", 0.0), ("1", 1.0)],
#                                        C66 = [("1", 0.0), ("1", 1.0)],
#                                        C12 = [("C12_00", 0.0), ("C12_02", 0.02), ("C12_04", 0.04), ("C12_10", 0.10), ("C12_20", 0.20),],
#                                        C13 = [("C13_00", 0.0), ("C13_02", 0.02), ("C13_04", 0.04), ("C13_10", 0.10), ("C13_20", 0.20),],
#                                        C23 = [("C13_00", 0.0), ("C13_02", 0.02), ("C13_04", 0.04), ("C13_10", 0.10), ("C13_20", 0.20),],
#                                        k = [("k_00", 0.0), ("k_02", 0.02), ("k_04", 0.04), ("k_10", 0.10), ("k_20", 0.20),],
#                                        a = "a"
#                                        )
#         self.helasticity = HandlerHooke("E", "nu")
#         self.ani_model = HandlerAnisotropicAsociated(self.hyield,
#                                                      self.hhard,
#                                                      self.helasticity,
#                                                      strain_plas_max=0.22)
        
#     def test_cazacu_123(self):
#         ani_model = self.ani_model
#         param = self.param
#         sig_ref = np.array([[1, 0, 0],
#                             [0, 0, 0],
#                             [0, 0, 0]], dtype=float)

#         sig, strain = ani_model.calc(param, sig_ref)

#         m_elas = self.helasticity.model(self.param)
#         # strain = strain - m_elas.strain(sig)
#         print(strain[-1,0,0], sig[-1,0,0])
#         print(strain[1,0,0], sig[1,0,0])



        
if __name__ == '__main__':
    unittest.main()