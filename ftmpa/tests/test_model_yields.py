from unittest import TestCase
import unittest
from ..models.yields import Cazacu, VonMises
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.interpolate import interp1d


class TestVonMises(TestCase):
    def test_vonmises_hidrostatic(self):
        ep = 0.0
        stress = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,1]], dtype=float)

        vm = VonMises()       
        stress_eq = vm(stress, ep)
        self.assertAlmostEqual(stress_eq, 0.0)


    def test_vonmises_multiples_loads(self):
        ep = None
        stress = np.array([[[1, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                        [[0, 0, 0],
                         [0, 1, 0],
                         [0, 0 ,0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,1]],
                        [[0.5, 0.5, 0],
                         [0.5, 0.5, 0],
                         [  0,   0, 0]],
                       ],
                       dtype=float)

        vm = VonMises()       
        stress_eq = vm(stress, ep)
        assert_almost_equal(stress_eq, 1.0)


    def test_vonmises_der_0g(self):
        ep = None
        stress = np.array([[1, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                       dtype=float)

        vm = VonMises()       
        der = vm.der(stress, ep)
        assert_almost_equal(der, [[1, 0, 0],
                                  [0, -0.5, 0],
                                  [0, 0, -0.5]])
        
    def test_vonmises_der_45g(self):
        ep = None
        stress = np.array([[0.5, 0.5, 0],
                        [0.5, 0.5, 0],
                        [0, 0 ,0]],
                       dtype=float)

        vm = VonMises()       
        der = vm.der(stress, ep)
        assert_almost_equal(der, [[0.25, 0.75, 0],
                                  [0.75, 0.25, 0],
                                  [0, 0, -0.5]])

    def test_vonmises_der_90g(self):
        ep = None
        stress = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,0]],
                       dtype=float)

        vm = VonMises()       
        der = vm.der(stress, ep)
        assert_almost_equal(der, [[-0.5, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -0.5]])

class TestCazacuIsotropic(TestCase):
    def setUp(self):
        ft1 = lambda ep: ep*0.0 + 1.0
        ft0 = lambda ep: ep*0.0
        self.const = {"C11": ft1, "C22": ft1, "C33": ft1,
                      "C44": ft1, "C55": ft1, "C66": ft1,
                      "C12": ft0, "C13": ft0, "C23": ft0,
                      "a": 2.0, "k": ft0,
                      }

    def test_cazacu_isotropic_hidrostatic(self):
        ep = 0.0
        stress = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,1]], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        self.assertAlmostEqual(stress_eq, 0.0)

    def test_cazacu_isotropic_hidrostatic_multiple_e(self):
        ep = np.array([0.0, 0.05])
        stress = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,1]], dtype=float)
        
        # sig = np.tile(sig, (...,2,1,1))
        stress = np.repeat(stress[..., np.newaxis, :, :], repeats=len(ep), axis=-3)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        assert_almost_equal(stress_eq, 0.0)

    def test_cazacu_isotropic_hidrostatic_multiple_stress(self):
        ep = 0.1
        stress = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,1]], dtype=float)
        
        stress = np.repeat(stress[..., np.newaxis, :, :], 3, axis=0)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        assert_almost_equal(stress_eq, 0.0)

    def test_cazacu_isotropic_0g_1(self):
        ep = 0.0
        stress = np.array([[1, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        self.assertAlmostEqual(stress_eq, 1.0)

    def test_cazacu_isotropic_0g_2(self):
        ep = 0.0
        stress = np.array([[2,  0, 0],
                        [0, -1, 0],
                        [0,  0 ,-1]], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        self.assertAlmostEqual(stress_eq, 3.0)

    def test_cazacu_isotropic_45g(self):
        ep = 0.0
        stress = np.array([[0.5, 0.5, 0],
                        [0.5, 0.5, 0],
                        [  0,   0, 0]], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        self.assertAlmostEqual(stress_eq, 1.0)

    def test_cazacu_isotropic_90g(self):
        ep = 0.0
        stress = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,0]], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        self.assertAlmostEqual(stress_eq, 1.0)

    def test_cazacu_isotropic_multiple_stress_ep(self):
        ep = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        stress = np.array([[[1, 0, 0],
                         [0, 1, 0],
                         [0, 0 ,1]],
                        [[1, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                        [[0.5, 0.5, 0],
                         [0.5, 0.5, 0],
                         [  0,   0, 0]],
                        [[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]],
                        [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 1]]
                       ],
                       dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        # print(sig_eq)
        assert_almost_equal(stress_eq, [0.0, 1, 1, 1, 1])


class TestCazacuIsotropicNonSym(TestCase):
    # Plasticity-Damage Couplings From Single Crystal to Polycrystalline Materials
    def setUp(self):
        ft1 = lambda ep: ep*0.0 + 1.0
        ft0 = lambda ep: ep*0.0
        self.const = {"C11": ft1, "C22": ft1, "C33": ft1,
                      "C44": ft1, "C55": ft1, "C66": ft1,
                      "C12": ft0, "C13": ft0, "C23": ft0,
                      "a": 2.0
                      }
        self.stress = np.array([[[1, 0, 0],
                              [0, 0, 0],
                              [0, 0 ,0]],
                             [[-1, 0, 0],
                              [0, 0, 0],
                              [0, 0 ,0]],
                            ], dtype=float)

    def test_cazacu_k_10(self):
        ep = 0.0
        kf = lambda ep: ep*0.0 - 1.0
        caz = Cazacu(**self.const, k=kf)       
        stress_eq = caz(self.stress, ep)
        self.assertAlmostEqual(stress_eq[0]/stress_eq[1], 2.0**0.5)

    def test_cazacu_k_04(self):
        ep = 0.0
        kf = lambda ep: ep*0.0 - 0.4
        caz = Cazacu(**self.const, k=kf)       
        stress_eq = caz(self.stress, ep)
        self.assertAlmostEqual(stress_eq[0]/stress_eq[1], 1.26, delta=0.005)

    def test_cazacu_k_02(self):
        ep = 0.0
        kf = lambda ep: ep*0.0 - 0.2
        caz = Cazacu(**self.const, k=kf)       
        stress_eq = caz(self.stress, ep)
        self.assertAlmostEqual(stress_eq[0]/stress_eq[1], 1.13, delta=0.01)

    def test_cazacu_k_00(self):
        ep = 0.0
        kf = lambda ep: ep*0.0 + 0.0
        caz = Cazacu(**self.const, k=kf)       
        stress_eq = caz(self.stress, ep)
        self.assertAlmostEqual(stress_eq[0]/stress_eq[1], 1.0)

class TestCazacuArticle2006(TestCase):
    # Orthotropic yield criterion for hexagonal closed packed metals
    # (doi:10.1016/j.ijplas.2005.06.001)
    # Mg–0.5% Fig9 and table 1
    def setUp(self):
        ep = np.array([0.01, 0.05, 0.1])
        self.const = {"C11": lambda ep: ep*0 + 1.0,
                      "C22": interp1d(ep, [0.9517, 0.9894, 1.4018]),
                      "C33": interp1d(ep, [0.4654, 0.1238, 0.7484]),
                      "C12": interp1d(ep, [0.4802, 0.3750, 0.6336]),
                      "C13": interp1d(ep, [0.2592, 0.0858, 0.2332]),
                      "C23": interp1d(ep, [0.2071, 0.0659, 0.5614]),
                      "C44": interp1d(ep, [1.0, 1.0, 1.0]),
                      "C55": interp1d(ep, [1.0, 1.0, 1.0]),
                      "C66": interp1d(ep, [1.0, 1.0, 1.0]),
                      "k": interp1d(ep, [0.3539, 0.2763, 0.0598]),
                      "a": 2.0,
                      }

    def test_cazacu_ep_01(self):
        ep = 0.01
        stress = np.array([[[186.0, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[-95.8, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, 167.7, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, -98.3, 0],
                        [0, 0 ,0]],
                        [[149.0, 0, 0],
                        [0, 149.0, 0],
                        [0, 0 ,0]],
                        ], dtype=float)
        
        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        assert_almost_equal(stress_eq, 186.0, decimal=-1)

    def test_cazacu_ep_05(self):
        ep = 0.05
        stress = np.array([[[208.0, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[-123.4, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, 203.1, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, -123.8, 0],
                        [0, 0 ,0]],
                        [[194.4, 0, 0],
                        [0, 194.4, 0],
                        [0, 0 ,0]],
                        ], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        assert_almost_equal(stress_eq, 208.0, decimal=-1)


    def test_cazacu_ep_10(self):
        ep = 0.10
        stress = np.array([[[213.6, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[-212.9, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, 221.4, 0],
                        [0, 0 ,0]],
                        [[0, 0, 0],
                        [0, -196.0, 0],
                        [0, 0 ,0]],
                        [[213.0, 0, 0],
                        [0, 213.0, 0],
                        [0, 0 ,0]],
                        ], dtype=float)

        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        assert_almost_equal(stress_eq, 213.6, decimal=-1)


class TestCazacuArticle2006(TestCase):
    # Experimental characterization and elasto-plastic modeling of the quasi-static
    # mechanical response of TA-6 V at room temperature
    # http://dx.doi.org/10.1016/j.ijsolstr.2011.01.011
    # Mg–0.5% Fig 9 and 10 and table 6
    def setUp(self):
        self.const = {"C11": lambda ep: ep*0 + 1.0,
                      "C22": lambda ep: ep*0 - 4.9101,
                      "C33": lambda ep: ep*0 - 0.5031,
                      "C12": lambda ep: ep*0 - 0.3408,
                      "C13": lambda ep: ep*0 - 2.6743,
                      "C23": lambda ep: ep*0 - 1.9254,
                      "C44": lambda ep: ep*0 + 1.0,
                      "C55": lambda ep: ep*0 + 1.0,
                      "C66": lambda ep: ep*0 - 3.7447,
                      "k": lambda ep: ep*0 + 0.047,
                      "a": 2.0,
                      }

    def test_cazacu_tensile_sheet(self):
        ep = 0.00
        stress = np.array([[[971.0, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                        [[461.4, 461.4, 0],
                         [461.4, 461.4, 0],
                         [0, 0 ,0]],
                        [[0, 0, 0],
                         [0, 991.0, 0],
                         [0, 0 ,0]],
                        ], dtype=float)
        
        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        assert_almost_equal(stress_eq, 971.0, decimal=-1)


    def test_cazacu_compresion_sheet(self):
        ep = 0.00
        stress = np.array([[[-904.65, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                        [[-448.2, -448.2, 0],
                         [-448.2, -448.2, 0],
                         [0, 0 ,0]],
                        [[0, 0, 0],
                         [0, -1085.8, 0],
                         [0, 0 ,0]],
                        ], dtype=float)
        
        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        assert_almost_equal(stress_eq, 971.0, decimal=-1)

    def test_cazacu_der_0g(self):
        ep = 0.0
        stress = np.array([[1, 0, 0],
                        [0, 0, 0],
                        [0, 0 ,0]],
                       dtype=float)

        caz = Cazacu(**self.const)       
        der = caz.der(stress, ep)
        # print("0g", der)
        assert_almost_equal(der[1,1]/der[2,2], 1.148, decimal=2)

    def test_cazacu_der_90g(self):
        ep = 0.0
        stress = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0 ,0]],
                       dtype=float)

        caz = Cazacu(**self.const)       
        der = caz.der(stress, ep)
        # print("90",der)
        assert_almost_equal(der[0,0]/der[2,2], 2.247, decimal=2)

    def test_cazacu_der_45g(self):
        ep = 0.0
        rot = np.array([[2**0.5/2, 2**0.5/2, 0.0],
                        [-(2**0.5/2), 2**0.5/2, 0.0],
                        [0.0, 0.0, 1.0],
                        ])
        stress = np.array([[1, 1, 0],
                        [1, 1, 0],
                        [0, 0 ,0]],
                       dtype=float)

        caz = Cazacu(**self.const)       
        der = caz.der(stress, ep)
        der2 = np.einsum('ij,jk,lk', rot, der, rot)
        # print("45",der2)
        # print("45 ", der2[1,1]/der[2,2])
        assert_almost_equal(der2[1,1]/der[2,2], 2.169, decimal=2)


class TestCazacuArticle_MG_AZ31(TestCase):
    #  Correlation between swift effects and tension–compression
    #  asymmetry in various polycrystalline materials
    #  https://doi-org.ezproxy.usach.cl/10.1016/j.jmps.2014.05.012
    def setUp(self):
        A0 = 315.4
        A1 = 140.6
        A2 = 16.3
        self.stress_fun = lambda ep: A0 - A1*np.exp(-A2*ep)
        self.const = {"C11": lambda ep: ep*0 + 1.0,
                      "C22": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 1.090,  1.090,  1.072,  1.099,  1.082], fill_value='extrapolate'),
                      "C33": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 3.342,  3.342,  2.905,  1.439,  0.885], fill_value='extrapolate'),
                      "C12": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [-0.168, -0.168, -0.595, -0.817, -0.762], fill_value='extrapolate'),
                      "C13": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 0.098,  0.098, -0.279, -0.516, -0.657], fill_value='extrapolate'),
                      "C23": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 0.243,  0.243, -0.096, -0.350, -0.509], fill_value='extrapolate'),
                      "C66": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 0.730,  0.730,  1.039,  1.128,  1.058], fill_value='extrapolate'),  # L44
                      "C44": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 7.300,  7.300,  10.29,  11.21,  10.12], fill_value='extrapolate'),  # L55
                      "C55": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [ 7.740,  7.740,  11.02,  11.95,  11.21], fill_value='extrapolate'),  # L66
                      "k": interp1d([0.00, 0.05, 0.06, 0.08, 0.10], [0.625, 0.625, 0.520, 0.215, 0.169], fill_value='extrapolate'),
                      "a": 2.0,
                      }
        self.ep = 0.6

    def test_cazacu_tensile_sheet(self):
        ep = self.ep
        stress_val = self.stress_fun(ep)
        stress = np.array([[stress_val, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                         dtype=float)
        
        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        print('tensile stres_eq', stress_eq, stress_val)
        # assert_almost_equal(sig_eq, 971.0, decimal=-1)


    def test_cazacu_compresion_sheet(self):
        ep = self.ep
        stress_val = self.stress_fun(ep)
        stress = np.array([[-stress_val, 0, 0],
                         [0, 0, 0],
                         [0, 0 ,0]],
                         dtype=float)
        
        caz = Cazacu(**self.const)       
        stress_eq = caz(stress, ep)
        print('compresion stress_eq', stress_eq)

if __name__ == '__main__':
    unittest.main()