import numpy as np
from scipy.spatial.transform import Rotation as R


# def from_real_to_plastic(data, young):
#     if isinstance(data, dict):
#         for name, values in data.items():
#             eps = values[:, 0]
#             sig = values[:, 1]
#             new_eps = eps - sig/young
#             data[name][:, 0] = new_eps
#     else:
#         eps = data[:, 0]
#         sig = data[:, 1]
#         new_eps = eps - sig/young
#         data[:, 0] = new_eps

#     return data


def to_voigt(stress: np.ndarray):
    # Pasa un tensor de orden 2 a notación de voigt
    if stress.ndim == 2:
        stressV = np.array([stress[0, 0],
                            stress[1, 1],
                            stress[2, 2],
                            stress[1, 2],
                            stress[0, 2],
                            stress[0, 1]
                            ])
    else:
        stressV = stress[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
    return stressV


def from_voigt(stress):
    # Pasa de notación de voigt a un tensor de orden 2
    if stress.ndim == 1:
        stress_n = np.array([[stress[0], stress[5], stress[4]],
                             [stress[5], stress[1], stress[3]],
                             [stress[4], stress[3], stress[2]],
                             ])
    else:
        stress_n = stress[..., [0, 5, 4, 5, 1, 3, 4, 3, 2]]
        stressShape = list(stress.shape); stressShape[-1] = 3; stressShape.append(3)
        stress_n = stress_n.reshape(stressShape)
    return stress_n


def rot_sheet_angle(angle):
    return R.from_euler('z', angle, degrees=True).as_matrix() 

def load_sheet_angle(angle):
    stress = np.array([[1,0,0],
                    [0,0,0],
                    [0,0,0]
                    ], dtype=float)
    rot = rot_sheet_angle(angle)
    return np.einsum('ij,jk,lk', rot, stress, rot)

def load_stress_voigt(text: str):
    stress = np.zeros((3, 3), dtype=float)
    stress_voigt = np.array([float(i) for i in text.strip().split()], dtype=float)
    stress = stress_voigt[[0, 5, 4,
                     5, 1, 3,
                     4, 3, 2]].reshape((3,3))
    return stress

# def rot_ang(stress, alpha, inverse=False):
#     # Rota el tensor de esfuerzo en el plano desde alpha a la conf global
#     alpha = alpha*np.pi/180.0
#     rot = np.array([[np.cos(alpha), -np.sin(alpha), 0.0],
#                     [np.sin(alpha), np.cos(alpha), 0.0],
#                     [0.0, 0.0, 1.0]
#                     ])

#     if not inverse:
#         # stress_rot = rot @ stress @ rot.T
#         stress_rot = np.einsum('ij,...jk,lk->...il', rot, stress, rot)
#     else:
#         # stress_rot = rot.T @ stress @ rot
#         stress_rot = np.einsum('ji,...jk,kl->...il', rot, stress, rot)
#     return stress_rot


# def calc_unit_stress_rot(alpha):
#     # Calcula el esfuerzo en notacion de voigt para un angulo alpha
#     sig0 = np.array([[1.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0],
#                      [0.0, 0.0, 0.0]
#                      ])
#     sig_alpha = rot_ang(sig0, alpha)
#     sig_alpha = to_voigt(sig_alpha)

#     return sig_alpha

# # sig_eq in tensor notation
# def calc_der_tensor(sig_eq, sigma, ep, eps=1.0e-6):

#     print("calc_der_tensor, sig_eq", sig_eq)
#     print("calc_der_tensor, sigma", sigma[-1])
#     print("calc_der_tensor, ep", ep[-1])

#     sigma = to_voigt(sigma)
#     eps1 = np.zeros_like(sigma)
#     eps2 = np.zeros_like(sigma)

#     print("calc_der_tensor, sig2", sigma[-1])

#     for i in range(6):
#         sig_ii = sigma.copy()
#         sig_ii[..., i] += eps
#         if i>=3: sig_ii[..., i] -= eps/2
#         sig_ii = from_voigt(sig_ii)
#         eps1[..., i] = sig_eq(sig_ii, ep)
#         print("calc_der_tensor_eps1:", sig_ii[-1], ep[-1], sig_eq(sig_ii, ep))

#     for i in range(6):
#         sig_ii = sigma.copy()
#         sig_ii[..., i] -= eps
#         if i>=3: sig_ii[..., i] += eps/2
#         sig_ii = from_voigt(sig_ii)
#         eps2[..., i] = sig_eq(sig_ii, ep)

#     print("calc_der_tensor, eps1 y 2", eps1[-1], eps2[-1])
    
#     # print(der, der-sig_i)
#     der = (eps1 - eps2) / (2*eps)
#     der = from_voigt(der)
#     return der

# def calc_cauchy_stress(hard_fun, yield_fun, stress_ref, ep):
#     shape_new_sig = list(ep.shape)
#     shape_new_sig[-1] = 6

#     new_sig = np.full(shape_new_sig, stress_ref)
#     new_sig_eq = yield_fun(new_sig, ep)
#     hard = hard_fun(ep)

#     beta = hard/new_sig_eq
#     new_sig = beta*new_sig
#     return new_sig