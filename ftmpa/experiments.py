"""
main.py
====================================
The core module of my example project
"""

from typing import Union
import numpy as np
from scipy.interpolate import interp1d

from .models.handlers.autoModels import AutoModels

class ExpData:
    """An example docstring for a class definition."""
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray

    def __init__(self, x, y, model_data, w=None, x_filter=None, y_filter = None):
        self.model_data = model_data

        if x_filter is not None:
            x_min = x_filter[0]
            x_max = x_filter[1]
            to_filter = np.logical_and((x <= x_max), (x >= x_min))
            x = x[to_filter]
            y = y[to_filter]
            if w is not None: w = w[to_filter]

        if y_filter is not None:
            y_min = y_filter[0]
            y_max = y_filter[1]
            to_filter = np.logical_and((y <= y_max), (y >= y_min))
            x = x[to_filter]
            y = y[to_filter]
            if w is not None: w = w[to_filter]

        self.x = x
        self.y = y
        self.w = w

    def from_model(self, stress, strain):
        x_from_model = self.model_data['x']
        y_from_model = self.model_data['y']
        fun_from_model = interp1d(x_from_model(stress=stress, strain=strain),
                                  y_from_model(stress=stress, strain=strain),
                                  fill_value =np.nan,
                                  bounds_error=False)
        # print('x_model:', x_from_model(stress=stress, strain=strain))
        # print('y_model:', y_from_model(stress=stress, strain=strain))
        # print('x:', self.x)
        y_model = fun_from_model(self.x)
        x_model = self.x
        return x_model, y_model
        
    def residual(self, stress, strain, nrmsd=True, exponent:float=2):
        x_model, y_model = self.from_model(stress, strain)
        res = self.y - y_model
        if nrmsd:
            delta_y = np.amax(res)-np.amin(res)
            res = res/(len(self.y)**(1.0/exponent))
            res = res/delta_y
        if self.w is not None: res = self.w*res
        return res

class ExpState:
    model: AutoModels
    load_state: np.ndarray
    results_test: dict[str, ExpData]
    rotate_test: Union[np.ndarray, None]

    def __init__(self, model: AutoModels, load_state: np.ndarray, results_test: dict[str, ExpData]):
        self.model = model
        self.load_state = load_state
        self.results_test = results_test

    def data_to_graph(self, param: dict[str, float]):
        stress, strain = self.model.calc(param, self.load_state)

        data = {}
        for exp_name, test in self.results_test.items():
            data[exp_name] = {}
            data[exp_name]['x_model'], data[exp_name]['y_model'] = test.from_model(stress=stress, strain=strain)
            data[exp_name]['x_exp'], data[exp_name]['y_exp'] = (test.x, test.y) 

        return data

    def residual(self, param):
        stress, strain = self.model.calc(param, self.load_state)
        if np.logical_or(np.isnan(stress), np.isnan(strain)).any():
            return np.nan

        res = []
        for exp_name, test in self.results_test.items():
            res.append(test.residual(stress=stress, strain=strain))
        
        return np.concatenate(res)

class ExpStateRotated:
    model: AutoModels
    load_state: np.ndarray
    results_test: dict[str, ExpData]
    rotate_test: Union[np.ndarray, None]

    def __init__(self, model: AutoModels, load_state: np.ndarray, results_test: dict[str, ExpData], rotate_test: Union[np.ndarray, None]=None):
        self.model = model
        self.load_state = load_state
        self.rotate_test = rotate_test
        self.results_test = results_test

        if self.rotate_test is not None:
            self.load_state = np.einsum('ij, jk, lk', rotate_test, load_state, rotate_test)
        else:
            self.rotate_test = np.eye(3)

    def data_to_graph(self, param: dict[str, float]):
        stress, strain = self.model.calc(param, self.load_state)
        stress_non_rotated = np.einsum('ji, ...jk, kl', self.rotate_test, stress, self.rotate_test)
        strain_non_rotated = np.einsum('ji, ...jk, kl', self.rotate_test, strain, self.rotate_test)

        data = {}
        for exp_name, test in self.results_test.items():
            data[exp_name] = {}
            data[exp_name]['x_model'], data[exp_name]['y_model'] = test.from_model(stress=stress_non_rotated, strain=strain_non_rotated)
            data[exp_name]['x_exp'], data[exp_name]['y_exp'] = (test.x, test.y) 

        return data

    def residual(self, param):
        stress, strain = self.model.calc(param, self.load_state)
        if np.logical_or(np.isnan(stress), np.isnan(strain)).any():
            return np.nan
        stress_non_rotated = np.einsum('ji, ...jk, kl', self.rotate_test, stress, self.rotate_test)
        strain_non_rotated = np.einsum('ji, ...jk, kl', self.rotate_test, strain, self.rotate_test)

        res = []
        for exp_name, test in self.results_test.items():
            res.append(test.residual(stress=stress_non_rotated, strain=strain_non_rotated))
        
        return np.concatenate(res)

