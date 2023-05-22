import xml.etree.ElementTree as ET
import numpy as np

from .experiments import ExpDataAnisotropic
from .common import load_sheet_angle, load_stress_voigt, rot_sheet_angle


def parser_loadState(elem: ET.Element):
    type_fun = elem.attrib["type"]
    if type_fun == "sheet angle":
        return load_sheet_angle(float(elem.text))
    elif type_fun == "stress":
        return load_stress_voigt(elem.text)
    else:
        raise TypeError("Non found a correct type loadState")


def parser_rotateResult(elem: ET.Element):
    type_fun = elem.attrib["type"]
    if type_fun == "sheet angle":
        return rot_sheet_angle(float(elem.text))
    else:
        raise TypeError("Non found a correct type loadState")


def parser_modelData(elem: ET.Element):
    var = elem.attrib["var"]
    text = elem.text
    return var, text


def parser_function(text: str):
    def parsed_function(strain: np.ndarray, stress: np.ndarray):
        e_xx = strain[0, 0]
        e_yy = strain[1, 1]
        e_zz = strain[2, 2]
        e_yz = strain[1, 2]; e_zy = e_yz
        e_xz = strain[0, 2]; e_zx = e_xz
        e_xy = strain[0, 1]; e_yx = e_xy
        s_xx = stress[0, 0]
        s_yy = stress[1, 1]
        s_zz = stress[2, 2]
        s_yz = stress[1, 2]; s_zy = s_yz
        s_xz = stress[0, 2]; s_zx = s_xz
        s_xy = stress[0, 1]; s_yx = s_xy
        return eval(text)
    return parsed_function


def parser_test(test: ET.Element):
    filename = test.find('file').text
    print("filename:", filename)
    x_column = 0
    y_column = 1
    w_column = 2
    data = np.loadtxt(filename)
    w = None
    if data.shape[1] > w_column: w = data[...,w_column] 

    load_state = parser_loadState(test.find('loadState'))
    rot = parser_rotateResult(test.find('rotateResult'))

    model_data_elems = test.findall('modelData')
    model_data = dict()
    for elem in model_data_elems:
        var, text = parser_modelData(elem)
        model_data[var] = parser_function(text=text)

    return ExpDataAnisotropic(x=data[...,x_column], y=data[...,y_column],
                    load_state=load_state, rotate_result=rot,
                    model_data=model_data, w=w)