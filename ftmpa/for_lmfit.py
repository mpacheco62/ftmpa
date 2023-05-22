import numpy as np
from ftmpa.experiments import ExpState

class IterPrintForLmfit:
    min_res: float
    last_param: dict[str, float]
    print: bool

    def __init__(self, print:bool=True, min_res:float =1e99, last_param: dict[str, float]=None):
        self.min_res = min_res
        self.last_param = last_param
        self.print = print

    def __call__(self, params, iter, resid, *args, **kws):
        try:
            len(resid)
            resid = (abs(resid)**2).sum()
        except:
            pass

        if resid < self.min_res:
            self.min_res = resid
            self.last_param = params.valuesdict()

        if self.print:
            print('**********************************')
            print(f'iter: {iter}, resid: {resid}, min_resid: {self.min_res}')


class ResidualScalarForLmfit:
    tests: list[ExpState]
    exponent: int
    nrmsd: bool

    def __init__(self, tests: list[ExpState], exponent: float=2, nrmsd: bool=True):
        self.tests = tests
        self.exponent = exponent
        self.nrmsd = nrmsd

    def __call__(self, pars, *args, **kws):
        param = pars.valuesdict()
        res = []
        for itest in self.tests:
            res.append(itest.residual(param))
        if self.nrmsd:
            res = (abs(np.concatenate(res)**self.exponent).sum())**(1.0/self.exponent)
        else:
            res = abs(np.concatenate(res)**self.exponent)
        return res
    
class ResidualVectorForLmfit:
    tests: list[ExpState]

    def __init__(self, tests: list[ExpState]):
        self.tests = tests

    def __call__(self, pars, *args, **kws):
        param = pars.valuesdict()
        res = []

        for itest in self.tests:
            res.append(itest.residual(param))

        res = abs(np.concatenate(res))
        return res