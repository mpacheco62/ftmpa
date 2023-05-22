import lmfit
import numpy as np
import matplotlib.pyplot as plt

from ftmpa.experiments import ExpState, ExpData
from ftmpa.models.handlers.elasticity import HandlerHooke
from ftmpa.models.handlers.autoModels import AutoElasticity
from ftmpa.for_lmfit import IterPrintForLmfit, ResidualVectorForLmfit


# Making virtual experiment
strain_exp = np.linspace(0.0, 0.02, 100)
stress_exp = 100e3*strain_exp
stress_exp += np.random.normal(0, 20, len(strain_exp))

# Define the parameters of models
params = lmfit.Parameters()
params.add('young', 1e3, min=0.0, max=2000e3)
params.add('nu', 0.3, vary=False)

# Define the name of parameters for models
h_elasticity = HandlerHooke(E="young", nu="nu")

# Define the auto calc
# this model calc the stress-strain curve for a stress 1e9 for default
model = AutoElasticity(h_elasticity=h_elasticity)

# Define the experiment and the function to obtain the results for the tensor stress and strain
exp_1 = ExpData(x=strain_exp, y=stress_exp,
                model_data={'x': lambda stress, strain: strain[...,0,0],
                            'y': lambda stress, strain: stress[...,0,0]}
                )

# Define the experiment load status 
exp_loads_1 = ExpState(model=model,
                       results_test={'specimen1': exp_1},
                       load_state=np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float))

# Minimize the function
out = lmfit.minimize(ResidualVectorForLmfit([exp_loads_1]), params, iter_cb=IterPrintForLmfit())
lmfit.report_fit(out.params)
final_params = out.params.valuesdict()

# Show the fitted parameters
data_fitted = exp_loads_1.data_to_graph(final_params)
plt.plot(data_fitted['specimen1']['x_exp'], data_fitted['specimen1']['y_exp'])
plt.plot(data_fitted['specimen1']['x_model'], data_fitted['specimen1']['y_model'])

plt.show()
