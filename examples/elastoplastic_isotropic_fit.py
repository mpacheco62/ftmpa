import lmfit
import numpy as np
import matplotlib.pyplot as plt

from ftmpa.experiments import ExpState, ExpData
from ftmpa.models.handlers.elasticity import HandlerHooke
from ftmpa.models.handlers.autoModels import AutoElastoPlasticAsociated
from ftmpa.for_lmfit import IterPrintForLmfit, ResidualScalarForLmfit

from ftmpa.models import hardening, elasticity
from ftmpa.models.handlers.hardening import HandlerVoceMod
from ftmpa.models.handlers.yields import HandlerVonMises

# Making virtual experiment
strain_p = np.linspace(0.0, 0.2, 100)
stress = np.zeros((len(strain_p), 3, 3))
stress[:, 0, 0] = hardening.VoceMod(sig0=100, k=0.0, q=100, n=10)(strain_p)
strain_e = elasticity.Hooke(E=100e3, nu=0.3).strain(stress=stress)[:, 0, 0]
strain_tensile = strain_p + strain_e

elastic_stress = np.linspace([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[stress[0, 0, 0], 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             50)

elastic_strain = elasticity.Hooke(
    E=100e3, nu=0.3).strain(elastic_stress)[:, 0, 0]

strain_tensile = np.concatenate((elastic_strain, strain_tensile))
stress_tensile = np.concatenate((elastic_stress[:, 0, 0], stress[:, 0, 0]))

stress[:, 1, 1] = stress[:, 0, 0]
strain_e = elasticity.Hooke(E=100e3, nu=0.3).strain(stress=stress)[:, 0, 0]
strain_biaxial = strain_p.copy()/2.0 + strain_e
elastic_stress = np.linspace([[0, 0, 0],
                              [0, 0, 0],
                              [0, 0, 0]],
                             [[stress[0, 0, 0], 0, 0],
                              [0, stress[0, 1, 1], 0],
                              [0, 0, 0]],
                             50)
elastic_strain = elasticity.Hooke(
    E=100e3, nu=0.3).strain(elastic_stress)[:, 0, 0]
strain_biaxial = np.concatenate((elastic_strain, strain_biaxial))
stress_biaxial = np.concatenate((elastic_stress[:, 0, 0], stress[:, 0, 0]))

stress_tensile += np.random.normal(0, 2.1, len(stress_tensile))
stress_biaxial += np.random.normal(0, 2.1, len(stress_biaxial))

# Define the parameters of models
params = lmfit.Parameters()

# Define elastic parameters
params.add('E', 100e3, vary=False)
params.add('nu', 0.3, vary=False)

# Define hardening parameters
params.add('sig0', 100, vary=False)
params.add('k', 0, vary=False)
params.add('q', 10, min=0.0, max=1000.0)
params.add('n', 1, min=0.0, max=1000.0)


# Define the name of parameters for models
h_elasticity = HandlerHooke(E="E", nu="nu")
h_hardening = HandlerVoceMod(sig0="sig0", k="k", q="q", n="n")
h_yield = HandlerVonMises()

# Define the auto calc
# this model calc the stress-strain curve for a range of equivalent plastic strain
model = AutoElastoPlasticAsociated(h_yield=h_yield,
                                   h_hardening=h_hardening,
                                   h_elasticity=h_elasticity,
                                   strain_plas_max=0.5)

# Define the experiment and the function to obtain the results for the tensor stress and strain
tensile = ExpData(x=strain_tensile, y=stress_tensile,
                  model_data={'x': lambda stress, strain: strain[..., 0, 0],
                              'y': lambda stress, strain: stress[..., 0, 0]}
                  )

biaxial = ExpData(x=strain_biaxial, y=stress_biaxial,
                  model_data={'x': lambda stress, strain: strain[..., 1, 1],
                              'y': lambda stress, strain: stress[..., 1, 1]}
                  )

# Define the experiment load status
exp_loads_tensile = ExpState(model=model,
                             results_test={'exp1': tensile},
                             load_state=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float))

exp_loads_biaxial = ExpState(model=model,
                             results_test={'exp1': biaxial},
                             load_state=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float))

# Minimize the function
out = lmfit.minimize(ResidualScalarForLmfit([exp_loads_tensile, exp_loads_biaxial]),
                     params,
                     iter_cb=IterPrintForLmfit(),
                     method="ampgo"
                     )
lmfit.report_fit(out.params)
final_params = out.params.valuesdict()

# Show the fitted parameters
data_fitted_tensile = exp_loads_tensile.data_to_graph(final_params)
data_fitted_biaxial = exp_loads_biaxial.data_to_graph(final_params)
plt.plot(data_fitted_tensile['exp1']['x_exp'],
         data_fitted_tensile['exp1']['y_exp'])
plt.plot(data_fitted_tensile['exp1']['x_model'],
         data_fitted_tensile['exp1']['y_model'])
plt.show()

plt.plot(data_fitted_biaxial['exp1']['x_exp'],
         data_fitted_biaxial['exp1']['y_exp'])
plt.plot(data_fitted_biaxial['exp1']['x_model'],
         data_fitted_biaxial['exp1']['y_model'])
plt.show()
