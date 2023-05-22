*******************************
Isotropic Elastoplastic Fitting
*******************************

Here is more Restructuredtext and Sphinx directives.

.. contents:: Table of Contents

-------------------------

Para ajustar el endurecimiento isotropo se utiliza la ley de Voce y se asume un esfuerzo equivalente de Von Mises.

.. math:: Y(\bar{\epsilon}^p) = \sigma_0 + Q\left[1-\exp(-n\bar{\epsilon}^p)\right]

Donde :math:`\bar{\epsilon}^p` es la deformación plástica efectica, :math:`\sigma_0` es el límite elástico y finalmente
:math:`n` y :math:`Q` son constantes del modelo.

Para realizar las curvas en cada estado de tensión se asume un rango de deformación plástica efectiva, que el estado de tensión
es proporcional a un estado de referencia y que la función de fluencia
sea homogénea de grado 1. Debido a lo anterior se asume un estado de esfuerzos de referencia en base a un multiplicador :math:`\beta`.

.. math:: \boldsymbol{\sigma} = \beta\boldsymbol{\sigma_{ref}}

Por lo tanto para una deformación plástica efectiva :math:`\bar{\epsilon}^p` dada se puede calcular el esfuerzo en base al 
esfuerzo equivalente :math:`\sigma_{eq}`.

.. math:: \sigma_{eq}(\boldsymbol\sigma) - Y(\bar{\epsilon}^p) = 0 \implies \sigma_{eq}(\beta\boldsymbol\sigma_{ref}) = \beta\sigma_{eq}(\boldsymbol\sigma_{ref}) = Y(\bar{\epsilon}^p)
.. math:: \beta = \frac{Y(\bar{\epsilon}^p)}{\sigma_{eq}(\boldsymbol\sigma_{ref})} 
.. math:: \boldsymbol{\sigma} = \frac{Y(\bar{\epsilon}^p)}{\sigma_{eq}(\boldsymbol\sigma_{ref})} \boldsymbol{\sigma_{ref}}

De esta forma se obtiene el estado de tensión con tan solo un el estado de tensión de referencia para cualquier deformación plástica efectiva.

Para la obtener la deformación se ocupa la regla de flujo además de una propiedad de las funciones homogéneas de grado 1
:math:`\frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma} (\boldsymbol\sigma) = \frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma} (\beta\boldsymbol\sigma_{ref}) = \frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma} (\boldsymbol\sigma_{ref}) = cte`.
Ademas, se asume que con la función de fluencia utilizada se cumple :math:`\dot\lambda = \dot{\bar{\epsilon}}^p`.

.. math:: \dot{\boldsymbol\epsilon}^p = \dot{\bar{\epsilon}}^p \frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma}(\boldsymbol\sigma) = \dot{\bar{\epsilon}}^p\frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma}(\boldsymbol\sigma_{ref})
.. math:: \boldsymbol{\epsilon}^p = \frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma}(\boldsymbol\sigma_{ref})  \int_0^{\bar{\epsilon}^p} \dot{\bar{\epsilon}}^p = \bar{\epsilon}^p\frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma}(\boldsymbol\sigma_{ref}) 

Finalmente se obtiene la deformación total :math:`\boldsymbol\epsilon = \boldsymbol{\epsilon}^e + \boldsymbol{\epsilon}^p` al calcular la deformación elástica.

.. math:: \boldsymbol{\epsilon}^e = \frac{1}{E}\left[(1+\nu)\boldsymbol{\sigma} - \nu\boldsymbol{I}\text{tr}(\boldsymbol\sigma)\right]

En resumen se utilizan las siguiente ecuaciones para calcular las curvas para un estado de tensión dado:

.. math:: \boldsymbol{\sigma} = \frac{Y(\bar{\epsilon}^p)}{\sigma_{eq}(\boldsymbol\sigma_{ref})} \boldsymbol{\sigma_{ref}}
.. math:: \boldsymbol{\epsilon}^p = \bar{\epsilon}^p\frac{\partial\sigma_{eq}}{\partial \boldsymbol\sigma}(\boldsymbol\sigma_{ref})
.. math:: \boldsymbol{\epsilon}^e = \frac{1}{E}\left[(1+\nu)\boldsymbol{\sigma} - \nu\boldsymbol{I}\text{tr}(\boldsymbol\sigma)\right]
.. math:: \boldsymbol{\epsilon} = \boldsymbol{\epsilon}^e + \boldsymbol{\epsilon}^p

Ajuste de tracción y biaxial
============================

El siguiente ejemplo demuestra como se ajusta el modelo elastoplastico
con fluencia Von Mises y endurecimiento de Voce utilizando la librería
lmfit. Esto para ajustar una curva biaxial-equiaxial y un ensayo de
tracción.

.. code-block:: python
   :linenos:
   :caption: elastoplastic_isotropic_fit.py

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


El código de ejemplo se analizará por partes.

.. code-block:: python
   :linenos:
   :lineno-start: 1

   import lmfit
   import numpy as np
   import matplotlib.pyplot as plt

En esta sección se importa la librería lmfit que nos permitirá ajustar los modelos,
numpy para trabajar datos y matplotlib para graficarlos.

.. code-block:: python
   :linenos:
   :lineno-start: 5

   from ftmpa.experiments import ExpState, ExpData
   from ftmpa.models.handlers.elasticity import HandlerHooke
   from ftmpa.models.handlers.autoModels import AutoElastoPlasticAsociated
   from ftmpa.for_lmfit import IterPrintForLmfit, ResidualScalarForLmfit

Se importan las librerías propias de esta documentación, en esta parte no se daran
detalles de ellas, pero sí mas adelante cuando se utilizen.

.. code-block:: python
   :linenos:
   :lineno-start: 10

   from ftmpa.models import hardening, elasticity
   from ftmpa.models.handlers.hardening import HandlerVoceMod
   from ftmpa.models.handlers.yields import HandlerVonMises

Se importan los modelos que nos permitiran crear experimentos virtuales.

.. code-block:: python
   :linenos:
   :lineno-start: 14

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

Se crean datos virtuales para un ensayo de tracción y uno
biaxial, además se le agrega ruido.

.. code-block:: python
   :linenos:
   :lineno-start: 54

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

En estas líneas se definen TODOS los parametros que van a necesitar los modelos, por medio de 
parámetros de lmfit. Aca se definene los valores iniciales de los parámetros, cuales varían y en que rangos.

.. code-block:: python
   :linenos:
   :lineno-start: 67

   # Define the name of parameters for models
   h_elasticity = HandlerHooke(E="E", nu="nu")
   h_hardening = HandlerVoceMod(sig0="sig0", k="k", q="q", n="n")
   h_yield = HandlerVonMises()

Se define que el manejador de la elasticidad será HandleHooke,
que el parámetro del modelo de elasticidad "E" se pasara con nombre de "E" 
y que el parámetro "nu" se pasara con nombre "nu".
Lo mismo con la ley de endurecimiento HandlerVoceMod. Esta ley tiene la siguiente
forma:

.. math:: Y(\bar{\epsilon}^p) = \sigma_0 k\bar{\epsilon}^p + Q\left[1-\exp(-n\bar{\epsilon}^p)\right]

Por lo tanto se define que :math:`k=0` para que sea la función de voce.

En la última línea se define como Von Mises la superficie de fluencia.

.. code-block:: python
   :linenos:
   :lineno-start: 72

   # Define the auto calc
   # this model calc the stress-strain curve for a range of equivalent plastic strain
   model = AutoElastoPlasticAsociated(h_yield=h_yield,
                                      h_hardening=h_hardening,
                                      h_elasticity=h_elasticity,
                                      strain_plas_max=0.5)

Se define el modelo que va a automaticamente crear "simulaciones" para el caso en estudio.

.. code-block:: python
   :linenos:
   :lineno-start: 79

   # Define the experiment and the function to obtain the results for the tensor stress and strain
   tensile = ExpData(x=strain_tensile, y=stress_tensile,
                     model_data={'x': lambda stress, strain: strain[..., 0, 0],
                                 'y': lambda stress, strain: stress[..., 0, 0]}
                     )

   biaxial = ExpData(x=strain_biaxial, y=stress_biaxial,
                     model_data={'x': lambda stress, strain: strain[..., 1, 1],
                                 'y': lambda stress, strain: stress[..., 1, 1]}
                     )

Aquí se definen los experimentos y como se obtienen sus resultados respecto del tensor
:math:`\boldsymbol \sigma` (:code:`stress`) y :math:`\boldsymbol \epsilon` (:code:`strain`).

En este caso se construyo la curva como deformación en el eje "x" y como esfuerzo en el eje "y". Por lo tanto
se necesita que el dato "x" sea la deformacion :math:`\epsilon_{xx}` (:code:`strain[...,0,0]`) y que el dato "y" sea
el esfuerzo :math:`\sigma_{xx}` (:code:`stress[...,0,0]`). Para el caso biaxial, se toma el par :math:`\epsilon_{yy}`,
:math:`\sigma_{yy}`, aunque también podría ser el "xx".

.. code-block:: python
   :linenos:
   :lineno-start: 90

   # Define the experiment load status
   exp_loads_tensile = ExpState(model=model,
                              results_test={'exp1': tensile},
                              load_state=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float))

   exp_loads_biaxial = ExpState(model=model,
                              results_test={'exp1': biaxial},
                              load_state=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float))

Aca es donde se le define al programa que estado de tensión se está aplicando y que modelo 
lo debe representar. El parámetro :code:`model` representa el modelo que se definió en la
línea 74 y que es el :code:`AutoElastoPlasticAsociated`.

El parámetro :code:`result_test` es un diccionario donde se deben colocar todos los
ensayos experimentales que tengan el mismo estado de tensión, en este caso una
tracción simple para el primero, y un estado biaxial para el segundo.

En :code:`load_state` se debe definir cual es el estado de tensiones que se está 
imponinendo en el ensayo. Para el primer caso como es un ensayo de tracción simple,
corresponde a una matriz que solo tiene esfuerzos en :math:`\sigma_{xx}`. Esto se
representa por el código :code:`np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float)`

Para el segundo caso como es un ensayo biaxial-equiaxial,
corresponde a una matriz que solo tiene esfuerzos en :math:`\sigma_{xx}` y 
:math:`\sigma_{yy}`. Esto se representa por el código
:code:`np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=float)`

Con esto queda todo definido para las funciones de la librería lo siguiente es para
realizar la minimización del residuo.

.. code-block:: python
   :linenos:
   :lineno-start: 99

   # Minimize the function
   out = lmfit.minimize(ResidualScalarForLmfit([exp_loads_tensile, exp_loads_biaxial]),
                        params,
                        iter_cb=IterPrintForLmfit(),
                        method="ampgo"
                        )
   lmfit.report_fit(out.params)
   final_params = out.params.valuesdict()

En este codigo se utiliza :code:`lmfit` para realizar la minimización, esta librería ya trae incorporadas
unas funciones de residuos y de iteraciones que son opcionales :code:`ResidualScalarForLmfit([...])` y
:code:`IterPrintForLmfit()`. En la inicialización de :code:`ResidualVectorForLmfit([...])` debe introducir
todos los estados de tensiones que se deban ajustar.

Cabe destacar que en este caso se está utilizando el método ampgo, que es un minimizador
global.

Luego que la minimización se haya realizado (línea 100), se reporta por la salida estandar el reporte final del
ajuste mendiante la línea 101.

Finalmente se guardan los parámetros finales, como diccionario, del ajuste en la variable :code:`final_params`.

.. code-block:: python
   :linenos:
   :lineno-start: 108

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

En estas líneas se obtienen la curvas de los ajustes realizados y se grafican, la línea 109 y 110 realiza los cálculos
con lo parámetros finales. Para cada estado de 
carga se pueden obtener sus ajustes, en este caso contamos con :code:`exp_loads_tensile` y :code:`exp_loads_biaxial,
además cada uno tiene solo un ensayo :code:`'exp1'`. Por lo tanto para obtener estos valores se debe indicar
como :code:`data_fitted['exp1']`, dentro de este diccionario tenemos cuatro datos :code:`x_exp`
los datos experimentales "x", :code:`y_exp` los datos experimentales "y", :code:`x_model` los datos "x" del modelo y 
:code:`y_model` los datos "y" del modelo.

Con estos datos se grafica el ajuste realizado.
