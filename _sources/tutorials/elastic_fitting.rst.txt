**************************
Elastic Fitting
**************************

Here is more Restructuredtext and Sphinx directives.

.. contents:: Table of Contents

-------------------------

Para ajustar la elasticidad se utiliza la ley de Hooke, desde la cual se obtienen los esfuerzos o deformaciones
según las siguientes ecuaciones.

.. math:: \boldsymbol{\epsilon} = \frac{1}{E}\left[(1+\nu)\boldsymbol{\sigma} - \nu\boldsymbol{I}\text{tr}(\boldsymbol\sigma)\right]
.. math:: \boldsymbol{\sigma} = \frac{E}{1+\nu}\left[\boldsymbol{\epsilon} + \frac{\nu}{1-2\nu}\boldsymbol{I}\text{tr}(\boldsymbol\epsilon)\right]


One experiment
==========================

Para ajustar la ley de elasticidad se puede utilizar solo un ensayo, o multiples ensayos.
El siguiente ejemplo demuestra como se ajusta el modelo elástico utilizando la librería
lmfit y un solo ensayo.

.. code-block:: python
   :linenos:
   :caption: elasticity_fit_1.py

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
    params.add('younh', 1e3, min=0.0, max=2000e3)
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
    from ftmpa.models.handlers.autoModels import AutoElasticity
    from ftmpa.for_lmfit import IterPrintForLmfit, ResidualVectorForLmfit

Se importan las librerías propias de esta documentación, en esta parte no se daran
detalles de ellas, pero sí mas adelante cuando se utilizen.

.. code-block:: python
   :linenos:
   :lineno-start: 5

    # Making virtual experiment
    strain_exp = np.linspace(0.0, 0.02, 100)
    stress_exp = 100e3*strain_exp
    stress_exp += np.random.normal(0, 20, len(strain_exp))

Se crean datos virtuales para un ensayo de tracción, además se le agrega ruido.

.. code-block:: python
   :linenos:
   :lineno-start: 16

    # Define the parameters of models
    params = lmfit.Parameters()
    params.add('young', 1e3, min=0.0, max=2000e3)
    params.add('nu', 0.3, vary=False)

En estas líneas se definen TODOS los parametros que van a necesitar los modelos, por medio de 
parámetros de lmfit. Aca se definene los valores iniciales de los parámetros, cuales varían y en que rangos.

.. code-block:: python
   :linenos:
   :lineno-start: 21

    # Define the name of parameters for models
    h_elasticity = HandlerHooke(E="young", nu="nu")

Se define que el manejador será HandleHooke, que el parámetro del modelo de
elasticidad "E" se pasara con nombre de "young", y que el parámetro "nu" se
pasara con nombre "nu".

.. code-block:: python
   :linenos:
   :lineno-start: 24

    # Define the auto calc
    # this model calc the stress-strain curve for a stress 1e9 for default
    model = AutoElasticity(h_elasticity=h_elasticity)

Se define el modelo que va a automaticamente crear "simulaciones" para el caso en estudio.

.. code-block:: python
   :linenos:
   :lineno-start: 28

    # Define the experiment and the function to obtain the results for the tensor stress and strain
    exp_1 = ExpData(x=strain_exp, y=stress_exp,
                    model_data={'x': lambda stress, strain: strain[...,0,0],
                                'y': lambda stress, strain: stress[...,0,0]}
                    )

Aquí se definen los experimentos y como se obtienen sus resultados respecto del tensor
:math:`\boldsymbol \sigma` (:code:`stress`) y :math:`\boldsymbol \epsilon` (:code:`strain`). 

En este caso se construyo la curva como deformación en el eje "x" (línea 6) y como esfuerzo en el eje "y" (linea 8). Por lo tanto
se necesita que el dato "x" sea la deformacion :math:`\epsilon_{xx}` (:code:`strain[...,0,0]`) y que el dato "y" sea
el esfuerzo :math:`\sigma_{xx}` (:code:`stress[...,0,0]`).

.. code-block:: python
   :linenos:
   :lineno-start: 34

    # Define the experiment load status 
    exp_loads_1 = ExpState(model=model,
                        results_test={'specimen1': exp_1},
                        load_state=np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float))

Aca es donde se le define al programa que estado de tensión se está aplicando y que modelo 
lo debe representar. El parámetro :code:`model` representa el modelo que se definió en la
línea 26 y que es el :code:`AutoElasticity`.

El parámetro :code:`result_test` es un diccionario donde se deben colocar todos los
ensayos experimentales que tengan el mismo estado de tensión, en este caso una
tracción simple.

En :code:`load_state` se debe definir cual es el estado de tensiones que se está 
imponinendo en el ensayo. Para este caso como es un ensayo de tracción simple,
corresponde a una matriz que solo tiene esfuerzos en :math:`\sigma_{xx}`. Esto se
representa por el código :code:`np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float)`

Con esto queda todo definido para las funciones de la librería lo siguiente es para
realizar la minimización del residuo.

.. code-block:: python
   :linenos:
   :lineno-start: 39

    # Minimize the function
    out = lmfit.minimize(ResidualVectorForLmfit([exp_loads_1]), params, iter_cb=IterPrintForLmfit())
    lmfit.report_fit(out.params)
    final_params = out.params.valuesdict()

En este codigo se utiliza :code:`lmfit` para realizar la minimización, esta librería ya trae incorporadas
unas funciones de residuos y de iteraciones que son opcionales :code:`ResidualVectorForLmfit([...])` y
:code:`IterPrintForLmfit()`. En la inicialización de :code:`ResidualVectorForLmfit([...])` debe introducir
todos los estados de tensiones que se deban ajustar.

Luego que la minimización se haya realizado (línea 40), se reporta por la salida estandar el reporte final del
ajuste mendiante la línea 41.

Finalmente se guardan los parámetros finales, como diccionario, del ajuste en la variable :code:`final_params`.

.. code-block:: python
   :linenos:
   :lineno-start: 45

    # Show the fitted parameters
    data_fitted = exp_loads_1.data_to_graph(final_params)
    plt.plot(data_fitted['specimen1']['x_exp'], data_fitted['specimen1']['y_exp'])
    plt.plot(data_fitted['specimen1']['x_model'], data_fitted['specimen1']['y_model'])

    plt.show()

En estas líneas se obtienen la curvas de los ajustes realizados y se grafican, la línea 46 realiza los calculos
con lo parámetros finales. Para cada estado de 
carga se pueden obtener sus ajustes, en este caso solo contamos con :code:`exp_loads_1` y que tiene
solamente un ensayo :code:`'specimen1'`. Por lo tanto para obtener estos valores debemo primero debemos
indicarlo como :code:`data_fitted['specimen1']`, dentro de este diccionario tenemos cuatro datos :code:`x_exp`
los datos experimentales "x", :code:`y_exp` los datos experimentales "y", :code:`x_model` los datos "x" del modelo y 
:code:`y_model` los datos "y" del modelo.

Con estos datos se grafica el ajuste realizado.


Multiple experiments
==========================

El siguiente ejemplo demuestra como se ajusta el modelo elástico utilizando la librería
lmfit y multiples ensayos. El ejemplo es muy similar al anterior, por lo cual solo se
resaltan las diferencias.

.. code-block:: python
   :linenos:
   :caption: elasticity_fit_2.py
   :emphasize-lines: 16-18,38-41,45,57

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

   strain_exp_2 = np.linspace(0.0, 0.02, 100)
   stress_exp_2 = 100e3*strain_exp_2
   stress_exp_2 += np.random.normal(0, 30, len(strain_exp_2))

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

   exp_2 = ExpData(x=strain_exp_2, y=stress_exp_2,
                  model_data={'x': lambda stress, strain: strain[...,0,0],
                              'y': lambda stress, strain: stress[...,0,0]}
                  )

   # Define the experiment load status 
   exp_loads_1 = ExpState(model=model,
                        results_test={'specimen1': exp_1, 'specimen2': exp_2},
                        load_state=np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float))

   # Minimize the function
   out = lmfit.minimize(ResidualVectorForLmfit([exp_loads_1]), params, iter_cb=IterPrintForLmfit())
   lmfit.report_fit(out.params)
   final_params = out.params.valuesdict()

   # Show the fitted parameters
   data_fitted = exp_loads_1.data_to_graph(final_params)
   plt.plot(data_fitted['specimen1']['x_exp'], data_fitted['specimen1']['y_exp'])
   plt.plot(data_fitted['specimen1']['x_model'], data_fitted['specimen1']['y_model'])
   plt.plot(data_fitted['specimen2']['x_exp'], data_fitted['specimen2']['y_exp'])

   plt.show()


Solamente comentaremos las diferencias respecto del ejemplo anterior, junto con las líneas agregadas, cambiadas.

.. code-block:: python
   :linenos:
   :lineno-start: 16

   strain_exp_2 = np.linspace(0.0, 0.02, 100)
   stress_exp_2 = 100e3*strain_exp_2
   stress_exp_2 += np.random.normal(0, 30, len(strain_exp_2))

Aca se agregó otro experimento virtual.

.. code-block:: python
   :linenos:
   :lineno-start: 38

   exp_2 = ExpData(x=strain_exp_2, y=stress_exp_2,
                  model_data={'x': lambda stress, strain: strain[...,0,0],
                              'y': lambda stress, strain: stress[...,0,0]}
                  )

Aca se agrega el experimento 2, en este caso solo cambian los datos de entrada respecto de :code:`exp_1`.

.. code-block:: python
   :linenos:
   :lineno-start: 43

   # Define the experiment load status 
   exp_loads_1 = ExpState(model=model,
                        results_test={'specimen1': exp_1, 'specimen2': exp_2},
                        load_state=np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=float))

En este caso, como el estado de tensión es el mismo el experimento se puede agregar al mismo
estado de tensión. Esto se realiza al agregar el siguiente texto :code:`, 'specimen2': exp_2`

.. code-block:: python
   :linenos:
   :lineno-start: 57

   plt.plot(data_fitted['specimen2']['x_exp'], data_fitted['specimen2']['y_exp'])

Finalmente se agrega el gráfico de los datos experimentales que agregamos.