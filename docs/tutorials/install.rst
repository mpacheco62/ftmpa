**************************
How to install
**************************

.. contents:: Table of Contents

-------------------------

Lo primero es tener instalado la aplicación git y los requisitos del programa
que son las librerías numpy scipy y lmfit de python.

.. code:: bash

    sudo apt install git python3-pip python3-numpy python3-scipy python3-lmfit

Luego, se debe descargar la librería, para esto se crea una carpeta llamada git en el home
(se puede hacer en algún otro sitio). Para despues, descargar directamente la librería.

.. code:: bash

    mkdir -p ~/git
    cd ~/git
    git clone https://github.com/mpacheco62/ftmpa

Con estos últimos comandos ya tenemos descargada la librería. Ahora toca instalarla, para
esto utilizamos la librería pip. En este caso se instalará solamente local (el usuario
actual solamente podrá utilizarla).

.. code:: bash

    pip3 install .

Con lo anterior ya quedó instalada. Además, es importante ver que en la carpeta actual (la de la librería)
hay una carpeta llamada "examples" donde podrá encontrar algunos ejemplos.