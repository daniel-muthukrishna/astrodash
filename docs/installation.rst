============
Installation
============

Using pip
---------
.. code-block:: bash

    pip install astrodash --upgrade

From Source
-----------
.. code-block:: bash

    git clone https://github.com/daniel-muthukrishna/DASH.git


Dependencies
------------
Using pip to install DASH will automatically install the mandatory dependencies: numpy, scipy, specutils, pyqtgraph, and tensorflow.

PyQt5 is the final dependency, and is optional. It is only required if you would like to use the graphical interface.
If you have an anaconda installation, this should already be preinstalled, but can otherwise be simply installed by running the following in the terminal:

.. code-block:: bash

    conda install pyqt


Or, ONLY if you do not have anaconda and if you have python 3, it can be installed by running the following in the terminal:
.. code-block:: bash

    pip3 install pyqt5


Platforms
---------
DASH can be run on Mac (tested on Sierra 10.12), most Linux distributions (tested on Ubuntu 16), and on Windows (tested on Windows 10).

1. Mac and Linux distributions:

    DASH is available on both Python2 and Python3 distributions, and easily installed with

        .. code-block:: bash

            pip install astrodash

2. Windows:

    Currently one of the primary dependencies, Tensorflow, is only available on Python 3 for Windows.
    So DASH is available on Python 3 distributions. It can be installed with:

        .. code-block:: bash

            pip install astrodash

    If this fails, try first installing specutils with the following:

        .. code-block:: bash

            conda install -c astropy specutils