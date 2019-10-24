============
Installation
============

Using pip
---------
The easiest and preferred way to install DASH (to ensure the latest stable version) is using pip:

.. code-block:: bash

    pip install astrodash --upgrade

From Source
-----------
Alternatively, the source code can be downloaded from GitHub by running the following command:

.. code-block:: bash

    git clone https://github.com/daniel-muthukrishna/astrodash.git

Dependencies
------------
Using pip to install DASH will automatically install the mandatory dependencies: :code:`numpy`, :code:`scipy`, :code:`pyqtgraph`, and :code:`tensorflow`.

PyQt5 is the final dependency, and is optional. It is only required if you would like to use the graphical interface.
If you have an anaconda installation, this should already be preinstalled, but can otherwise be simply installed by running the following in the terminal:

.. code-block:: bash

    conda install pyqt


Or, if you do not have anaconda and if you have python 3, it can be installed by running the following in the terminal:

.. code-block:: bash

    pip install PyQt5


Platforms
---------
DASH can be run on Mac (tested on Sierra 10.12), most Linux distributions (tested on Ubuntu 16), and on Windows (tested on Windows 10).

1. Mac and Linux distributions:

    DASH is available on both Python 2 and Python 3 distributions, and can be installed using pip.

2. Windows:

    DASH is only available on Python 3 distributions on Windows and can be installed using pip.

    If the installation fails, try first installing specutils with the following:

        .. code-block:: bash

            conda install -c astropy specutils
