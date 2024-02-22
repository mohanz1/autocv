.. autocv documentation master file, created by
   sphinx-quickstart on Wed Feb 14 15:24:45 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Welcome to AutoCV
=================

AutoCV is an innovative computer vision library that simplifies image processing and analysis. With a focus on ease of use and flexibility, AutoCV enables rapid development of computer vision applications.



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   autocv.autocv
   autocv.core.window_capture
   autocv.core.vision
   autocv.core.input
   autocv.models
   autocv.utils



Features
--------

* Easy-to-use interface
* Comprehensive image processing functions
* High performance with real-time capabilities
* Extensive documentation



Quick Start
-----------

To get started with AutoCV, install the package using pip:

.. code-block:: bash

   pip install autocv



Prerequisites
-------------

- Autocv requires Python 3.10+

- Install `Google Tesseract OCR <https://github.com/tesseract-ocr/tesseract>`_ (additional info how to install the engine on Windows). You must be able to invoke the tesseract command as `tesseract`. If this isn't the case, for example because tesseract isn't in your PATH, you will have to change the "tesseract_cmd" variable `pytesseract.pytesseract.tesseract_cmd`.

- *Note:* In some rare cases, you might need to additionally install `tessconfigs` and `configs` from `tesseract-ocr/tessconfigs <https://github.com/tesseract-ocr/tessconfigs>`_ if the OS specific package doesn't include them.



Example
-------

.. code-block:: python

   from autocv import AutoCV


   # initialize AutoCV class
   autocv = AutoCV()

   # set handle
   autocv.set_hwnd_by_title("RuneLite")

   # set inner handle recursively
   while autocv.set_inner_hwnd_by_title("SunAwtCanvas"):
       pass

    # re-write memory to disable getCursorPos
   autocv.antigcp()

   # refresh (or in this case, set) the backbuffer image
   autocv.refresh()

   # find the first green contour with an area over 50 and a tolerance of 50
   contour = autocv.find_contours((0, 255, 0), tolerance=50, min_area=50).first()

   #  move and click the mouse to a random point in the contour
   autocv.move_mouse(*contour.random_point())
   autocv.click_mouse()
