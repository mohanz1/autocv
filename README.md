# AutoCV
![Supported Python versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
![CI](https://github.com/mohanz1/autocv/actions/workflows/build.yml/badge.svg?branch=main&event=push)
![Mypy Checked](https://img.shields.io/badge/mypy-checked-green.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Welcome to AutoCV

AutoCV is an innovative computer vision library that simplifies image processing and analysis. With a focus on ease of use and flexibility, AutoCV enables rapid development of computer vision applications.



### Features

* Easy-to-use interface
* Comprehensive image processing functions
* High performance with real-time capabilities
* Extensive documentation



### Quick Start

To get started with AutoCV, install the package using pip:

```bash
pip install autocv
```

### Prerequisites:

- AutoCV requires Python 3.10+

- Install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (additional info how to install the engine on Linux, Mac OSX and Windows). You must be able to invoke the tesseract command as `tesseract`. If this isn't the case, for example because tesseract isn't in your PATH, you will have to change the "tesseract_cmd" variable `pytesseract.pytesseract.tesseract_cmd`. Under Debian/Ubuntu you can use the package **tesseract-ocr**. For Mac OS users. please install homebrew package **tesseract**.

- *Note:* In some rare cases, you might need to additionally install `tessconfigs` and `configs` from [tesseract-ocr/tessconfigs](https://github.com/tesseract-ocr/tessconfigs) if the OS specific package doesn't include them.

### Example
```py
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
```
