# AutoCV
![Supported Python versions](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)
![CI](https://github.com/mohanz1/autocv/actions/workflows/ci.yml/badge.svg?branch=main&event=push)
![Mypy Checked](https://img.shields.io/badge/mypy-checked-green.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Prerequisites:

- Python-tesseract requires Python 3.10+

- Install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (additional info how to install the engine on Linux, Mac OSX and Windows). You must be able to invoke the tesseract command as `tesseract`. If this isn't the case, for example because tesseract isn't in your PATH, you will have to change the "tesseract_cmd" variable `pytesseract.pytesseract.tesseract_cmd`. Under Debian/Ubuntu you can use the package **tesseract-ocr**. For Mac OS users. please install homebrew package **tesseract**.

- *Note:* In some rare cases, you might need to additionally install `tessconfigs` and `configs` from [tesseract-ocr/tessconfigs](https://github.com/tesseract-ocr/tessconfigs) if the OS specific package doesn't include them.
