# README
## Prerequisites:

- Python-tesseract requires Python 3.6+

- Install [Google Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (additional info how to install the engine on Linux, Mac OSX and Windows). You must be able to invoke the tesseract command as `tesseract`. If this isn't the case, for example because tesseract isn't in your PATH, you will have to change the "tesseract_cmd" variable `pytesseract.pytesseract.tesseract_cmd`. Under Debian/Ubuntu you can use the package **tesseract-ocr**. For Mac OS users. please install homebrew package **tesseract**.

- *Note:* In some rare cases, you might need to additionally install `tessconfigs` and `configs` from [tesseract-ocr/tessconfigs](https://github.com/tesseract-ocr/tessconfigs) if the OS specific package doesn't include them.
