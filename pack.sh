echo "THIS IS SQUAREFACE PACKAGES SCRIPT (Valid version of PyInstaller is required!!!)"
echo "Enter path to virtual environment: "
PATH_TO_VENV=$1

pyinstaller --windowed --hiddenimport appdirs --onefile --paths $PATH_TO_VENV SquareFace.py
