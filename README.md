SQUAREFACEAPP setup


-- Local run without package

NOTE: Python3.9 and Conda required!!!
1. Use SquareFaceVenv virtual environment for package installation via "source SquareFaceVenv/bin/activate"
2. Use requirements.txt to install all modules via "pip install -r requirements.txt"
3. Run via "python3 SquareFace.py"

-- Packaged run

NOTE: Valid version of PyInstaller is required!!!
1. To package app run following command "sudo sh pack.sh /YOUR-ROOT-PATH/SquareFaceApp/SquareFaceVenv/lib/python3.9/site-packages" 
2. You will need to copy "assets" directory to the following path after packaging "/YOUR-ROOT-PATH/SquareFaceApp/dist/SquareFace/Contents/MacOs/SquareFace"
3. Run SquareFace.app