@echo off
echo ' :: activate venv :: '
CALL env\Scripts\activate.bat
echo ' :: DONE :: '
echo ' :: launch app :: '
python main.py
