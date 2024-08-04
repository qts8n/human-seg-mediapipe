REM @echo off
echo ':: install python venv :: '
python -m venv env
echo ' :: activate venv :: '
CALL env\Scripts\activate.bat
echo ' :: install requirements :: '
pip install -r requirements.txt
echo ' :: DONE :: '
echo ' :: launch app :: '
CALL init.bat
