@echo off

IF exist venv ( echo ':: venv exists :: ' ) ELSE ( echo ':: install python venv :: ' && python -m venv venv )

echo ' :: activate venv :: '
CALL venv\Scripts\activate.bat

echo ' :: install requirements :: '
pip install -r requirements.txt

echo ' :: DONE :: '
echo ' :: launch app :: '

python main.py -c 1
