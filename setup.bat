@echo off
git submodule init
git submodule update
python -m venv .venv
(
    echo @echo off
    echo .venv\Scripts\activate 
) > activate.bat
call activate
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
(
    echo @echo off
    echo call activate 
    echo python run.py
) > run.bat