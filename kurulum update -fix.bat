@echo off
title kurulum 
start python kurulum.exe 
set/p devam et:
python -m pip install numpy
python -m pip install --upgrade tensorflow
python -m pip install tensor
python -m pip install sklearn
python -m pip install pandas
python -m pip install pillow
python -m pip install keras 
set/p konum=python kurulum konumu:
copy cudart64_101.dll C:\Users\%username%\AppData\Local\Programs\Python\Python36
echo kurulum tamamlandi
python -m detection.py 
