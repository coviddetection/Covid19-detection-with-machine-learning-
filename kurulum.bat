start python kurulum.exe 
set/p devam et:
pythom -m pip install numpy
python -m pip install --upgrade tensorflow
python -m pip install tensor
pythom -m pip install sklearn
pythom -m pip install pandas
pythom -m pip install pillow
pythom -m pip install keras 
set/p konum=python kurulum konumu:
copy cudart64_101.dll %konum%
echo kurulum tamamlandi
python -m detection.py 
