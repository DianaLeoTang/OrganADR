import os
os.add_dll_directory(r"C:\Users\tangw\.conda\envs\organadr\Lib\site-packages\torch\lib")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin")

from torchdrug.layers.functional import spmm
print("OK:", spmm.__file__)
