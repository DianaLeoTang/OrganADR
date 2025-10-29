System
OrganADR is developed and tested on Ubuntu 22.04 with GPU RTX 4090.
Linux system with GPU(s) is recommend to reproduce and/or use OrganADR.

Languages
Python (Jupyter Notebook), Shell.

Environment
Conda is recommended for managing Python virtual environments. Two environments are used during OrganADR's development.
Environment1: the environment for data analysis, visualizing the results, etc.
Environment2: the environment for developing and evaluting the deep learning model(s).

Install
For environment1, common packages for data analysis (such as numpy pandas scipy matplotlib) are used.
You can refer to the specific packages imported in the file you need to run, and then to configure this environment.
A .txt file and a .yaml file are provided for reference. You can try to create a conda environment directly using these files (if that works on your system).
If that doesn't work, you can refer to these files to install the specific packages you need using conda or pip.
For environment2, we have provided a .bash file for one-click installation. You need to confirm that the path in the .bash file is compatible with your system and platform.
In the .bash file, pytorch related packages are installed through the local .whl files. You need to download them following the file names before run the .bash file.

Note
Considering that conda sources, pip sources, and packages are constantly updated, it is possible that you encounter some problems during the installation process.
In order to make a fair comparison, OrganADR uses the same deep learning packages as EmerGNN (pytorch + torchdrug).
One important thing to be aware of is that if you want to use the GPU version of pytorch, please install pytorch first and then install torchdrug (as shown in the .bash file).

Download
Before running Part_02, please refer to the Data Availability section of the article (please also follow github for updates and more details, if necessary) and download the necessary rawdata, preprocessed data and other files.

Contact us
If you have any questions, feel free to contact BoyangLi@bit.edu.cn or the corresponding authors.