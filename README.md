# Overview
This repository contains all codes and (sample) dataset of the paper - 
***Advancing accessibility and equity of public electric vehicle charging stations in China***. 

Note that the **full dataset** can be requested through our [Global EV Data Initiative](https://globalevdata.github.io/data.html).

# Requirements and Installation
The whole analysis-related codes should run with a **Python** environment, regardless of operating systems theoretically. 
We successfully execute all the codes in Windows (Win11) machines.
More detailed info is as below:

## Prerequisites 
It is highly recommended to install and use the following versions of python/packages to run the codes:
- ``esda`` == 2.7.1
- ``fiona`` == 1.10.1
- ``geopandas`` == 1.1.0
- ``libpysal`` == 4.13.0
- ``matplotlib`` == 3.10.7
- ``numpy`` == 2.3.5
- ``pandas`` == 2.3.3
- ``rasterio`` == 1.4.3
- ``scikit_learn`` == 1.7.2
- ``scipy`` == 1.16.3
- ``seaborn`` == 0.13.2
- ``tqdm`` == 4.67.1

## Installation
It is highly recommended to download [AnaConda](https://www.anaconda.com) to create/manage Python environments.
You can create a new Python environment and install required aforementioned packages via both the GUI or Command Line.
Typically, the installation should be prompt (around _10-20 min_ from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed)
- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.


- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment
     ```
     conda create --name <input_your_environment_name> python=3.10.6
     ```
  2. Activate the new environment 
     ```
     conda activate <input_your_environment_name>
     ```
  3. Install all packages one by one 
     ```
     conda install <package_name>=<specific_version>
     ```

# Usage
1. Git clone/download the repository to your local disk.
2. Unzip the full datasets in ``data`` (which can be provided upon request, see [Overview](https://github.com/DingkangTeng/China-EVCS-Accessibility-and-Equity?tab=readme-ov-file#overview))
   > The structure of the provided full datasets should look like as below:
   > 
   > ```
   > - _China EVCS_Dataset_with_Boudaries.7z
   > - _RowData.7z
   > - _Population_GDP_Highway.7z
   > ```
3. Unzip each compressed dataset (``.7z`` file) and drag folders/files into corresponding dir of this repo. 
 For example, extract all files from the ``_China EVCS_Dataset_with_Boudaries`` to the root dir ``.``.
4. Run
   1. **preprocessing**: ...
   2. **analysis**: modify the datasets dir in the first cell and run each cell in the jupyter ``main.ipynb``
5. Outputs (including excel files and figures) will be stored in the dir ``./_AnalysisData/result/`` and ``./_AnalysisData/figure/``, respectively.

# Contact
- Leave questions in [Issues on GitHub](https://github.com/DingkangTeng/China-EVCS-Accessibility-and-Equity/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.