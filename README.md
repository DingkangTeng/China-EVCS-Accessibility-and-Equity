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
- ``pyproj`` == 3.7.1

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

## Project Structure 
```
┌─┬ code
│ ├── code for accessibility and equity computation/ # Code for 2SFCA & M2SFCA accessibility and equity calculation  
│ └── code for analysis and visualization/ # Code for output data analysis and visualization  
│  
└─┬ data # Full dataset for China analysis (too large for GitHub)  
  ├─┬ _AnalysisData  
  │ ├── figure/ # Dataset for visualization  
  │ └── result/ # Output Dataset for acc and equity analysis  
  │    
  └─┬ _SampleData     
    ├── CN-EV charging station/ # Dataset for China EVCS datasets 2015-2025
    ├── CN-map/ # Dataset for China and Beijing boundaries
    ├── CN-WorldPOP/ # Population counts dataset for worldpop, too large to upload
    ├── _Population_GDP_Highway.7z.001 # Population density dataset for worldpop (too large to upload), gdp per capita, and road (highway) network data
    ├── _Population_GDP_Highway.7z.002 # _Population_GDP_Highway.7z part 2
    ├── _Population_GDP_Highway.7z.003 # _Population_GDP_Highway.7z part 3
    ├── _Population_GDP_Highway.7z.004 # _Population_GDP_Highway.7z part 4
    ├── _RowData.7z # City level GDP data, EVCS data, and NEV ownership data
    └── _China EVCS_Dataset_with_Boudaries.7z # ataset for China EVCS datasets 2015-2025 and Chiese cities boundaries
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
4. Run script in ``code``
   1. **code for accessibility and equity computation**: configure the paths in ``main.py`` as needed and set up the 2SFCA and M2SFCA functions within the ``process_evse_type`` function in ``tool/overlay_tool_thread.py`` to generate the respective outputs.
   2. **code for analysis and visualization**: modify the datasets dir in the first cell and run each cell in the jupyter ``main.ipynb``
5. Outputs (including excel files and figures) will be stored in the dir ``./_AnalysisData/result/`` and ``./_AnalysisData/figure/``, respectively.

## Sample Data Testing

One Python scripts are provided for testing the project using small sample datasets (located in the `_SampleData` folder). The paths in the scripts are already set up, so you can download the sample data and run the scripts directly:

- `code/code for accessibility and equity computation/main_sample_data.py`  
  - Test the **M2SFCA accessibility and equity** calculations using sample data from Beijing in 2015.

# Contact
- Leave questions in [Issues on GitHub](https://github.com/DingkangTeng/China-EVCS-Accessibility-and-Equity/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk)
or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
