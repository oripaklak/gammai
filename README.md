# HPGe Gamma Spectrum Processing Pipeline

## Introduction
Recent advances in data processing and machine learning have greatly improved the analysis of gamma-ray spectra. However, applying these methods directly to raw physical experiment files remains challenging. We propose this pipeline to parse raw high-purity germanium (HPGe) `.spe` files, converting them into clean `.txt` files, and finally compiling them into a unified NumPy matrix for downstream analysis.

## Dataset
The data used for this pipeline can be downloaded here: [https://www-nds.iaea.org/idb/](https://www-nds.iaea.org/idb/). 

To prepare the dataset:
* Navigate to the link and select the **HPGe detector** option.
* Download the provided archive.
* Unzip the downloaded data.
* Save the extracted `.spe` files under the matching folder name `IAEA_HPGe_spe` within your project's `data/` directory.

## Instructions
In order to run the pipeline, you should create the `gammai` environment and execute the scripts. 

### Requirements
First, create an Anaconda environment with the required packages:
```bash
conda create --name gammai python=3.10 numpy -y
```

Then, activate your environment:
```bash
conda activate gammai
```

### Project Structure
Ensure your project follows this hierarchy before running the code:
```text
project_root/
+-- data/
¦   +-- IAEA_HPGe_spe/                 <-- Place unzipped .spe files here
+-- analyze_files/
¦   +-- plot_spe.py                    <-- Plots the .spe files
¦   +-- open_cnf_rpt.py                <-- Plots .cnf/.rpt files with a legend
¦   +-- spe2txt.py                     <-- Converts .spe to .txt
¦   +-- create_save_data_matrix.py     <-- Builds the .npy matrix
+-- README.md
```

### Execution
To analyze, parse the data, and build the matrix, run the following scripts from the root directory:

**1. Plot `.spe` files:**
In order to visualize the raw `.spe` files, run:
```bash
python analyze_files/plot_spe.py
```

**2. Plot `.cnf` / `.rpt` files:**
To create a plot that includes a legend for these file types, run:
```bash
python analyze_files/open_cnf_rpt.py
```

**3. Convert `.spe` to `.txt`:**
This script creates a new directory containing `.txt` files with the extracted channels and counts.
```bash
python analyze_files/spe2txt.py
```

**4. Build and save the data matrix:**
This script reads the converted text files and saves that data into a single `.npy` matrix file for machine learning or statistical analysis.
```bash
python analyze_files/create_save_data_matrix.py
```