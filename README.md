# HPGe Gamma Spectrum Processing & Generation Pipeline

## Introduction
Recent advances in data processing and machine learning have greatly improved the analysis of gamma-ray spectra. However, applying these methods directly to raw physical experiment files remains challenging. We propose this pipeline to parse raw high-purity germanium (HPGe) `.spe` files, convert them into clean `.txt` files, and compile them into a unified NumPy matrix. Finally, this project leverages deep generative models (1D Convolutional VAE and DiffWave) to learn the underlying physical distribution and generate highly realistic synthetic spectra.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Data Retrieval & Memorization Check](#data-retrieval--memorization-check)


## Dataset Preparation
The data used for this pipeline is sourced from the [IAEA Photonuclear Data Library](https://www-nds.iaea.org/idb/). 

To prepare the dataset:
1. Navigate to the link and select the **HPGe detector** option.
2. Download and unzip the provided archive.
3. Save the extracted `.spe` files under the matching folder name `IAEA_HPGe_spe` within your project's `data/` directory.

## Requirements
First, create an Anaconda environment with the required packages:
```bash
conda create --name gammai python=3.10.20 pyyaml=6.0.3 -y
conda activate gammai
```
Then, install the remaining dependencies using `pip`:
```bash
pip install -r requirements.txt
 ```

## Project Structure
Ensure your project follows this hierarchy before running the code:
```text
project_root/
+-- analyze_files/           <-- Scripts for parsing, converting, and visualizing raw experimental data (.cnf, .spe, .npy)
+-- checkpoints/             <-- Automatically generated model weights and training artifacts
+-- configs/                 <-- Centralized YAML configuration files (split into /experiments and /models)
+-- data/                    <-- Raw data folders (CTBT, IAEA, MCNP) and unified .npy matrix/scaling files
+-- models/                  <-- PyTorch model architectures (vae.py, diffwave.py)
+-- results/                 <-- Generated spectra arrays and plots organized by specific inference runs
+-- scripts/                 <-- Core execution scripts for training, inference, and latent space visualization
+-- utils/                   <-- Shared helper functions (helpers.py), metrics.py, and dataloaders (dataset.py)
+-- wandb/                   <-- Weights & Biases local sync directories, debug logs, and crash reports
+-- README.md                <-- Project documentation
```
>**Note:** Directories like `checkpoints/`, `results/`, and `wandb/` will generate their own subfolders dynamically during training and inference).

## Data Processing Pipeline
To analyze, parse the data, and build the matrix, run the following scripts from the root directory:

**1. Visualize raw files:**
```bash
python analyze_files/plot_spe.py
python analyze_files/plot_cnf.py  
python analyze_files/plt_npy.py
```

**2. Convert `.spe` to `.txt`:**
This script creates a new directory containing `.txt` files with the extracted channels and counts.
```bash
python analyze_files/spe2txt.py
```

**4. Build and save the data matrix:**
Reads the converted `.txt` files and compiles them into a single `.npy` matrix file for machine learning or statistical analysis.
```bash
python analyze_files/create_save_data_matrix.py
```
## Model Training
Before training, you can calculate the baseline Fréchet Distance (FD) and Kernel Inception Distance (KID) scores to establish target metrics for your dataset.
```bash
python utils/metrics.py
```
>**Note:** Update the `data_path` variable inside `metrics.py` if your dataset is stored in a non-default location.

Both the VAE and DiffWave models are controlled entirely via their respective .yaml configuration files found in the configs/ directory.

**To train the VAE:**
```bash
python scripts/train_vae.py --exp configs/experiments/vae_base.yaml --model configs/models/vae.yaml
```

**To train the DiffWave model:**
```bash
python scripts/train_diffwave.py --exp configs/experiments/diffwave_base.yaml --model configs/models/diffwave.yaml
```

## Model Inference 
>**Important:** Before running inference, ensure the `model_path` inside your experiment YAML files points to the specific checkpoint you wish to generate from.
### VAE

**Generate synthetic spectra using:**
```bash
python scripts/inference_vae.py --exp configs/experiments/vae_base.yaml --model configs/models/vae.yaml
```

**Latent Space Visualization & Clustering:**

Generate a 2D PCA plot of the latent space with overlaid, color-coded miniature spectra. Run with default configurations (5 clusters) or customize via arguments:

```bash
# Run with default configurations and 5 clusters
python scripts/plot_latent_space.py \
    --exp configs/experiments/vae_base.yaml \
    --model configs/models/vae.yaml \
    --clusters 8
```

### Diffusion
**Generate synthetic spectra using the DiffWave model:**
```bash
python scripts/inference_diffwave.py --exp configs/experiments/diffwave_base.yaml --model configs/models/diffwave.yaml
```

## Data Retrieval & Memorization Check

To verify the model generates novel data rather than memorizing the training set, this script calculates the Euclidean distance between your generated spectra and the original `.txt` training samples to find the nearest real neighbor.

**Run the analysis:**
Point `--results_dir` to the folder containing your generated `all_N_samples.npy` file:

```bash
python scripts/find_closest_spectra.py --results_dir results/YOUR_RUN_NAME/YOUR_TIMESTAMP_FOLDER
```

> **Note:** You can modify the script directly to change which `.txt` training data files are used for the comparison.