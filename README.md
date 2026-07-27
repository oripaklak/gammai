# HPGe Gamma Spectrum Processing & Generation Pipeline

## Introduction
Recent advances in data processing and machine learning have greatly improved the analysis of gamma-ray spectra. However, applying these methods directly to raw physical experiment files remains challenging. We propose this pipeline to parse raw high-purity germanium (HPGe) `.spe` files, convert them into clean `.txt` files, and compile them into a unified NumPy matrix. Finally, this project leverages deep generative models (1D Convolutional VAE and DiffWave) to learn the underlying physical distribution and generate highly realistic synthetic spectra.

## Dataset
The data used for this pipeline can be downloaded here: [https://www-nds.iaea.org/idb/](https://www-nds.iaea.org/idb/). 

To prepare the dataset:
1. Navigate to the link and select the **HPGe detector** option.
2. Download the provided archive.
3. Unzip the downloaded data.
4. Save the extracted `.spe` files under the matching folder name `IAEA_HPGe_spe` within your project's `data/` directory.

## Requirements
First, create an Anaconda environment with the required packages:
```bash
conda create --name gammai python=3.10 numpy pytorch scipy scikit-learn matplotlib pyyaml tqdm wandb -c pytorch -y
```

Then, activate your environment:
```bash
conda activate gammai
```

## Project Structure
Ensure your project follows this hierarchy before running the code:
```text
project_root/
+-- analyze_files/           <-- Scripts for parsing and visualizing raw experimental data
+-- checkpoints/             <-- Automatically generated model weights during training
+-- configs/                 <-- Centralized YAML configuration files for models
+-- data/                    <-- Raw data, text files, and unified matrix files
+-- models/                  <-- PyTorch model architectures (VAE, DiffWave)
+-- results/                 <-- Generated spectra arrays and plots from inference
+-- scripts/                 <-- Core execution scripts for training and inference
+-- utils/                   <-- Shared helper functions, metrics, and dataloaders
+-- README.md
```

## Data Processing Pipeline
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
## Model Training

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
*Note*: Before running inference, make sure the ```weights_to_use``` path inside your experiment YAML files (e.g., ```configs/experiments/vae_base.yaml```) points to the specific checkpoint you want to generate from.

### VAE

**Generate synthetic spectra using:**
```bash
python scripts/inference_vae.py --exp configs/experiments/vae_base.yaml --model configs/models/vae.yaml
```

**Latent Space Visualization & Clustering:**

To generate a 2D plot of the latent space with overlaid, color-coded miniature spectra (using PCA and K-Means), run the following command from the project root:

```bash
# Run with default configurations and 5 clusters
python scripts/plot_latent_space.py
```
You can customize the number of clusters or point the script to different configuration files using command-line arguments:
```bash
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

