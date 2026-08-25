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
conda create --name gammai python=3.10.20 pyyaml=6.0.3 -y
conda activate gammai
```
Then, install the remaining dependencies using `pip`:
```bash
pip install torch==2.6.0 torchvision torchaudio numpy==1.26.4 scipy==1.15.3 scikit-learn==1.7.2 matplotlib==3.10.9 tqdm==4.67.3 wandb==0.28.0
```

## Project Structure
Ensure your project follows this hierarchy before running the code:
```text
project_root/
+-- analyze_files/           <-- Scripts for parsing, converting, and visualizing raw experimental data (.cnf, .spe, .npy)
+-- checkpoints/             <-- Automatically generated model weights and training artifacts
|   +-- diff_7M_L36_C128_bs16_lr0.0002_cosine_ep1050/ <-- Example of a specific training run directory
|       +-- tracked_noise/   <-- Plots of diffusion noise schedules (alpha_cum_schedule.png, beta_schedule.png)
|       +-- tracked_signals/ <-- Visual comparisons of signals saved during training and evaluation
|       |   +-- eval/        <-- Generated plots from the fast-sampling evaluation loop
|       |   +-- train/       <-- Forward diffusion process plots (e.g., clean x_0 vs noisy x_100)
|       +-- wandb_id.txt     <-- Cached WandB run ID to seamlessly stitch resumed runs
|       +-- weights-2000.pt  <-- Step-specific checkpoint (contains weights, optimizer, and RNG states)
|       +-- weights.pt       <-- Symlink pointing to the most recent step checkpoint
+-- configs/                 <-- Centralized YAML configuration files (split into /experiments and /models)
+-- data/                    <-- Raw data folders (CTBT, IAEA, MCNP) and unified .npy matrix/scaling files
+-- models/                  <-- PyTorch model architectures (vae.py, diffwave.py)
+-- results/                 <-- Generated spectra arrays and plots organized by specific inference runs
|   +-- diff_7M_L36_C128_bs16_lr0.0002_cosine_ep1000/
|       +-- weights-106000_2026-08-12_08-34-28/   <-- Timestamped inference folder for a specific checkpoint
|           +-- denoising_process/     <-- Step-by-step visualizations of the reverse diffusion inference
|           +-- retrieval_comparisons/ <-- Visual and text logs comparing generated samples to nearest real data
|           +-- all_200_samples.npy    <-- The raw generated matrix of synthetic spectra
|           +-- spectrum_001.png       <-- Individual plots of the generated samples
+-- scripts/                 <-- Core execution scripts for training, inference, and latent space visualization
+-- utils/                   <-- Shared helper functions (helpers.py), metrics.py, and dataloaders (dataset.py)
+-- wandb/                   <-- Weights & Biases local sync directories, debug logs, and crash reports
+-- README.md                <-- Project documentation
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
This script reads the converted `.txt` files and saves that data into a single `.npy` matrix file for machine learning or statistical analysis.
```bash
python analyze_files/create_save_data_matrix.py
```
## Model Training
Before training your model, you can calculate the baseline Fréchet Distance (FD) and Kernel Inception Distance (KID) scores. This compares real data against real data to establish the "perfect" target scores for your dataset. 

Run the metrics script using the command below. Note: Update the `data_path` variable inside the script if your dataset is stored in a different location.

```bash
python utils/metrics.py
```

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

## Data Retrieval & Memorization Check

After generating synthetic spectra, it is crucial to evaluate whether the model is generating novel data or simply memorizing the training set (overfitting). The retrieval script calculates the Euclidean distance between the generated spectra and all original training samples directly from the `.txt` directories, finding the nearest real neighbor for each generated sample.

**To run the retrieval analysis:**
Run the following command, replacing the `--results_dir` path with your specific inference output directory (where the `all_N_samples.npy` file is located):

```bash
python scripts/find_closest_spectra.py --results_dir results/YOUR_RUN_NAME/YOUR_TIMESTAMP_FOLDER
