
# Practical Work in AI JKU
# Sound Event Detection with Transformers

This repository contains the practical work for the AI course at JKU, focusing on Sound Event Detection (SED) using pre-trained transformers and the DCASE 2023 baseline system.

---

## Overview

Sound Event Detection aims to identify and classify sound events in audio streams, providing temporal information for their occurrence. This project reproduces the DCASE 2023 baseline system and integrates pre-trained transformers to enhance performance.

---

## Project Workflow

### Clone DCASE Repository
```bash
git clone https://github.com/DCASE-REPO/DESED_task.git
```

---

### Install Needed Dependencies
```bash
apt-get install -y sox libsox-dev libsox-fmt-all
pip install desed
pip install pytorch_lightning
pip install sed_scores_eval
pip install codecarbon
pip install psds_eval
pip install thop
pip install torchlibrosa
```

---

### Note on Pretrained BEATS Model
The weights for the 'BEATS' model are corrupted in the official repository of 2023. Ensure you download the pretrained BEATS model for extracting embeddings from the following link:  
[Download BEATS Pretrained Model](https://onedrive.live.com/?authkey=%21AGOyB4YHPatKU%2D0&id=6B83B49411CA81A7%2125958&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp)

---

### Dataset Preparation
After downloading the repository, navigate to it and download the dataset:
```bash
cd DESED_task/recipes/dcase2023_task4_baseline
python generate_dcase_task4_2023.py --only_synth
python generate_dcase_task4_2023.py --only_real
```
Use these commands if you do not need the strong labeled training set.

---

### Pre-compute Embeddings
```bash
python extract_embeddings.py --output_dir ./embeddings --pretrained_model "beats"
```
Ensure the embeddings are stored in the `embeddings` folder.

---

### Train the Model
Run the training script:
```bash
python train_pretrained.py --test_from_checkpoint /path/to/downloaded.ckpt
```

---

## Additional Testing with Pretrained SED Model
The BEATS model provided by the DESED repository was tested on the dataset. Additionally, the weakly-supervised BEATS model was tested using the repository [PretrainedSED](https://github.com/fschmid56/PretrainedSED?tab=readme-ov-file).

### Testing BEATS Weak Model
To test the weakly-supervised BEATS model:
1. Download the model weights from the following link:  
[BEATS Weak Model Weights](https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf)
2. Place the downloaded file `BEATs_weak.pt` in the `embeddings/beats` folder.
3. Modify the training script to use the new model path:
```bash
python train_pretrained.py --test_from_checkpoint /path/to/new_model.pt
```

### Results
- BEATS Model provided by DESED_task:  
  **PSDS_scenario 1:** 0.463
- Weakly-Supervised BEATS Model:  
  **PSDS_scenario 1:** 0.47

---
