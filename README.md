# 

# \# MAX- | CISE2181 Machine Learning Midterm Project

# 

# \## 1. Project Overview

# 

# This project investigates small-data industrial defect detection using the MVTec AD dataset.

# 

# Track B: Anomaly Detection

# 

# The main method is a simplified PatchCore-style anomaly detection baseline using an ImageNet-pretrained WideResNet50\_2 backbone.

# 

# The model builds a memory bank from normal training images and calculates anomaly scores using nearest-neighbor distances.

# 

# \## 2. Dataset

# 

# Dataset: MVTec AD

# 

# Selected categories:

# \- Carpet (Texture)

# \- Bottle (Object)

# \- Hazelnut (Object)

# 

# Image size: 224 × 224

# 

# Data split:

# \- Official MVTec AD split

# \- 80% of normal training images for training

# \- 20% of normal training images for validation

# \- Original test set remains unchanged

# 

# The dataset is not included in this repository.

# 

# \## 3. Method

# 

# \- Backbone: WideResNet50\_2

# \- Pretrained weights: ImageNet

# \- Feature layers: layer2 and layer3

# \- Memory bank: Random patch sampling

# \- Memory bank maximum size: 1024

# \- Random seed: 42

# \- Image-level anomaly score: Maximum nearest-neighbor patch distance

# \- Evaluation metric: AUROC

# 

# Note: This is a simplified PatchCore-style baseline.

# It uses random memory sampling rather than the greedy coreset selection used in the original PatchCore method.

# 

# \## 4. Week 11 Baseline Results

# 

# | Category | Train | Validation | Test | AUROC |

# |----------|------:|-----------:|-----:|------:|

# | Carpet | 224 | 56 | 117 | 0.9803 |

# | Bottle | 167 | 42 | 83 | 0.9857 |

# | Hazelnut | 312 | 79 | 110 | 0.9907 |

# | Mean | - | - | - | 0.9856 |

# 

# These are actual image-level AUROC results from a single-seed experiment using the simplified PatchCore-style baseline.

# 

# \## 5. Requirements

# 

# \- Python 3.12

# \- PyTorch

# \- Torchvision

# \- NumPy

# \- Pillow

# \- scikit-learn

# 

# Install the required Python packages:

# 

# ```bash

# python -m pip install torch torchvision numpy pillow scikit-learn

# ```

# 

# \## 6. How to Run

# 

# Place the MVTec AD dataset in the following structure:

# 

# ```text

# data/

# ├── carpet/

# │   ├── train/

# │   ├── test/

# │   └── ground\_truth/

# ├── bottle/

# ├── hazelnut/

# ```

# 

# Run each category separately:

# 

# ```bash

# python patchcore.py --category carpet

# python patchcore.py --category bottle

# python patchcore.py --category hazelnut

# ```

# 

# Results are saved to:

# 

# ```text

# outputs/<category>/metrics.json

# outputs/<category>/split\_manifest.json

# ```

# 

# Note: The current script uses Windows-specific paths under C:\\MAX. Modify ROOT and OUTPUT in patchcore.py if the project is located elsewhere.

# 

# \## 7. Planned Experiments

# 

# Future work includes:

# 

# \- PatchCore hyperparameter comparisons

# \- Horizontal flip (p = 0.5)

# \- Rotation (±10 degrees)

# \- Brightness adjustment (±10%)

# \- Contrast adjustment (±10%)

# \- Three random-seed experiments

# \- Error analysis

# \- A supervised Track A CNN baseline on one category, as required by the course

# 

# \## 8. References

# 

# Bergmann et al., MVTec AD, CVPR 2019.

# 

# Roth et al., Towards Total Recall in Industrial Anomaly Detection (PatchCore), CVPR 2022.

# 

# CISE2181 Machine Learning Midterm Project Instructions, Fall 2026.



