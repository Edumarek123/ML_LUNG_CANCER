<div align="center">

# Lung Cancer Survey Classification

**An educational machine-learning experiment using Python and scikit-learn.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ExtraTrees-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-educational%20experiment-lightgrey)

</div>

> **Medical disclaimer:** This project is for educational purposes only. It is not a medical device, does not provide a diagnosis, and must never be used to make healthcare decisions.

## Overview

This repository explores a supervised classification workflow using a lung-cancer survey dataset. The Python script uses pandas and scikit-learn to prepare data, train an Extra Trees classifier, and produce an example prediction.

The focus is the machine-learning workflow — not clinical validation.

## Dataset

The repository includes:

- `survey_lung_cancer.csv` — survey dataset used by the experiment

Before using the data, review its columns, target definition, origin, and any applicable usage terms.

## Machine-learning workflow

1. Load the CSV dataset.
2. Inspect and prepare the feature data.
3. Separate model inputs and target labels.
4. Split data into training and testing subsets.
5. Fit an `ExtraTreesClassifier`.
6. Run an example inference.

## Technology stack

- Python
- pandas
- scikit-learn
- Extra Trees ensemble classifier

## Getting started

### Requirements

- Python 3
- pip

### Install dependencies

```bash
git clone https://github.com/Edumarek123/ML_LUNG_CANCER.git
cd ML_LUNG_CANCER
python3 -m pip install pandas scikit-learn
```

### Run

```bash
python3 lung_cancer.py
```

Run the script from the repository root so the relative dataset path resolves as expected.

## Repository layout

```text
.
├── lung_cancer.py
├── survey_lung_cancer.csv
└── README.md
```

## Limitations

Survey data and a simple train/test workflow do not establish clinical reliability. Dataset bias, class balance, data leakage, feature quality, and external validation all affect model behavior. No clinical performance claim is made here.

## Project status

Educational machine-learning experiment.

## License

No license file was identified in the repository at the time of writing. Contact the author before redistributing or reusing the project.
