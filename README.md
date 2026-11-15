# Lung Cancer Prediction — Machine Learning Study

A small educational machine-learning project exploring classification on a lung-cancer survey dataset. The workflow uses Python and scikit-learn to prepare data, train an Extra Trees classifier, evaluate predictions, and run an example inference.

> **Disclaimer:** This repository is for learning and experimentation only. It is not a medical device, does not provide a diagnosis, and must not be used to make healthcare decisions.

## Project overview

The project demonstrates a basic supervised-learning pipeline:

1. Load and inspect the dataset with pandas.
2. Prepare the input features and target.
3. Split the data into training and test sets.
4. Train an `ExtraTreesClassifier`.
5. Generate predictions for an example patient record.

## Technologies

- Python
- pandas
- scikit-learn

## Running the project

1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install pandas scikit-learn
   ```
3. Run the project's Python script from the repository directory.

Make sure the dataset file is available at the path expected by the script.

## Scope and limitations

This is an exploratory study based on a public survey dataset. Results depend on the dataset, preprocessing choices, and evaluation setup. The project should not be interpreted as evidence of clinical accuracy or real-world diagnostic performance.

## Repository status

An educational machine-learning experiment. The implementation and dataset configuration may require adaptation before reuse.
