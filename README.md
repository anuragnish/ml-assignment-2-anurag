# Diagnostic ML Model Lab

## a. Problem statement

The project builds and compares classification models that predict whether a breast mass is **malignant (0)** or **benign (1)** from measurements derived from digitised fine-needle aspirate images. It also provides an interactive Streamlit application for uploading compatible test data, selecting a model, viewing all required metrics, and inspecting a confusion matrix and classification report.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Public source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569
- **Input features:** 30 continuous numeric features
- **Target:** Diagnosis; malignant = 0, benign = 1 in this implementation
- **Missing values:** None in the bundled scikit-learn copy
- **Split:** 80% training (455 rows), 20% testing (114 rows), stratified, `random_state=42`

The 30 predictors describe the mean, standard error, and worst/largest values of ten cell-nucleus characteristics, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. The dataset satisfies the assignment minimums of 12 features and 500 instances.

## c. GitHub repository link

**Replace before submission:** `https://github.com/anuragnish/ml-assignment-2-anurag

## Live Streamlit app

**Replace before submission:** `https://<YOUR-APP-NAME>.streamlit.app`

## d. Models used and comparison

Run `python train_models.py` to reproduce the saved models, `test_data.csv`, and exact metrics. The five models named by the assignment are mandatory. SVM is included as a clearly labelled sixth model because the assignment text states that six models must be implemented, although it lists only five.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9329 | 0.9429 | 0.9167 | 0.9296 | 0.8139 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| Support Vector Machine (Additional) | 0.9561 | 0.9950 | 0.9855 | 0.9444 | 0.9645 | 0.9085 |

> The values above are checked against the generated `model/metrics.json`. If the software version or split is changed, update this table from the regenerated file.

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best mandatory model by Accuracy, F1 and MCC. Its strong AUC indicates excellent ranking of malignant and benign cases. Standardisation helps because features use different measurement scales. |
| Decision Tree | Lowest AUC and MCC among the tested models. It remains interpretable, but a single tree is more sensitive to the particular training sample and makes more errors than the linear and ensemble approaches. |
| kNN | Perfect recall on this test split and the second-highest mandatory-model accuracy. Its lower precision than Logistic Regression means the gain in sensitivity comes with more false-positive benign predictions. |
| Gaussian Naive Bayes | High AUC and recall but lower accuracy/MCC. The conditional-independence and Gaussian assumptions are restrictive because several cell measurements are correlated. |
| Random Forest (Ensemble) | Strongest tree-based model. Bagging many decorrelated trees reduces the variance seen in the single Decision Tree and raises Accuracy from 0.9123 to 0.9474. |
| Support Vector Machine (Additional) | Very high AUC and precision, but lower recall than Logistic Regression and kNN on this split. It is included only to satisfy the brief's repeated reference to six models. |
| Overall Winner | **Logistic Regression among the five mandatory models.** It achieves the highest mandatory-model Accuracy (0.9825), F1 (0.9861) and MCC (0.9623), with AUC 0.9954. |

## Repository structure

```text
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    ├── random_forest.joblib
    ├── support_vector_machine.joblib
    ├── metadata.json
    └── metrics.json
```

## Local execution

1. Create and activate a Python virtual environment.
2. Install packages with `pip install -r requirements.txt`.
3. Rebuild artifacts if required with `python train_models.py`.
4. Start the application with `streamlit run app.py`.
5. Upload the supplied `test_data.csv`, select each model, and verify the metrics and confusion matrix.

## Streamlit Community Cloud deployment

1. Push the complete project folder to a public GitHub repository.
2. Sign in at [Streamlit Community Cloud](https://streamlit.io/cloud) using GitHub.
3. Select **New app**, choose the repository and `main` branch, and set the entry file to `app.py`.
4. Deploy and confirm that the URL opens without errors.
5. Test the upload, model dropdown, metrics, confusion matrix, and classification report.

## Academic-integrity note

The implementation uses explicit pipelines, upload validation, a customized diagnostic dashboard, reproducible artifacts, and project-specific observations. Review and understand every section, make your own Git commits while developing/testing, and do not submit placeholders.
