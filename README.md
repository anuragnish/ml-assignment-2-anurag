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

https://github.com/anuragnish/ml-assignment-2-anurag

## Live Streamlit app

https://anurag-ml-assignment-2.streamlit.app/

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
## Design Decisions and Learning Summary

### Design Decisions

**1. Dataset selection**

I selected the Breast Cancer Wisconsin (Diagnostic) dataset because it is a real-world binary classification problem with 569 observations and 30 numerical input features. The dataset satisfies the assignment requirement of at least 500 instances and 12 features.

The prediction target was encoded as:

* `1` — Benign
* `0` — Malignant

This encoding makes the positive-class evaluation metrics represent the model’s ability to identify benign cases.

**2. Consistent train-test split**

I used a stratified 80:20 train-test split with `random_state=42`. Stratification preserves the original class distribution in both subsets, while the fixed random state makes the experiment reproducible.

The test data was kept separate from model training to reduce the risk of data leakage and to ensure that every model was evaluated on exactly the same unseen observations.

**3. Model selection**

The application compares six classification algorithms:

* Logistic Regression
* Decision Tree
* k-Nearest Neighbours
* Gaussian Naive Bayes
* Random Forest
* Support Vector Machine

These models were selected because they represent different machine-learning approaches, including linear, probabilistic, distance-based, tree-based, ensemble and maximum-margin methods. Support Vector Machine was included as an additional sixth model to provide a broader comparison.

**4. Feature scaling**

Standardisation was applied to Logistic Regression, kNN and SVM because these algorithms are sensitive to differences in feature magnitude.

Scaling was implemented inside a scikit-learn pipeline so that the scaler was fitted only on the training data. The same fitted transformation was then applied to the test data, helping prevent leakage from the test set.

Decision Tree and Random Forest were trained without standardisation because tree-based models make decisions using feature thresholds and are generally unaffected by differences in feature scale. Gaussian Naive Bayes was also trained on the original numerical features.

**5. Evaluation approach**

All models were evaluated using the same fixed test set and the following metrics:

* Accuracy
* AUC
* Precision
* Recall
* F1-score
* Matthews Correlation Coefficient (MCC)

Accuracy alone can hide differences in the types of prediction errors made by a classifier. For this reason, I also used precision, recall, F1-score and MCC. MCC was particularly useful because it considers all four values in the confusion matrix and provides a balanced measure of classification quality.

AUC was calculated using predicted probabilities or decision scores rather than predicted class labels. This measures how effectively each model separates the two classes across different classification thresholds.

**6. Streamlit application design**

The Streamlit interface was designed specifically for this project rather than using an unchanged template. It includes:

* A customized project title and visual layout
* A downloadable compatible test dataset
* CSV file upload
* Input-schema validation
* A dropdown containing all six trained models
* Metric cards for the six required evaluation measures
* A confusion matrix
* A classification report
* An all-model comparison table
* Brief explanations of the selected algorithm and evaluation results

The schema validation checks whether the uploaded file contains the expected target and feature columns. This allows the application to display a clear message when an incompatible CSV file is supplied instead of failing with an unclear technical error.

### Results and Interpretation

Logistic Regression achieved the best overall result on the fixed test set, with:

* Accuracy: **0.9825**
* AUC: **0.9954**
* Precision: **0.9861**
* Recall: **0.9861**
* F1-score: **0.9861**
* MCC: **0.9623**

The strong performance of Logistic Regression suggests that the standardized diagnostic features provide good linear separation between the two classes.

kNN achieved a recall of **1.0000**, meaning that it correctly identified every benign case in the test set. However, its precision and overall MCC were slightly lower than those of Logistic Regression.

SVM also produced strong results, including an AUC of **0.9950** and an MCC of **0.9085**. Random Forest achieved an AUC of **0.9937**, showing strong ranking ability, although its final classification accuracy was lower than that of Logistic Regression.

The single Decision Tree produced the weakest overall result, with an accuracy of **0.9123** and an MCC of **0.8139**. This demonstrates that an individual tree can be sensitive to the training sample. Random Forest improved stability by combining predictions from multiple trees.

These results also show why model selection should not be based on only one metric. For example, the model with the highest recall was not the model with the highest MCC or accuracy.

### Learning Summary

This project helped me understand the complete machine-learning workflow, from dataset preparation and reproducible model training to evaluation, model persistence and deployment.

The main lessons I learned were:

* Data preprocessing must be selected according to the model rather than applied identically to every algorithm.
* Preprocessing should be included in a pipeline to prevent data leakage and ensure consistent predictions after deployment.
* Every model must be tested on the same unseen data for the comparison to be fair.
* Accuracy is insufficient on its own; AUC, precision, recall, F1-score and MCC provide different and complementary information.
* A confusion matrix is useful for understanding the exact types of errors hidden behind an overall score.
* Ensemble methods can reduce the instability of a single decision tree.
* The most complex model does not necessarily produce the best result.
* Saving the trained preprocessing pipeline together with the model is essential for reliable deployment.
* A deployed ML application should validate user data and provide meaningful error messages instead of assuming that every uploaded file is correctly formatted.

Overall, this assignment connected the theoretical concepts from the lectures with a working end-to-end classification system. It also demonstrated that reproducibility, leakage prevention, evaluation design and user-interface validation are as important as training the models themselves.

