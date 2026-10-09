# Logistic Regression Studies

This project explores logistic regression through two binary-classification tasks: email spam detection and car detection. Both tasks are now complete, and each notebook demonstrates a working end-to-end training and evaluation pipeline.

## Project status

| Task | Status | Summary |
| --- | --- | --- |
| Email spam detection | Completed | Uses TF-IDF features and a custom logistic regression implementation to classify email as spam or non-spam. |
| Car detection | Completed | Uses image features extracted from vehicle and non-vehicle samples, then classifies each sample with logistic regression. |

### Email spam detection results

The notebook splits the 5,728 labeled emails into training and test sets using an 80/20 stratified split. The saved classification report shows **79% test accuracy**. For the spam class, precision is **1.00**, but recall is only **0.12** (F1-score **0.22**), so the current model misses most spam emails despite the reported accuracy.

### Car detection results

The car-detection workflow builds a dataset from the image folders in `dataset/car detection/color/`, exports processed features to `dataset/car detection/csv/vehicle_dataset.csv`, and trains the project's logistic regression implementation on those feature vectors. The notebook reports:

- Training accuracy: **1.00**
- Test accuracy: **0.99**

This indicates the model separates vehicle and non-vehicle samples with very high accuracy on the provided dataset.

## Repository layout

```text
assets/
  car_features.xlsx
  LogisticRegressionProjectPlan.png
clazz/
  LogisticRegression.py
dataset/
  email spam/
    emails.csv
  car detection/
    color/
      non-vehicles/
      vehicles/
    csv/
      vehicle_dataset.csv
src/
  email spam/
    LogisticRegressionNotebook.ipynb
  car detection/
    DataProcessing.ipynb
    LogisticRegressionNotebook.ipynb
```

The spam dataset has `text` and `spam` columns, where `spam` is the binary label. The reusable logistic regression implementation is in [`clazz/LogisticRegression.py`](clazz/LogisticRegression.py).

## Running the notebooks

From the repository root, start Jupyter in the relevant source folder:

```bash
cd "src/email spam"
jupyter notebook LogisticRegressionNotebook.ipynb
```

```bash
cd "src/car detection"
jupyter notebook LogisticRegressionNotebook.ipynb
```

The email notebook builds TF-IDF features, makes a stratified train/test split, trains the project's logistic regression implementation, and prints classification reports. Its current feature-building code creates the vocabulary and IDF values before splitting the data; consider fitting those steps on the training data only when revisiting the evaluation.

The car-detection notebook reads the prepared dataset from `dataset/car detection/csv/vehicle_dataset.csv` and evaluates the trained classifier on the held-out test set. The data-processing notebook prepares the image-derived feature table and should be run first when regenerating the CSV dataset.

## Dependencies

There is currently no dependency manifest. The email notebook uses Python with Jupyter, NumPy, pandas, Matplotlib, Seaborn, and scikit-learn. The car data-processing notebook additionally imports OpenCV (`opencv-python`) and Gradio.
