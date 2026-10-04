# Logistic Regression Studies

This project explores logistic regression through two binary-classification tasks: email spam detection and car detection. The spam-detection notebook is complete; car detection is still under research.

## Project status

| Task | Status | Summary |
| --- | --- | --- |
| Email spam detection | Completed | Uses TF-IDF features and a custom logistic regression implementation to classify email as spam or non-spam. |
| Car detection | Under research | Contains vehicle/non-vehicle image data, exploratory notebooks, and a car-features spreadsheet. The workflow is not yet presented as a completed model. |

### Email spam detection results

The notebook splits the 5,728 labeled emails into training and test sets using an 80/20 stratified split. The saved classification report shows **79% test accuracy**. For the spam class, precision is **1.00**, but recall is only **0.12** (F1-score **0.22**), so the current model misses most spam emails despite the reported accuracy.

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
src/
  email spam/
    LogisticRegressionNotebook.ipynb
  car detection/
    DataProcessing.ipynb
    LogisticRegressionNotebook.ipynb
```

The spam dataset has `text` and `spam` columns, where `spam` is the binary label. The reusable logistic regression implementation is in [`clazz/LogisticRegression.py`](clazz/LogisticRegression.py).

## Running the spam notebook

The notebook uses paths relative to its working directory. From the repository root, start Jupyter from the email-spam source folder:

```bash
cd "src/email spam"
jupyter notebook LogisticRegressionNotebook.ipynb
```

The notebook builds TF-IDF features, makes a stratified train/test split, trains the project's logistic regression implementation, and prints classification reports. Its current feature-building code creates the vocabulary and IDF values before splitting the data; consider fitting those steps on the training data only when revisiting the evaluation.

## Car detection research

The car-detection materials are exploratory and are not a finished end-to-end detection system. Images are organized into `vehicles` and `non-vehicles` classes under `dataset/car detection/color/`; feature data is in [`assets/car_features.xlsx`](assets/car_features.xlsx). See [`DataProcessing.ipynb`](<src/car detection/DataProcessing.ipynb>) and [`LogisticRegressionNotebook.ipynb`](<src/car detection/LogisticRegressionNotebook.ipynb>) for the current notebooks. Check and update their data paths for your working directory before running them.

## Dependencies

There is currently no dependency manifest. The email notebook uses Python with Jupyter, NumPy, pandas, Matplotlib, Seaborn, and scikit-learn. The car data-processing notebook additionally imports OpenCV (`opencv-python`) and Gradio.
