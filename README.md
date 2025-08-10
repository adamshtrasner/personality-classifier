# personality-classifier

## 📌 Project Overview

This project predicts a person’s personality type (**Introvert** or **Extrovert**) based on behavioral and social features such as time spent alone, social event attendance, and frequency of going outside.

The process included evaluating multiple machine learning algorithms, addressing class imbalance, optimizing the best-performing model, and deploying the final solution with an interactive interface.

This is a project which is part of a [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e7).

I also added an interactive web app to check whether a user is an introvert or an extrovert based on their input using the `Gradio` library from Python.

You can add your own model or improve it. Instructions on this is under [Instructions](#instructions).

---

## 💻 Dataset

The dataset for this competition (both train and test) was generated from a deep learning model trained on the [Extrovert vs. Introvert](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data) Behavior dataset.

Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition,

both to explore differences as well as to see whether incorporating the original in training improves model performance.

## 🔍 Models Considered

Eight classification models were evaluated:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Naive Bayes**
6. **Multi-Layer Perceptron (MLP)**
7. **XGBoost**
8. **LightGBM**

To address **class imbalance** (three times more Extroverts than Introverts), the parameter `class_weight='balanced'` was applied to models that support it. This ensured that the minority class received appropriate weight during training.

---

## 📊 Model Evaluation

TThe models were compared using **Accuracy**, **F1-score**, and **ROC AUC**:

- **KNN**, **XGBoost**, and **LightGBM** achieved the highest accuracy and F1-scores.
- **LightGBM** achieved the **best ROC AUC score**, making it the most reliable model for distinguishing between the two classes.

---

## ⚙️ Hyperparameter Optimization

The **LightGBM** model was tuned using **Optuna**, a hyperparameter optimization library.

**Best Hyperparameters:**

```
n_estimators: 181
learning_rate: 0.038099687186893334
num_leaves: 80
max_depth: 9
min_child_samples: 28
subsample: 0.8578050266925555
colsample_bytree: 0.5366587253794737
reg_alpha: 3.269238332213025
reg_lambda: 2.341665794508301
```

---

## 🤝 Model Stacking

To further improve performance, the optimized **LightGBM** model was stacked with:

- **Random Forest**
- **XGBoost**

A meta-learner combined predictions from all three models to make the final classification.

---

## 🏆 Final Model Performance

On the training set, the stacked model achieved:

- **Accuracy:** `0.9688`
- **F1-score:** `0.9398`
- **ROC AUC:** `0.9691`

On the Kaggle conest over a hidden test set:

- **Highest score**: 0.978137

- **My score**: 0.973279

---

## 🚀 Deployment

The final model was integrated into an **interactive web interface** using the **Gradio** library.  
The app asks the user seven questions corresponding to the dataset features and outputs the predicted personality type (**Introvert** or **Extrovert**) along with the model’s confidence score.

To ensure portability and easy deployment, the application was **containerized with Docker**:

- **Gradio** runs in a Python environment inside a Docker container.
- The container is configured to expose port `7860` so the app is accessible from any browser.
- The final setup includes:
  - A **model service** for experimentation and notebook work.
  - A **web service** running the Gradio UI connected to the trained model.

With this setup, the model can be deployed locally or to any cloud platform supporting Docker, enabling easy access without requiring any local Python setup.

---

## Instructions

### Virtual Enviroment

1. Create a virtual enviroment and run the `requirement.txt` files.

2. run: `web/src/app.py`.

You will then be given a local URL to the Gradio app in your terminal.

If you want to deploy your own model, change the code in `web/src/model.py`, run the code, and then run `web/src/app.py`.

### Docker

1. Run: `docker compose build` and then `docker compose up`.

2. In Docker Desktop, Click the port given under the "web" image.

If you want to play with the data and the model, click on the port under the "model" image.

This will direct you to a jupyter notebook. Notice that if you want your model to work you need to copy it to `web/src/app.py`

and change it accordingly.
