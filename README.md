# 🧠 Bully Barrier - Social Media Risk Detection Pipeline

## Detecting False Positives and False Negatives in Self-Harm and Abuse Posts

This repository contains a full machine learning pipeline for analyzing social media posts to detect **self-harm** and **abuse-related content**, with a focus on identifying **false positives** and **false negatives** in classification results.

---

## 📌 Objective

To build a robust ML system that classifies sensitive social media posts (self-harm, abuse) and flags misclassifications (i.e., FP/FN), ensuring high precision and recall for real-world safety and moderation systems.

---

## 🛠️ Pipeline Overview

-   **Data Collection**: The process begins with raw social media data, which is loaded for cleaning and processing.
-   **Data Preprocessing**: Text data is cleaned by converting it to lowercase, removing URLs, special characters, and extra spaces.
-   **Feature Engineering**: The cleaned text is then vectorized using TF-IDF to convert the text data into a numerical format suitable for machine learning models.
-   **Model Training & Validation**: A Logistic Regression model is trained on the processed data. The model is configured to handle class imbalance by using the `class_weight='balanced'` parameter. The data is split into training and testing sets, with 80% used for training and 20% for testing.
-   **Error Analysis (False Positives/Negatives)**: The model's performance is evaluated using a classification report and a confusion matrix to analyze false positives and false negatives.
-   **Reporting & Visualization**: The results, including precision, recall, and F1-score, are reported to assess the model's effectiveness in identifying sensitive content.

---

## 🛠️ Technologies Used

This project utilizes the following key technologies and libraries:

-   **pandas**: For data manipulation and analysis.
-   **scikit-learn**: For machine learning, including `TfidfVectorizer` for text vectorization and `LogisticRegression` for classification.
-   **NLTK**: For natural language processing tasks such as tokenization and stopword removal.
-   **matplotlib & seaborn**: For data visualization and creating plots like the label distribution and text length distribution.
-   **Jupyter Notebook**: For interactive development, data exploration, and model training.

---

## ⚙️ Installation

To get started with this project, follow these steps:

1.  **Clone the repository**:
    ```sh
    git clone [https://github.com/ggwellplayed4101/BullyBarrier.git](https://github.com/ggwellplayed4101/BullyBarrier.git)
    cd BullyBarrier
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

The project is organized into a series of Jupyter notebooks that walk through the entire pipeline. You can run the notebooks in the following order to reproduce the results:

1.  `notebooks/01_data_cleaner.ipynb`: Cleans the raw data and saves it to a CSV file.
2.  `notebooks/02_data_exploration.ipynb`: Explores the cleaned data to understand its characteristics.
3.  `notebooks/03_data_preprocessing.ipynb`: Preprocesses the text data and creates the TF-IDF matrix.
4.  `notebooks/04_modeling.ipynb`: Trains the Logistic Regression model and saves it.
5.  `notebooks/05_evaluation.ipynb`: Evaluates the trained model's performance.

To run the notebooks, start the Jupyter Notebook server:

```sh
jupyter notebook
