# Movie Review Sentiment Analysis

This project aims to predict the sentiment (Positive or Negative) of movie reviews using **Sentence Transformers** (specifically `mpnet` from Hugging Face) for text embeddings. **Random Forest** is used as the classification model, along with hyperparameter tuning using **Grid Search** and **Random Search**.

## 🧠 About

The goal of this project is to classify movie reviews as either **Positive** or **Negative** based on the text content. We use **Sentence Transformers** from Hugging Face (`mpnet`) to create vector embeddings of the reviews, which are then fed into a **Random Forest** classifier. To optimize the model's performance, hyperparameter tuning is performed using both **Grid Search** and **Random Search**.

## 🚀 Key Steps

- **Text Embedding**: Used **Sentence Transformers** (specifically the `mpnet` model from Hugging Face) to transform movie review texts into dense vector embeddings.
- **Modeling**: Built a **Random Forest classifier** to predict the sentiment based on the generated embeddings.
- **Hyperparameter Tuning**: Performed **Grid Search** and **Random Search** for tuning hyperparameters to improve the model's performance.

## 🔍 Dataset

- **Source**: The dataset typically consists of labeled movie reviews, where each review is classified as either **Positive** or **Negative**.
- **Preprocessing**: Tokenization, lowercasing, and embedding using **Sentence Transformers**.

## 🧾 Requirements

- sentence-transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- transformers

---
