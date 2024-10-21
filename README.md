Here’s a structured `README.md` file for your GitHub project on "Product Recommendation System Using Sentiment Analysis":

---

# Product Recommendation System Using Sentiment Analysis

## Overview

This project focuses on building a **Product Recommendation System** that leverages **Sentiment Analysis** to enhance recommendation quality. The system was developed as part of a project at **IIT Roorkee**, utilizing Machine Learning (ML) and Natural Language Processing (NLP) techniques. The main objective is to recommend products more accurately by combining traditional collaborative filtering with sentiment scores derived from product reviews.

## Project Structure

- **Sentiment Analysis:** Used various ML classifiers to predict sentiment from product reviews.
- **Recommendation System:** Built a recommendation system using both **user-based** and **item-based filtering** techniques.
- **Sentiment Integration:** Incorporated sentiment scores into the final recommendation system to further improve accuracy.

## Features

- **Text Preprocessing:** Utilized the **Natural Language Toolkit (NLTK)** for cleaning and preprocessing text data.
- **Feature Extraction:** Applied **TF-IDF** and **Word2Vec** techniques to extract features from a dataset with over **30,000 product reviews**.
- **Machine Learning Models:** Implemented the following classifiers for sentiment analysis:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - XGBoost (XGB)
  - AdaBoost (AB)
  - K-Nearest Neighbors (KNN)
  - **Support Vector Machine (SVM)** achieved the best performance, fine-tuned using **5-fold cross-validation**.
- **Recommender System:**
  - Utilized **cosine similarity** for both user-user and item-item collaborative filtering.
  - Final model selected was **User-User based collaborative filtering** with an **RMSE of 1.6**.
  - Integrated sentiment scores into the recommendation engine for enhanced performance.

## Key Technologies

- **Python:** Programming language for model development and system integration.
- **NLTK:** For text preprocessing (tokenization, stemming, etc.).
- **TF-IDF & Word2Vec:** Feature extraction methods for text data.
- **Sci-kit Learn, XGBoost, and AdaBoost:** ML libraries for training classifiers.
- **Cosine Similarity:** Distance metric used in collaborative filtering.
- **Collaborative Filtering:** User-based and item-based recommendation approaches.

## Model Performance

- **Sentiment Analysis:** SVM performed best, outperforming other classifiers.
- **Recommendation System:** The user-user based collaborative filtering model with sentiment integration significantly improved recommendation accuracy.

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Omshrivastav12/product-recommendation-sentiment-analysis.git
   cd product-recommendation-sentiment-analysis
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the sentiment analysis and recommendation scripts:**
   ```bash
   python sentiment_analysis.py
   python recommendation_system.py
   ```

4. **Modify and integrate your data:**
   Replace `data.csv` with your dataset containing product reviews for recommendation.

## Future Improvements

- Add deep learning models for better sentiment analysis accuracy.
- Experiment with hybrid recommendation approaches combining content-based filtering and collaborative filtering.
- Optimize runtime for larger datasets.

## Contact

For any questions or issues, feel free to contact me:

- **Name:** Om Subhash Shrivastav
- **Email:** omshrivastav1005@gmail.com
- **GitHub:** [Omshrivastav12](https://github.com/Omshrivastav12)

---

You can copy and paste this content into your `README.md` file for a clear explanation of the project.
