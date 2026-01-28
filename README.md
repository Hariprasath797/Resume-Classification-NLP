# Resume Classification using NLP

This project focuses on automating resume screening using Natural Language Processing (NLP) 
and Machine Learning. The system classifies resumes into predefined job categories, helping 
reduce manual effort in HR processes.

## Business Objective
To build an automated resume classification solution that minimizes human intervention 
while improving efficiency in resume screening.

## Dataset
- Resume documents in DOCX format
- Categorized resumes (e.g., Peoplesoft, Workday, etc.)
- Real-world structured resume data used for experimentation

## Models Experimented
Nearly 10 different machine learning models were trained and evaluated, including:
- KNN Classifier
- Naive Bayes Classifier
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Bagging Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier

## Model Evaluation
Each model was evaluated using the following metrics:
- Train Accuracy
- Test Accuracy
- Precision
- Recall
- F1-Score

Some models achieved very high accuracy due to the structured nature and limited size of the dataset. 
The primary goal of this project was to develop a functional prototype for resume classification 
rather than a large-scale production system.

## Final Model Selection
Although multiple models performed well, **Decision Tree Classifier** was selected as the final model 
based on:
- Strong performance on evaluation metrics
- Better interpretability
- Simplicity in deployment
- Suitability for a prototype-level HR automation system

## Project Workflow
1. Resume data collection (DOCX files)
2. Text preprocessing (cleaning, normalization)
3. Feature extraction using TF-IDF
4. Training and evaluating multiple ML models
5. Final model selection
6. Web application deployment

## Deployment
The final trained model is deployed using **Streamlit**, allowing users to upload resumes 
and receive predicted job categories through a web interface.

## Future Enhancements
- Increase dataset size
- Apply cross-validation to reduce overfitting
- Compare with deep learning-based NLP models
- Cloud deployment (AWS / GCP)

## Author
Hari Prasath P
