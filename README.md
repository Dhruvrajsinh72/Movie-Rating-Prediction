# Movie-Rating-Prediction

This project is a **Movie Rating Prediction System** that utilizes machine learning techniques to analyze and predict the ratings of movies based on various features. The project uses a dataset containing information about movies, including their attributes and ratings, to train and evaluate predictive models.

## Features

- **Data Analysis**: Comprehensive analysis of the dataset, including data cleaning, visualization, and preprocessing.
- **Feature Engineering**: Extraction and transformation of relevant features to improve model performance.
- **Machine Learning Models**:
  - Linear Regression
  - Support Vector Machines (SVR)
- **Evaluation**: Performance evaluation using appropriate metrics like Mean Squared Error (MSE), R-squared, etc.

## Technology Stack

1. **Programming Language**: Python
2. **Libraries**:
   - Pandas and NumPy for data manipulation
   - Matplotlib and Seaborn for data visualization
   - Scikit-learn for machine learning algorithms and evaluation
3. **Dataset**: `imdbmoviesindia.csv`, containing details about movies such as genre, director, cast, and ratings.

## How It Works

1. **Data Loading**: The dataset is loaded and inspected to understand its structure and quality.
2. **Preprocessing**:
   - Handling missing values
   - Encoding categorical variables using `LabelEncoder`
   - Splitting data into training and testing sets
3. **Model Training**:
   - Train multiple machine learning models to predict movie ratings.
   - Tune hyperparameters for optimal performance.
4. **Evaluation**:
   - Use metrics such as MSE and R-squared to evaluate model performance.
5. **Visualization**: Generate plots to explore correlations and insights from the data.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruvrajsinh72/movie-rating-prediction.git
   cd movie-rating-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Movie_rating_Prediction.ipynb
   ```

## Results

- Models trained on the dataset achieved significant accuracy in predicting movie ratings.
- Insights from data visualization revealed correlations between genres, directors, and rating trends.

## Future Improvements

- Incorporate advanced machine learning models such as Random Forest, Gradient Boosting, or Neural Networks.
- Include sentiment analysis of user reviews as a feature for prediction.
- Automate hyperparameter tuning using tools like GridSearchCV or Optuna.
- Create a user-friendly web interface for real-time rating predictions.
