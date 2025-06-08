
# Podcast Listening Time Prediction

## Project Overview
This project aims to predict the listening time of podcast episodes based on various features such as episode length, genre, host popularity, guest popularity, publication day, publication time, number of ads, and episode sentiment. The project utilizes a linear regression model implemented in Python to make predictions on a dataset containing podcast episode metadata.

The dataset consists of a training set (`train.csv`) with 750,000 entries and a test set (`test.csv`) with 250,000 entries. The goal is to predict the `Listening_Time_minutes` for the test set and generate a submission file (`submission.csv`) for evaluation.

## Repository Structure
- **podcast_time.ipynb**: The main Jupyter Notebook containing the data preprocessing, model training, and prediction pipeline.
- **train.csv**: The training dataset with podcast episode features and listening time (not included in the repository due to size).
- **test.csv**: The test dataset for generating predictions (not included in the repository due to size).
- **submission.csv**: The output file containing predicted listening times for the test set.
- **README.md**: This file, providing an overview of the project and instructions for use.

## Dataset Description
The dataset includes the following columns:
- **Podcast_Name**: Name of the podcast.
- **Episode_Title**: Title of the episode.
- **Episode_Length_minutes**: Duration of the episode in minutes.
- **Genre**: Category of the podcast (e.g., Comedy, Education, True Crime).
- **Host_Popularity_percentage**: Popularity score of the host (0-100).
- **Publication_Day**: Day of the week the episode was published.
- **Publication_Time**: Time of day the episode was published (Morning, Afternoon, Evening, Night).
- **Guest_Popularity_percentage**: Popularity score of the guest (0-100, if applicable).
- **Number_of_Ads**: Number of advertisements in the episode.
- **Episode_Sentiment**: Sentiment of the episode (Positive, Neutral, Negative).
- **Listening_Time_minutes**: Target variable, the total listening time in minutes (available in training data only).

## Methodology
1. **Data Preprocessing**:
   - Loaded the training and test datasets using pandas.
   - Dropped irrelevant columns (`id`, `Podcast_Name`, `Episode_Title`) as they are not used for modeling.
   - Handled missing values by filling them with the mean for numerical columns (`Episode_Length_minutes`, `Guest_Popularity_percentage`, `Number_of_Ads`).
   - Encoded categorical variables:
     - `Publication_Day` mapped to numerical values (Monday=1, ..., Sunday=7).
     - `Episode_Sentiment` mapped to numerical values (Negative=0, Neutral=1, Positive=2).
     - `Genre` and `Publication_Time` converted to dummy variables using one-hot encoding.
2. **Model Training**:
   - Used a `LinearRegression` model from scikit-learn to predict `Listening_Time_minutes`.
   - Split the training data into features (X) and target (y), then trained the model.
3. **Prediction**:
   - Applied the same preprocessing steps to the test dataset.
   - Generated predictions using the trained model.
   - Created a submission file (`submission.csv`) with columns `id` and `Listening_Time_minutes`.

## Installation and Setup
To run this project locally, follow these steps:

### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/BilalIrfanContact/Podcast-Listening-Time-Prediction.git
   cd podcast-listening-time-prediction
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Ensure you have the `train.csv` and `test.csv` datasets in the project directory.

### Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `podcast_time.ipynb` and run all cells to preprocess the data, train the model, and generate predictions.
3. The output will be saved as `submission.csv` in the project directory.

## Usage
- **Training**: Run the notebook cells up to the model training section to train the linear regression model on the training data.
- **Prediction**: Continue running the notebook to preprocess the test data and generate predictions.
- **Output**: The predictions are saved in `submission.csv`, which can be submitted to a competition platform or used for further analysis.

## Results
The model uses linear regression to predict listening time based on the provided features. The performance of the model can be evaluated using metrics like Root Mean Squared Error (RMSE) and R² score on a validation set (not implemented in the current notebook but can be added).

## Future Improvements
- **Feature Engineering**: Explore additional features, such as interaction terms or episode title analysis (e.g., using NLP to extract sentiment or topics).
- **Model Selection**: Experiment with more complex models like Random Forest, Gradient Boosting, or neural networks to improve prediction accuracy.
- **Hyperparameter Tuning**: Implement cross-validation and grid search to optimize model performance.
- **Handling Missing Values**: Explore alternative imputation methods (e.g., median, KNN imputation) for missing values.
- **Evaluation Metrics**: Add validation steps to compute RMSE and R² on a holdout set to assess model performance.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Contact
For questions or feedback, please contact [bilalirfancontact@gmail.com](mailto:bilalirfancontact@gmail.com) or open an issue on GitHub.

