![image](https://github.com/user-attachments/assets/1006c0ba-54a5-4f5a-bd98-8f3842e3102e)
# Podcast Listening Time Prediction

This project predicts the listening time of podcast episodes based on features such as episode length, genre, host popularity, guest popularity, publication day, publication time, number of ads, and episode sentiment. It includes two components:
1. **Jupyter Notebook (`podcast_time.ipynb`)**: Contains data analysis, preprocessing, and linear regression model training on a dataset of 750,000 podcast episodes.
2. **Flask Web Application (`app.py`)**: A web interface for users to input podcast features and receive predicted listening times using the trained model.

The dataset consists of a training set (`train.csv`) with 750,000 entries and a test set (`test.csv`) with 250,000 entries. The goal is to predict the `Listening_Time_minutes` for podcast episodes, with the notebook generating a submission file (`submission.csv`) and the Flask app providing real-time predictions.

## 📂 Repository Structure
- **app.py**: Flask application script for the web interface, handling model loading, preprocessing, and predictions.
- **podcast_time.ipynb**: Jupyter notebook with data exploration, preprocessing, and model training.
- **templates/**: Folder containing HTML templates for the Flask app.
  - **index.html**: Web interface for inputting podcast features and displaying predictions.
- **train.csv**: Training dataset with podcast episode features and listening time.
- **test.csv**: Test dataset for generating predictions.
- **model.pkl**: Serialized trained model (generated on first run of `app.py`; excluded from repository).
- **submission.csv**: Output file with predictions for the test set (generated by `podcast_time.ipynb`).
- **README.md**: This file, providing project overview and instructions.
- **.gitignore**: Excludes `model.pkl` and Python artifacts (e.g., `__pycache__/`).

## 📊 Dataset Description
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

## 🔧 Methodology
### Jupyter Notebook (`podcast_time.ipynb`)
1. **Data Preprocessing**:
   - Loaded `train.csv` and `test.csv` using pandas.
   - Dropped irrelevant columns (`id`, `Podcast_Name`, `Episode_Title`).
   - Filled missing values with means for numerical columns (`Episode_Length_minutes`, `Guest_Popularity_percentage`, `Number_of_Ads`).
   - Encoded categorical variables:
     - `Publication_Day` mapped to numerical values (Monday=1, ..., Sunday=7).
     - `Episode_Sentiment` mapped to numerical values (Negative=0, Neutral=1, Positive=2).
     - `Genre` and `Publication_Time` converted to dummy variables using one-hot encoding.
2. **Model Training**:
   - Trained a `LinearRegression` model from scikit-learn on the preprocessed training data.
3. **Prediction**:
   - Applied preprocessing to `test.csv` and generated predictions, saved as `submission.csv`.

### Flask Application (`app.py`)
1. **Model Loading/Training**:
   - Loads `model.pkl` if available; otherwise, trains a linear regression model using `train.csv` and saves it as `model.pkl`.
2. **Web Interface**:
   - Provides a form at `http://127.0.0.1:5000` for users to input podcast features.
   - Processes inputs with the same preprocessing as the notebook and predicts listening time.
3. **Prediction Output**:
   - Displays the predicted listening time on the webpage.

## ⚙️ Installation and Setup
### Prerequisites
- Python 3.11 or higher
- Jupyter Notebook (for running `podcast_time.ipynb`)
- Required Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, flask

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/BilalIrfanContact/Podcast-Listening-Time-Prediction.git
   cd Podcast-Listening-Time-Prediction
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn flask
   ```

### Running the Notebook
1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `podcast_time.ipynb` and run all cells to preprocess data, train the model, and generate `submission.csv`.

### Running the Flask App
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open a browser and go to `http://127.0.0.1:5000`.
3. Enter podcast features in the form and click "Predict" to see the predicted listening time.

## 🚀 Usage
- **Notebook**:
  - Run `podcast_time.ipynb` to explore the dataset, train the model, and generate predictions for `test.csv`.
  - Output is saved as `submission.csv`.
- **Flask App**:
  - Access the web interface to input podcast features and get real-time predictions.
  - The app automatically loads or trains the model using `train.csv`.

## 📈 Results
The linear regression model predicts listening time based on the provided features. Performance can be evaluated using metrics like Root Mean Squared Error (RMSE) and R² score on a validation set (not implemented in the current notebook but can be added).

## 🔮 Future Improvements
- **Feature Engineering**: Add interaction terms or analyze episode titles using NLP for additional features.
- **Model Selection**: Experiment with advanced models (e.g., Random Forest, Gradient Boosting) for better accuracy.
- **Hyperparameter Tuning**: Implement cross-validation and grid search for model optimization.
- **Flask Enhancements**: Add input validation, improve UI with CSS frameworks (e.g., Bootstrap), or support batch predictions via file uploads.
- **Deployment**: Deploy the Flask app to a cloud platform (e.g., Heroku, AWS) for public access.

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## 📬 Contact
For questions or feedback, contact [bilalirfancontact@gmail.com](mailto:bilalirfancontact@gmail.com) or open an issue on GitHub.
