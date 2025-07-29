# fakenewsdetector 📰
This project is a simple system designed to classify news articles as either "fake" or "not fake" using machine learning. It includes a script to train the classification models and a web API built with Flask to make predictions.

## Features

* **News Classification**: Determines if a given text is fake or real news.
* **Machine Learning Models**: Uses trained models to perform the classification.
* **Web API**: Provides an easy way to get predictions by sending text to an endpoint.
* **Text Preprocessing**: Cleans and prepares text for analysis.

## How to Run

Follow these steps to set up and run the project:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/git-samia/fakenewsdetector.git](https://github.com/git-samia/fakenewsdetector.git)
    cd fakenewsdetector
    ```

2.  **Install dependencies:**
    Ensure you have Python 3 and `pip` installed.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Dataset:**
    Create a folder structure like `fake-newzz/Datasets/` and place your `Fake.csv` and `True.csv` files inside it.

    Example structure:

    ```
    fakenewsdetector/
    ├── fake-newzz/
    │   └── Datasets/
    │       ├── Fake.csv
    │       └── True.csv
    └── ... (other project files)
    ```

4.  **Train the Models:**
    Run the training script to prepare the machine learning models. This will also download necessary NLTK data.

    ```bash
    python train_models.py
    ```

5.  **Run the Flask API:**
    Start the web service.

    ```bash
    python app.py
    ```

    The API will be available at `http://0.0.0.0:5000`.

6.  **Make Predictions (API Usage):**
    Send a POST request to `http://0.0.0.0:5000/predict` with your news text in JSON format:

    ```json
    {
        "text": "Your news article text goes here."
    }
    ```

## Technologies Used

* Python 3
* Flask (Web API)
* scikit-learn (Machine Learning)
* NLTK (Natural Language Processing)
* Pandas (Data Handling)

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or want to contribute code, please feel free to open an issue or submit a pull request on the GitHub repository.

## License

No specific license defined.
