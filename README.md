# Aristotle-Chatbot-Cohere

This repository contains the updated version of the Aristotle Digital Immortality Chatbot, now utilizing Cohere's advanced language models, including Command R+. This project aims to simulate conversations with Aristotle, offering insights into his philosophy and teachings.

## Features

1. Enhanced Natural Language Processing: Leveraging Cohere’s Command R+ for more accurate and context-aware responses.

2. Improved Training Data: Updated and augmented dataset for better interaction quality.

3. Seamless Integration: Easy-to-use interface for interacting with the chatbot.
4. Scalability: Designed to handle a large number of queries with optimized performance.

## Project Structure

```
Aristotle-Chatbot-Cohere/
├── data/
│   ├── processed_aristotle_texts.json  # Your dataset containing processed texts
│
├── notebooks/
│   ├── data_preparation.ipynb          # Jupyter notebook for data preparation
│   ├── training_and_evaluation.ipynb   # Jupyter notebook for model training and evaluation
│
├── src/
│   ├── __init__.py
│   ├── cohere_client.py               # Code for initializing and interacting with Cohere API
│   ├── data_preparation.py            # Scripts for preparing and processing data
│   ├── train_model.py                 # Script to train the model using Cohere
│   ├── chatbot.py                     # Main script to run the chatbot
│
├── tests/
│   ├── test_chatbot.py                # Unit tests for the chatbot
│
├── .gitignore
├── LICENSE                            # Custom License file
├── README.md
├── requirements.txt                   # List of dependencies
└── app.py                             # Entry point to run the chatbot application
```


## Getting Started

### Prerequisites

- Python 3.7 or higher
- Cohere API Key

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/Aristotle-Chatbot-Cohere.git
cd Aristotle-Chatbot-Cohere
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up Cohere API Key:

- Obtain your API key from Cohere.
- Set it as an environment variable or directly in the code.

## Data Preparation

- Preprocess the PDF texts:
- Ensure your PDF files are in the specified folder.
- Run the data preparation script to extract and preprocess the texts:

```
python src/data_preparation.py
```

## Running the Application
1. Configure the API Key:

- Set your API key in the environment:
```
export COHERE_API_KEY='your-api-key-here'
```

- Alternatively, you can set it directly in your Python script where you initialize the Cohere client.

2. Run the application:
```
python app.py
```

## Usage

- Interact with the chatbot by typing your questions or prompts.

- The chatbot will respond with contextually relevant answers based on Aristotle's philosophy.

### Example
```
import cohere

# Initialize Cohere client
co = cohere.Client('your-api-key-here')

def get_response(prompt):
    response = co.generate(
        model='command-r',
        prompt=prompt,
        max_tokens=50
    )
    return response.generations[0].text

# Test the chatbot
prompt = "Tell me about Aristotle's philosophy."
response = get_response(prompt)
print(response)
```

## Notebooks

The notebooks folder contains Jupyter notebooks for data preparation and model training/evaluation.

### Data Preparation:

- data_preparation.ipynb: This notebook demonstrates the process of extracting and preprocessing text from PDF files.

### Training and Evaluation:

- training_and_evaluation.ipynb: This notebook demonstrates the process of training and evaluating the model using Cohere's Command R+.

To run these notebooks, navigate to the notebooks folder and start Jupyter Notebook:

```
jupyter notebook
```
Then, open the desired notebook to explore and execute the cells.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature-branch).
6. Open a pull request.


## License
This project is licensed under the Custom License. See the LICENSE.md file for details.

## Acknowledgements
- Thanks to Cohere for providing the language model.
- Inspired by the teachings and philosophy of Aristotle.