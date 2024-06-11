# Aristotle-Chatbot-Cohere

This repository contains the updated version of the Aristotle Digital Immortality Chatbot, now utilizing Cohere's advanced language models, including Command R+. This project aims to simulate conversations with Aristotle, offering insights into his philosophy and teachings.

## Features

1. Enhanced Natural Language Processing: Leveraging Cohere’s Command R+ for more accurate and context-aware responses.

2. Improved Training Data: Updated and augmented dataset for better interaction quality.

3. Seamless Integration: Easy-to-use interface for interacting with the chatbot.
4. Scalability: Designed to handle a large number of queries with optimized performance.


## Getting Started

### Prerequisites

Python 3.7 or higher
Cohere API Key

## Installation

1. Clone the repository:

bash
```
git clone https://github.com/yourusername/Aristotle-Chatbot-Cohere.git
cd Aristotle-Chatbot-Cohere
```

2. Install dependencies:

bash
```
pip install -r requirements.txt
```

3. Set up Cohere API Key:

- Obtain your API key from Cohere.
- Set it as an environment variable or directly in the code.

## Running the Application
1. Configure the API Key:

- Set your API key in the environment:
bash
```
export COHERE_API_KEY='your-api-key-here'
```

- Alternatively, you can set it directly in your Python script where you initialize the Cohere client.

2. Run the application:

bash
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


## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature-branch).
6. Open a pull request.


## License
This project is licensed under the Custom License. See the LICENSE file for details.

## Acknowledgements
- Thanks to Cohere for providing the language model.
- Inspired by the teachings and philosophy of Aristotle.