from cohere_client import CohereClient

def train_model(api_key, dataset_url):
    cohere_client = CohereClient(api_key)
    response = cohere_client.client.train(
        dataset_url=dataset_url,
        model_type='generation',
        parameters={
            'max_training_steps': 1000,
            'learning_rate': 0.0001,
            'batch_size': 16
        }
    )
    return response

if __name__ == "__main__":
    api_key = 'your-api-key-here'  # Replace with your actual API key
    dataset_url = 'data/processed_aristotle_texts.json'  # URL or path to your dataset
    train_model(api_key, dataset_url)
