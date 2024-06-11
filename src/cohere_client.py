import cohere

class CohereClient:
    def __init__(self, api_key):
        self.client = cohere.Client(api_key)

    def generate_response(self, prompt, max_tokens=50):
        response = self.client.generate(
            model='command-r',
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response.generations[0].text
