from cohere_client import CohereClient

class AristotleChatbot:
    def __init__(self, api_key):
        self.cohere_client = CohereClient(api_key)

    def get_response(self, prompt):
        return self.cohere_client.generate_response(prompt)

if __name__ == "__main__":
    api_key = 'your-api-key-here'
    chatbot = AristotleChatbot(api_key)
    prompt = "Tell me about Aristotle's philosophy."
    response = chatbot.get_response(prompt)
    print(response)
