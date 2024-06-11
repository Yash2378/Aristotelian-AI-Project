import unittest
from src.chatbot import AristotleChatbot

class TestAristotleChatbot(unittest.TestCase):
    def setUp(self):
        api_key = 'your-api-key-here'  # Replace with a valid API key for testing
        self.chatbot = AristotleChatbot(api_key)

    def test_get_response(self):
        prompt = "What is the essence of Aristotle's philosophy?"
        response = self.chatbot.get_response(prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()
