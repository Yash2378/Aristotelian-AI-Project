from src.chatbot import AristotleChatbot

def main():
    api_key = 'your-api-key-here'  # Replace with your actual Cohere API key
    chatbot = AristotleChatbot(api_key)
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit']:
            break
        response = chatbot.get_response(prompt)
        print(f"Aristotle: {response}")

if __name__ == "__main__":
    main()
