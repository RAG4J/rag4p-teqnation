from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    secret_key = key_loader.get_openai_api_key()

    if secret_key is None:
        print("No OpenAI API key found. Please continue the README to create '.env' file with the key.")
    else:
        print(f"You are ready to go, have fun during the workshop!")