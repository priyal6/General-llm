from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Check if API key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    exit(1)

print("API Key loaded successfully")

try:
    print("Creating ChatOpenAI model...")
    model = ChatOpenAI(model='gpt-4', temperature=0)
    
    print("Invoking model...")
    result = model.invoke("Write about attention is all you need")
    
    print("Result received:")
    print(result.content)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
