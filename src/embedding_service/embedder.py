import os
import time
import random
from PIL import Image
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.embedding_service.document_processor import image_to_base64

class LangchainEmbedder:
    def __init__(self):
        """
        Initializes the LangChain embedder with EURI and ChatModel with Groq.
        """
        euri_api_key = os.environ.get("EURI_API_KEY")
        if not euri_api_key:
            raise ValueError("EURI_API_KEY environment variable not set.")
            
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

        self.embeddings = OpenAIEmbeddings(
            model="gemini-embedding-2",
            base_url="https://api.euron.one/api/v1/euri",
            api_key=euri_api_key
        )
        
        self.llm = ChatOpenAI(
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            max_tokens=1000,
            temperature=0
        )

    def embed_image(self, image: Image.Image, max_retries: int = 6, initial_backoff: float = 4.0) -> list[float]:
        """
        Embeds a single PIL Image with automatic exponential backoff retry for rate limits (429).
        Converts the image to base64 first.
        """
        print(f"Embedding image using EURI OpenAIEmbeddings...")
        # Convert PIL Image to base64 string
        base64_str = image_to_base64(image)
        
        # We pass the base64 string directly as the text query, assuming the EURI API accepts this
        image_text = f"data:image/jpeg;base64,{base64_str}"
        
        for attempt in range(max_retries):
            try:
                # Use embed_query since OpenAIEmbeddings expects strings
                return self.embeddings.embed_query(image_text)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(1.0, 3.0)
                    print(f"⏳ Rate limit (429) hit. Backing off for {sleep_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error embedding image: {e}")
                    raise e

        # Final attempt
        return self.embeddings.embed_query(image_text)

    def embed_text(self, text: str, max_retries: int = 5, initial_backoff: float = 2.0) -> list[float]:
        """
        Embeds a text query with automatic retry on rate limits (429).
        """
        print(f"Embedding text using EURI OpenAIEmbeddings...")
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"⏳ Rate limit (429) hit on text embedding. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error embedding text: {e}")
                    raise e

        return self.embeddings.embed_query(text)
