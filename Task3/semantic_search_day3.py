import os
import faiss
from pathlib import Path
from dotenv import load_dotenv
from numpy._typing import NDArray

# from langchain_community.docstore import InMemoryDocstore
# from langchain_community.vectorstores import FAISS
from Task1.chat_cli_day1 import ChatSession
from openai import OpenAI
from rich.console import Console
import numpy as np

load_dotenv()
MODEL_NAME = "gpt-4o"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"

load_dotenv()

console = Console()

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(1536)
        self.id_to_index = {}
        self.client = self._init_client()
        self.texts = []

    @staticmethod
    def _init_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAI(api_key=api_key)

    def get_texts(self):
        return self.texts

    def get_index(self):
        return self.index

    def embed(self, text: str):
        embedding = self.client.embeddings.create(input=text, model=EMBEDDINGS_MODEL_NAME)
        vector = np.array(embedding.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    def search(self, text: str, top: int = 3):
        vector = self.embed(text=text)
        distances, indices = self.index.search(vector, top)
        return indices[0]

    def get_by_id(self, id: str) -> NDArray[np.float32]:
        index = self.id_to_index[id]
        return self.index.reconstruct(index)

    def add_text(self, text: str, id: str ) -> None:
        vector = self.embed(text)
        index =  self.index.ntotal
        self.index.add(vector)
        self.id_to_index[id] = index
        self.texts.append(text)


class RAGEngine:
    def __init__(self, vector_storage: VectorStore):
        self.model_name = self._init_model_name()
        self.log_path = Path("logs")
        self.log_path.mkdir(exist_ok=True)
        self.vector_storage = vector_storage

    @staticmethod
    def _init_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        return OpenAI(api_key=api_key)

    @staticmethod
    def _init_model_name() -> str:
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME not found in .env")
        return model_name

    def embed_documents(self) -> None:
        for txt_file in Path(__file__).parent.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as reader:
                txt_input = reader.read()

                chunks = [c.strip() for c in txt_input.split("\n\n") if c.strip()]

                for i, chunk in enumerate(chunks):
                    self.vector_storage.add_text(id=str(i), text=chunk)


    def vector_search(self, query: str, top: int = 3):
        response = self.client.embeddings.create(input=query, model="text-embedding-3-small")
        query_vector = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top)
        return indices[0]


def main() -> None:
    session = ChatSession()
    engine = RAGEngine()
    try:
        while True:
            console.print("\n[bold cyan]You:[/bold cyan] ", end="")
            user_input = input()
            if user_input.lower() in ["exit", "quit"]:
                break

            indices = engine.vector_search(user_input)
            print("indices: " + f'{indices}')
            found_texts = [engine.texts[i] for i in indices]
            result = "\n".join(found_texts)
            prompt = f"""Context of knowledge database:
            {result}

            User question: {user_input}

            Answer in language of the question"""

            session.log_message(text=prompt, source="user")
            session.stream_response()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")

if __name__ == "__main__":
    main()