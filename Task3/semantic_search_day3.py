import os
import faiss
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from rich.console import Console
from langchain_openai import OpenAIEmbeddings
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)



load_dotenv()

console = Console()

# class VectorStore:
#     def __init__(self, embedding_model: str = "text-embedding-3-small"):
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         self.embedding_model = embedding_model
#         self.index = None
#         self.documents: List[Dict[str, Any]] = []
#         self.dimension = 1536




class RAGEngine:
    def __init__(self):
        load_dotenv()
        self.model_name = self._init_model_name()
        self.client = self._init_client()
        self.log_path = Path("logs")
        self.log_path.mkdir(exist_ok=True)


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

    def embed_documents(self) -> np.ndarray:
        all_vectors = []

        for txt_file in Path(__file__).parent.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as reader:
                txt_input = reader.read()

                chunks = [c.strip() for c in txt_input.split("\n\n") if c.strip()]

                for chunk in chunks:
                    response = self.client.embeddings.create(
                        input=chunk,
                        model="text-embedding-3-small"
                    )

                    print("chunk: " + chunk)

                    vector = np.array(response.data[0].embedding, dtype='float32')
                    all_vectors.append(vector)

        if len(all_vectors) == 0:
            raise ValueError("❌ Вектори не створено. Перевір вміст файлів у corpus/")

        vectors_matrix = np.array(all_vectors, dtype='float32')

        if len(vectors_matrix.shape) == 1:
            vectors_matrix = vectors_matrix.reshape(1, -1)


        faiss.normalize_L2(vectors_matrix)

        return vectors_matrix


def main() -> None:
    RAGEngine1 = RAGEngine()
    RAGEngine1.embed_documents()

if __name__ == "__main__":
    main()