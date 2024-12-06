from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        
    def add_documents(self, documents):
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        
    def get_relevant_context(self, query, k=3):
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs]) 