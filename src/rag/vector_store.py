import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src import config
from src.core.models import get_embedding_model




def get_vector_store():
    """
    ChromaDB 벡터 스토어를 로드하거나, 존재하지 않으면 새로 생성합니다.
    """
    embedding_model = get_embedding_model()
    
    if not os.path.exists(config.CHROMA_DB_PATH):
        print("📁 벡터 DB를 찾을 수 없어 새로 생성합니다...")
        loader = DirectoryLoader(
            config.DOCUMENTS_PATH,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        all_splits = text_splitter.split_documents(docs)

        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH,
            embedding_function=embedding_model
        )

        batch_size = 16 
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i : i + batch_size]
            vector_store.add_documents(batch)
            vector_store.persist()
        print("✅ 벡터 DB 생성 완료.")
    else:
        print("📁 기존 벡터 DB를 로드합니다.")
        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH,
            embedding_function=embedding_model
        )
    return vector_store

# 모듈 로드 시점에 벡터 스토어 인스턴스를 한 번만 생성
vector_store_instance = get_vector_store()