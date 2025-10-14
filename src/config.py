from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate




# 기본 디렉토리 경로
BASE_DIR = Path(__file__).parent.parent.resolve()

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# 모델 경로
LLM_MODEL_PATH = str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "Qwen3-Embedding-0.6B")
TOKENIZER_MODEL_PATH = str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct_tokenizer")

# 데이터 및 DB 경로
DOCUMENTS_PATH = str(DATA_DIR / "documents")
CHROMA_DB_PATH = str(DATA_DIR / "db" / "chroma_db")
SQLITE_DB_FILE = str(DATA_DIR / "db" / "chat_db" / "threads.sqlite")


# 모델 설정
class LLMConfig:
    """
    LLM 모델 관련 설정
    """
    model_path = LLM_MODEL_PATH
    n_gpu_layers = -1
    main_gpu = 1 # gpu 0, 1 중 1을 메인 gpu로 선택
    tensor_split = [1, 1] # 50:50으로 두 gpu의 vram에 모델 레이어를 나누어 올림 
    use_mmap = True
    use_mlock = False
    n_batch = 512
    n_ctx = 4096
    max_tokens = -1
    f16_kv = True
    verbose = False
    temperature = 1.0
    top_p = 1
    top_k = 20
    min_p = 0
    stop = ["<|eot_id|>", "<|end_of_text|>"]


class EmbeddingConfig:
    """
    Embedding 모델 관련 설정
    """
    model_name = EMBEDDING_MODEL_PATH
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}


class TrimmerConfig:
    """
    Trimmer 관련 설정
    """
    max_tokens = 3000
    strategy = "last"
    include_system = True
    allow_partial = False
    start_on = "human"


class RAGConfig:
    """
    RAG 파이프라인 관련 설정
    """
    chunk_size = 200
    chunk_overlap = 50
    batch_size = 16
    retrieval_k = 3


# 시스템 프롬프트 및 변수
VARIABLES = {
    "language": "Korean",
}

SYSTEM_PROMPT = """
You are Llama3.1, a large language model trained by Meta, based on the Llama architecture. You are chatting with the user via the Chating app. This means most of the time your lines should be a sentence or two unless the user's request requires reasoning or long-form outputs. Never use emojis unless explicitly asked to. When you receive a tool call response, use the output to format an answer to the orginal user question. The response language is {language}.
"""