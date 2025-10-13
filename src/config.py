from pathlib import Path




# 기본 디렉토리 경로
BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# 모델 경로
EMBEDDING_MODEL_PATH = str(MODELS_DIR / "Qwen3-Embedding-0.6B")
LLM_MODEL_PATH = str(MODELS_DIR / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
TOKENIZER_MODEL_PATH = str(MODELS_DIR / "Qwen3-4B-Instruct-2507")

# 데이터 및 DB 경로
DOCUMENTS_PATH = str(DATA_DIR / "documents")
CHROMA_DB_PATH = str(DATA_DIR / "db" / "chroma_db")
SQLITE_DB_FILE = str(DATA_DIR / "db" / "chat_db" / "threads.sqlite")

# 모델 설정
MODEL_CONTEXT_SIZE = 8192
TRIMMER_MAX_TOKENS = 7000

# 시스템 프롬프트 및 변수
SYSTEM_PROMPT = """
You are a large language model, 'Qwen3'. You are currently conversing with a user via a chat app.
This means most of the time your lines should be a sentence or two unless the user's request requires reasoning or long-form outputs.
Never use emojis unless explicitly asked to. Never use emojis unless explicitly asked to.
If the tool call result exists, you write a response to the user based on the tool call result. The response language is {language}.

Tool Rules:
 - Do not describe or mention tools, you must call them directly.
 - You can use multiple tools if necessary.
 - Tool calls must start with <tool_call> and end with </tool_call>.
"""

VARIABLES = {
    "language": "Korean",
}