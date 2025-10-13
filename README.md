# 💬 Qwen3 & LangGraph 기반 RAG 챗봇

이 프로젝트는 Qwen3-4B 모델을 기반으로 **LangGraph**를 사용하여 에이전트 워크플로우를 구성하고, **RAG (Retrieval-Augmented Generation)** 기술을 접목한 다목적 챗봇입니다. 사용자와의 대화 기록은 SQLite 데이터베이스에 영구적으로 저장되며, **Gradio**를 통해 직관적인 웹 UI를 제공합니다.

[Image of a modern chatbot interface]

## ✨ 주요 기능

* **RAG (Retrieval-Augmented Generation)**: `data/documents` 폴더의 텍스트 파일을 기반으로 LLM이 알지 못하는 정보에 대해 답변할 수 있습니다.
* **에이전트 워크플로우 (LangGraph)**: 단순한 질의응답을 넘어, LLM이 상황에 따라 RAG 검색 도구를 사용할지 스스로 판단하는 능동적인 워크플로우를 가집니다.
* **영구적인 대화 기록**: 모든 채팅 세션은 SQLite DB에 저장되어 언제든지 이전 대화를 불러오고 이어갈 수 있습니다.
* **편리한 채팅 관리**: 채팅방 생성, 삭제, 이름 변경 등 사용자 편의 기능을 UI에서 직접 제어할 수 있습니다.
* **모듈화된 구조**: 설정, UI, 데이터베이스, 핵심 로직이 분리되어 있어 기능 추가 및 유지보수가 용이합니다.

---

## 📂 디렉토리 구조

프로젝트는 기능별로 명확하게 분리된 구조를 가집니다.

```
rag_chatbot/
├── data/                 # 데이터(문서, DB) 저장소
├── models/               # LLM, 임베딩 모델 저장소
├── src/                  # 전체 소스 코드
│   ├── config.py         # 경로, 프롬프트 등 핵심 설정
│   ├── core/             # 모델 로딩, 에이전트, 도구 정의
│   ├── db/               # 채팅 DB 관리 로직
│   ├── rag/              # RAG 벡터 저장소 관리
│   ├── ui/               # Gradio UI 코드
│   └── utils/            # 보조 함수
└── main.py               # 애플리케이션 시작점
└── requirements.txt      # 필요 라이브러리 목록
```

---

## 🛠️ 설치 및 실행 방법

### 1. 사전 준비

* **Python 3.9 이상**이 설치되어 있어야 합니다.
* GPU(NVIDIA) 사용 시 **CUDA Toolkit**이 설치되어 있어야 합니다.

### 2. 프로젝트 클론 및 설정

```bash
# 1. 이 저장소를 클론합니다.
git clone https://your-repository-url/rag_chatbot.git
cd rag_chatbot

# 2. 가상 환경을 생성하고 활성화합니다. (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필요한 라이브러리를 설치합니다.
pip install -r requirements.txt
```

### 3. 모델 및 데이터 준비

다운로드한 모델과 RAG용 문서를 아래 경로에 맞게 배치합니다.

* **LLM 모델**: `/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`
* **토크나이저**: `/models/Qwen3-4B-Instruct-2507/`
* **임베딩 모델**: `/models/Qwen3-Embedding-0.6B/`
* **RAG 문서**: `/data/documents/` 폴더 내에 `.txt` 파일들

### 4. 챗봇 실행

```bash
python main.py
```
터미널에 출력되는 URL(예: `http://127.0.0.1:7860`)을 웹 브라우저에서 열어 챗봇을 사용하세요.

---

## ⚙️ 작동 원리 (아키텍처)

이 챗봇은 다음과 같은 흐름으로 사용자의 입력을 처리합니다.