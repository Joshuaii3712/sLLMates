import gradio as gr
import uuid
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage

from src.core.agent import LangChainAgent
from src.config import SYSTEM_PROMPT, VARIABLES
from src.db.chat_metadata import (
    save_chat_metadata, update_chat_metadata, get_chat_list,
    delete_chat, rename_chat, get_chat_name, generate_chat_name_from_message
)
from src.db.bio_metadata import (
    get_all_bios, add_bio, update_bio, delete_bio, get_bio_by_id
)


agent = LangChainAgent()


# UI Helper Functions
def create_chatbot_response(message, history, thread_id):
    if not thread_id:
        history.append([message, "⚠️ '새 채팅'을 눌러 대화를 시작해주세요."])
        return history, ""

    if not message.strip():
        return history, ""

    if message.lower().strip() in ["exit", "q", "끝"]:
        history.append([message, "대화를 종료합니다."])
        return history, ""

    history.append([message, "💭 생각 중..."])
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        input_messages = HumanMessage(content=message)
        response_parts = []
        
        for step in agent.app.stream(
            {
                "variables": VARIABLES, 
                "system_prompt": SYSTEM_PROMPT,
                "messages": None, 
                "tools_result": None,
                "query": input_messages, 
                "final_answer": None
            },
            config=config,
            stream_mode="values",
        ):
            if "final_answer" in step and step["final_answer"]:
                if hasattr(step["final_answer"], 'content'):
                    response_parts.append(step["final_answer"].content)
                else:
                    response_parts.append(str(step["final_answer"]))
        
        full_response = "\n".join(response_parts) if response_parts else "응답을 생성하지 못했습니다."
        history[-1][1] = full_response
        
        # 메타데이터 업데이트
        update_chat_metadata(thread_id)
        
    except Exception as e:
        print(f"응답 생성 오류: {e}")
        history[-1][1] = f"❌ 오류가 발생했습니다: {str(e)}"
    
    return history, ""

#LangGraph state를 Gradio Chatbot 형식으로 변환
def format_history_for_chatbot(thread_data):
    if not thread_data or 'history' not in thread_data:
        return []
    
    history = thread_data.get('history', [])
    chatbot_history = []
    
    for msg in history:
        if isinstance(msg, HumanMessage):
            chatbot_history.append([msg.content, None])
        elif isinstance(msg, AIMessage):
            if chatbot_history and chatbot_history[-1][1] is None:
                chatbot_history[-1][1] = msg.content
            else:
                chatbot_history.append([None, msg.content])
    
    return chatbot_history

# Bio 관리 함수들
def load_bio_list():
    """Bio 목록을 테이블 형식으로 반환"""
    bios = get_all_bios()
    
    if not bios:
        return []
    
    # [ID, 텍스트, 중요도, 생성일, 수정일] 형식
    return [
        [
            bio["id"], 
            bio["document"][:50] + "..." if len(bio["document"]) > 50 else bio["document"],
            bio["importance"],
            bio["last_updated"][:16]
        ]
        for bio in bios
    ]

def get_bio_choices():
    """Dropdown용 Bio 선택지 반환"""
    bios = get_all_bios()
    if not bios:
        return []
    
    return [(f"[{bio['importance']}] {bio['document'][:40]}...", bio["id"]) for bio in bios]

def add_new_bio(text, importance):
    """새로운 Bio 추가"""
    if not text or not text.strip():
        return load_bio_list(), "⚠️ 텍스트를 입력하세요", ""
    
    try:
        importance_val = int(importance) if importance else 3
        if not (1 <= importance_val <= 10):
            return load_bio_list(), "⚠️ 중요도는 1-10 사이여야 합니다", ""
        
        bio_id = add_bio(text.strip(), importance_val)
        return load_bio_list(), f"✅ Bio 추가 완료 (ID: {bio_id[:8]}...)", ""
    except Exception as e:
        print(f"Bio 추가 오류: {e}")
        return load_bio_list(), f"❌ 추가 실패: {str(e)}", ""

def update_existing_bio(bio_id, new_text, new_importance):
    """기존 Bio 업데이트"""
    if not bio_id:
        return load_bio_list(), "⚠️ Bio를 선택하세요"
    
    if not new_text or not new_text.strip():
        return load_bio_list(), "⚠️ 텍스트를 입력하세요"
    
    try:
        importance_val = int(new_importance) if new_importance else None
        if importance_val and not (1 <= importance_val <= 10):
            return load_bio_list(), "⚠️ 중요도는 1-10 사이여야 합니다"
        
        update_bio(bio_id, text=new_text.strip(), importance=importance_val)
        return load_bio_list(), f"✅ Bio 업데이트 완료"
    except Exception as e:
        print(f"Bio 업데이트 오류: {e}")
        return load_bio_list(), f"❌ 업데이트 실패: {str(e)}"

def delete_selected_bio(bio_id):
    """선택된 Bio 삭제"""
    if not bio_id:
        return load_bio_list(), "⚠️ Bio를 선택하세요", None, "", ""
    
    try:
        delete_bio(bio_id)
        return load_bio_list(), "✅ Bio 삭제 완료", None, "", ""
    except Exception as e:
        print(f"Bio 삭제 오류: {e}")
        return load_bio_list(), f"❌ 삭제 실패: {str(e)}", bio_id, "", ""

def load_bio_for_edit(bio_id):
    """선택된 Bio의 정보를 편집 필드에 로드"""
    if not bio_id:
        return "", ""
    
    try:
        bio = get_bio_by_id(bio_id)
        if bio:
            return bio["document"], str(bio["importance"])
        return "", ""
    except Exception as e:
        print(f"Bio 로드 오류: {e}")
        return "", ""

# Gradio UI 
def create_simple_ui():
    css = """
    /* 전역 스타일 */
    .gradio-container {  
        min-height: 100vh;
        width: 80vh;
        margin: auto !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* 레이아웃 */
    .main-row {
        gap: 0 !important;
    }
    
    /* 사이드바 */
    #sidebar {
        border-right: 1px solid var(--border-color-primary);
        padding: 12px;
        min-height: calc(100vh - 40px);
        background: var(--background-fill-primary);
    }
    
    /* 헤더 */
    .app-title {
        font-size: 20px;
        font-weight: 1000;
        margin-bottom: 16px;
        padding-bottom: 12px;
        color: var(--body-text-color);
    }
    
    /* 레이블 스타일 */
    .field-label {
        font-size: 13px;
        font-weight: 600;
        color: var(--body-text-color-subdued);
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        display: block;
    }
    
    /* 버튼 - 모든 선 제거 */
    button {
        background: var(--background-fill-primary) !important;
        border: none !important;
        color: var(--body-text-color) !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
        font-size: 13px !important;
        box-shadow: none !important;
        margin: 2px 0 !important;
    }
    
    button:hover {
        background: var(--border-color-primary) !important;
    }
    
    button:active {
        background: var(--color-accent) !important;
        color: white !important;
        transform: scale(0.98);
    }
    
    /* 전송 버튼 */
    .primary-action {
        background: var(--color-accent) !important;
        color: white !important;
    }
    
    .primary-action:hover {
        opacity: 0.9 !important;
        background: var(--color-accent) !important;
    }
    
    /* 위험 버튼 */
    .danger-btn:active {
        background: #ef4444 !important;
        color: white !important;
    }
    
    /* 버튼 간격 */
    .button-row {
        gap: 0px;
        margin-bottom: 0px;
    }
    
    /* 입력 필드 - 선 제거, 배경색 통일 */
    input[type="text"], textarea, select {
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 10px !important;
        font-size: 13px !important;
        transition: all 0.15s ease !important;
        background: var(--background-fill-primary) !important;
        color: var(--body-text-color) !important;
        box-shadow: inset 0 0 0 1px var(--border-color-primary) !important;
    }
    
    input[type="text"]:focus, textarea:focus, select:focus {
        box-shadow: inset 0 0 0 2px var(--color-accent) !important;
        outline: none !important;
    }
    
    /* 드롭다운 */
    .dropdown-container {
        margin-bottom: 8px;
    }
    
    /* 채팅박스 */
    #chatbox {
        height: 80vh !important;
        border: none !important;
        border-radius: 8px !important;
        background: var(--background-fill-primary) !important;
        box-shadow: inset 0 0 0 1px var(--border-color-primary) !important;
        margin: 12px; /* 채팅창 주위 여백 */
    }
    
    /* Bio 테이블 */
    #bio-table {
        max-height: 450px !important;
        overflow-y: auto !important;
        border: none !important;
        border-radius: 6px !important;
        box-shadow: inset 0 0 0 1px var(--border-color-primary) !important;
    }
    
    #bio-table table {
        font-size: 13px;
    }
    
    #bio-table th {
        background: var(--background-fill-primary);
        font-weight: 600;
        padding: 8px;
        border-bottom: 1px solid var(--border-color-primary);
        position: sticky;
        top: 0;
        z-index: 10;
    }
    
    #bio-table td {
        padding: 8px;
        border-bottom: 1px solid var(--border-color-primary);
    }
    
    /* 슬라이더 */
    input[type="range"] {
        -webkit-appearance: none;
        height: 5px;
        border-radius: 3px;
        background: var(--background-fill-secondary);
        outline: none;
        margin: 10px 0;
    }
    
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--color-accent);
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    input[type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--color-accent);
        cursor: pointer;
        border: 2px solid white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    /* 탭 - 모든 선 제거 */
    /* .tab-nav 스타일은 Gradio의 기본 구조이므로 그대로 둡니다. */
    .tab-nav {
        border: none !important;
        background: var(--background-fill-primary) !important;
        gap: 4px !important;
        margin-bottom: 12px !important;
    }
    
    /* [최종 수정본]
      - 존재하지 않는 #sidebar ID 제거
      - 실제 HTML 구조인 .tab-wrapper > .tab-container 를 셀렉터에 추가
    */
    
    /* 상위 탭 (채팅, Bio) */
    .main-tabs > .tab-wrapper > .tab-container button[role="tab"] {
        font-size: 24px !important;     /* <-- 원하는 대로 크게 (예: 24px) */
        padding: 18px 22px !important; /* <-- 원하는 대로 크게 (예: 18px 22px) */
        border-radius: 6px !important;
        margin: 0 !important;
        background: var(--background-fill-primary) !important;
        border: none !important;
        color: var(--body-text-color-subdued) !important;
        font-weight: 500 !important;
    }
    
    .main-tabs > .tab-wrapper > .tab-container button[role="tab"]:hover {
        background: var(--background-fill-secondary) !important;
    }
    
    /* '.selected' 클래스는 HTML에 존재하므로 그대로 사용합니다. */
    .main-tabs > .tab-wrapper > .tab-container button[role="tab"].selected {
        color: var(--body-text-color) !important;
        background: var(--background-fill-secondary) !important;
        font-weight: 600 !important;
    }
    
    /* 하위 탭 (목록, 수정 등) */
    .sub-tabs > .tab-wrapper > .tab-container button[role="tab"] {
        font-size: 13px !important;   /* <-- 하위 탭 크기 (예: 13px) */
        padding: 8px 12px !important; /* <-- 하위 탭 여백 (예: 8px 12px) */
        border-radius: 6px !important;
        margin: 0 !important;
        background: var(--background-fill-primary) !important;
        border: none !important;
        color: var(--body-text-color-subdued) !important;
        font-weight: 500 !important;
    }
    
    .sub-tabs > .tab-wrapper > .tab-container button[role="tab"]:hover {
        background: var(--background-fill-secondary) !important;
    }
    
    .sub-tabs > .tab-wrapper > .tab-container button[role="tab"].selected {
        color: var(--body-text-color) !important;
        background: var(--background-fill-secondary) !important;
        font-weight: 600 !important;
    }
    
   /* [신규] '항상 열린' 채팅 목록 (Radio) 스타일 */
    #chat-list-radio {
        /* 컨테이너는 완벽하다고 하셨으니, 그대로 둡니다. */
        border: none !important; 
        border-radius: 6px !important; 
        padding: 0px !important; 
        max-height: 300px !important;
        overflow-y: auto !important;
        overflow: hidden; 
        background: var(--background-fill-primary) !important;
    }
    
    /* [수정] Radio의 '각 항목'을 감싸는 래퍼 (핵심 수정) */
    #chat-list-radio .radio-container {
        display: block !important;
        width: 100% !important;   /* [수정] 너비 100% 강제 */
        margin: 0 !important;     /* [수정] 요소 간 간격 0 */
        padding: 0 !important;    /* [수정] 요소 간 간격 0 */
    }
    
    /* [수정] Radio의 동그란 버튼 숨기기 */
    #chat-list-radio .radio-container label input[type="radio"] {
        display: none !important; /* [수정] 점 숨기기 */
        appearance: none !important;
        -webkit-appearance: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* [신규] '항상 열린' 채팅 목록 (Radio) 스타일 - HTML 기반 최종 수정 */
    
    /* 1. 컨테이너 (<fieldset id="chat-list-radio">) 
       'padded' 클래스가 있으므로 padding: 0 으로 강제합니다. */
    #chat-list-radio {
        border: none !important; 
        border-radius: 6px !important; 
        padding: 0px !important; /* [Fix] 요소 간 간격 0 (컨테이너 패딩 제거) */
        max-height: 300px !important;
        overflow-y: auto !important;
        overflow: hidden; /* 자식 둥글게 자르기 */
        background: var(--background-fill-primary) !important;
    }

    /* 2. 내부 래퍼 (<div class="wrap svelte-1kzox3m">) 
       이 래퍼에도 여백이 있을 수 있으므로 0으로 만듭니다. */
    #chat-list-radio .wrap {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 3. 각 항목 (<label class="svelte-1bx8sav">) */
    #chat-list-radio .wrap label {
        display: block !important;
        width: 100% !important;   /* [Fix] 너비 100% */
        box-sizing: border-box; /* 패딩이 너비를 밀어내지 않도록 */
        margin: 0 !important;     /* [Fix] 요소 간 간격 0 */
        padding: 8px 10px !important;
        border: none !important;
        border-radius: 0px !important;
        font-size: 13px !important;
        transition: all 0.1s ease !important;
        cursor: pointer !important;
    }

    /* 4. [Fix] 점 숨기기 (<input type="radio">)
       label 내부에 중첩되어 있으므로 이 선택자를 사용합니다. */
    #chat-list-radio .wrap label input[type="radio"] {
        display: none !important;
        appearance: none !important;
        -webkit-appearance: none !important;
    }

    /* 5. 텍스트 래퍼 (<span>) */
    #chat-list-radio .wrap label span {
        color: var(--body-text-color) !important;
    }
    
    /* 6. 선택된 항목 (HTML에서 label.selected 클래스 확인) */
    #chat-list-radio .wrap label.selected {
        background: var(--color-accent) !important;
    }
    
    #chat-list-radio .wrap label.selected span {
        color: white !important;
        font-weight: 600 !important;
    }

    /* 7. 호버 효과 (선택 안 된 항목) */
    #chat-list-radio .wrap label:not(.selected):hover {
        background: var(--background-fill-secondary) !important;
    }
    
    /* 스크롤바 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color-primary);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--body-text-color-subdued);
    }
    
    /* 간격 조정 */
    .gr-form {
        gap: 6px !important;
    }
    
    .gr-box {
        border: none !important;
        background: var(--background-fill-primary) !important;
    }
    
    /* 컴포넌트 간격 축소 */
    .gr-group {
        gap: 8px !important;
    }
    
    /* 패딩 축소 */
    .gr-panel {
        padding: 10px !important;
    }
    

   /* [신규] 'button-row' 안의 버튼 스타일 수정 (호버/클릭 포함) */
    
    /* 공통: 글씨 두껍게, 효과 전환 속도 */
    .button-row button {
        font-weight: 600 !important; 
        transition: all 0.15s ease !important; 
    }

    /* '새 채팅' 버튼: 좌측 정렬 */
    .button-row .new-chat-btn {
        justify-content: flex-start !important; 
        padding-left: 12px !important; 
    }

    /* '삭제' 버튼: 중앙 정렬 */
    .button-row .danger-btn {
        justify-content: flex-start !important;
        color: var(--body-text-color-subdued) !important;
    }


    /* 2. [요청] '새 채팅' 버튼 클릭(active) 시 효과 */
    .button-row .new-chat-btn:active {
        background: var(--color-accent) !important; /* 원본 CSS의 active 색상 */
        color: white !important;
        transform: scale(0.98); /* 살짝 눌리는 효과 */
    }
    
    /* 3. [요청] '삭제' 버튼 클릭(active) 시 효과 (원본 CSS의 danger-btn:active) */
    .button-row .danger-btn:active {
        background: #ef4444 !important; /* 원본 CSS의 위험 버튼 active 색상 */
        color: white !important;
        transform: scale(0.98); /* 살짝 눌리는 효과 */
    }
    
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
        thread_id_state = gr.State(None)
        bio_id_state = gr.State(None)
        
        with gr.Row(elem_classes="main-row"):
            # ==================== 사이드바 ====================
            with gr.Sidebar(elem_id="sidebar"):                
                with gr.Tabs(elem_classes="main-tabs"):
                    # ==================== 채팅 탭 ====================
                    with gr.Tab("채팅"):
                        with gr.Tabs(elem_classes="sub-tabs"):
                            # 새 채팅 + 목록 탭
                            with gr.Tab("목록"):
                                with gr.Row(elem_classes="button-row"):
                                    new_chat_btn = gr.Button("📝 새 채팅", size="sm", scale=2, elem_classes="new-chat-btn")
                                gr.HTML('<label class="field-label">채팅 목록</label>')
                                chat_dropdown = gr.Radio(
                                  choices=[],
                                  value=None,
                                  interactive=True,
                                  container=False,
                                  show_label=False,
                                  elem_id="chat-list-radio"
                                )
                                with gr.Row(elem_classes="button-row"):
                                    delete_btn = gr.Button("🗑️ 채팅방 삭제", size="sm", scale=1, elem_classes="danger-btn")
                                
                            
                            # 채팅방 수정 탭
                            with gr.Tab("수정"):
                                gr.HTML('<label class="field-label">현재 채팅방</label>')
                                current_chat_info = gr.Textbox(
                                    value="없음",
                                    interactive=False,
                                    container=False,
                                    show_label=False
                                )
                                
                                gr.HTML('<label class="field-label" style="margin-top: 12px;">새 이름</label>')
                                rename_input = gr.Textbox(
                                    placeholder="새 이름 입력",
                                    show_label=False,
                                    container=False
                                )
                                rename_btn = gr.Button("✏️ 이름 변경", size="sm")

                    # ==================== Bio 관리 탭 ====================
                    with gr.Tab("Bio"):
                        with gr.Tabs(elem_classes="sub-tabs"):
                            # Bio 목록 탭
                            with gr.Tab("목록"):
                                bio_table = gr.Dataframe(
                                    headers=["ID", "내용", "중요도", "수정일"],
                                    datatype=["str", "str", "number", "str"],
                                    interactive=False,
                                    elem_id="bio-table",
                                    wrap=True
                                )
                                refresh_bio_btn = gr.Button("🔄 새로고침", size="sm")
                                bio_status = gr.Markdown("")
                            
                            # Bio 추가 탭
                            with gr.Tab("추가"):
                                gr.HTML('<label class="field-label">텍스트</label>')
                                add_bio_text = gr.Textbox(
                                    placeholder="예: 사용자는 Python을 좋아합니다",
                                    lines=3,
                                    show_label=False,
                                    container=False
                                )
                                gr.HTML('<label class="field-label" style="margin-top: 8px;">중요도</label>')
                                add_bio_importance = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    show_label=False,
                                    container=False
                                )
                                add_bio_btn = gr.Button("➕ 추가", size="sm")
                            
                            # Bio 수정 탭
                            with gr.Tab("수정"):
                                gr.HTML('<label class="field-label">수정할 Bio</label>')
                                bio_dropdown = gr.Dropdown(
                                    choices=[],
                                    interactive=True,
                                    show_label=False,
                                    container=False
                                )
                                gr.HTML('<label class="field-label" style="margin-top: 8px;">텍스트</label>')
                                edit_bio_text = gr.Textbox(
                                    placeholder="수정할 내용",
                                    lines=3,
                                    show_label=False,
                                    container=False
                                )
                                gr.HTML('<label class="field-label" style="margin-top: 8px;">중요도</label>')
                                edit_bio_importance = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    show_label=False,
                                    container=False
                                )
                                with gr.Row(elem_classes="button-row"):
                                    update_bio_btn = gr.Button("💾 저장", size="sm", scale=2)
                                    delete_bio_btn = gr.Button("🗑️", size="sm", scale=1, elem_classes="danger-btn")
                
            # ==================== 메인 채팅 영역 ====================
            with gr.Column(scale=5):
                gr.HTML('<div class="app-title">💬 sLLMates 챗봇</div>')
                chatbot = gr.Chatbot(
                    show_label=False,
                    avatar_images=None, 
                    show_copy_button=True,
                    elem_id="chatbox",
                    height=600 # 이 높이는 CSS에서 #chatbox로 제어됩니다.
                )

                with gr.Row(elem_id="message-row"):
                    msg = gr.Textbox(
                        show_label=False, 
                        placeholder="메시지를 입력하세요...",
                        container=False, 
                        scale=10,
                        elem_id="message-input"
                    )
                    submit = gr.Button("전송", scale=2, variant="primary")

        # ==================== 채팅 이벤트 핸들러 (UI 스코프 내) ====================
        
        def start_new_chat():
            new_id = str(uuid.uuid4())
            chat_name = f"채팅 {datetime.now().strftime('%m/%d %H:%M')}"
            
            # 더미 헬퍼 호출
            save_chat_metadata(new_id, chat_name)
            updated_choices = get_chat_list() 
            
            return (
                new_id, # thread_id_state
                [[None, f"✨ 새 채팅이 시작되었습니다."]], # chatbot
                "", # msg
                gr.update(choices=updated_choices, value=new_id), # chat_dropdown
                chat_name # current_chat_info
            )
        
        def load_chat_history(selected_thread_id):
            if not selected_thread_id:
                return [], "없음", None
            
            try:
                config = {"configurable": {"thread_id": selected_thread_id}}
                state = agent.app.get_state(config)
                history = format_history_for_chatbot(state.values)
                
                chat_name = get_chat_name(selected_thread_id)
                info_text = f"📍 {chat_name}"
                
                return history, info_text, selected_thread_id
            except Exception as e:
                print(f"채팅 로드 오류: {e}")
                return [], f"❌ 로드 실패", None

        def refresh_list(current_id):
            choices = get_chat_list()
            value = current_id if current_id and any(c[1] == current_id for c in choices) else None
            return gr.update(choices=choices, value=value)

        def rename_current(thread_id, new_name):
            if not thread_id:
                return gr.update(), "", "선택된 채팅 없음"
            
            if not new_name.strip():
                return gr.update(), "", "이름을 입력하세요"
            
            if rename_chat(thread_id, new_name.strip()):
                return refresh_list(thread_id), "", new_name.strip()
            return gr.update(), "", "변경 실패"

        def delete_current(thread_id):
            if not thread_id:
                return gr.update(), [], "없음", None
            
            if delete_chat(thread_id):
                choices = get_chat_list()
                return gr.update(choices=choices, value=None), [], "없음", None
            return gr.update(), gr.update(), gr.update(), thread_id

        def send_message_with_update(message, history, thread_id):
            # 첫 메시지인 경우 자동 이름 생성
            if thread_id and len(history) == 1 and history[0][1] and "(이)가 시작되었습니다" in history[0][1]:
                auto_name = generate_chat_name_from_message(message)
                rename_chat(thread_id, auto_name)
            
            # 더미 응답 함수 호출
            return create_chatbot_response(message, history, thread_id)

        # ==================== 채팅 이벤트 바인딩 ====================
        new_chat_btn.click(
            start_new_chat,
            outputs=[thread_id_state, chatbot, msg, chat_dropdown, current_chat_info]
        )

        chat_dropdown.change(
            load_chat_history,
            inputs=[chat_dropdown],
            outputs=[chatbot, current_chat_info, thread_id_state]
        )

        rename_btn.click(
            rename_current,
            inputs=[thread_id_state, rename_input],
            outputs=[chat_dropdown, rename_input, current_chat_info]
        )

        delete_btn.click(
            delete_current,
            inputs=[thread_id_state],
            outputs=[chat_dropdown, chatbot, current_chat_info, thread_id_state]
        )

        submit.click(
            send_message_with_update,
            inputs=[msg, chatbot, thread_id_state],
            outputs=[chatbot, msg]
        ).then(
            refresh_list,
            inputs=[thread_id_state],
            outputs=[chat_dropdown]
        )
        
        msg.submit(
            send_message_with_update,
            inputs=[msg, chatbot, thread_id_state],
            outputs=[chatbot, msg]
        ).then(
            refresh_list,
            inputs=[thread_id_state],
            outputs=[chat_dropdown]
        )

        # ==================== Bio 이벤트 바인딩 ====================
        # (이 함수들은 외부의 더미 헬퍼 함수를 호출합니다)
        
        add_bio_btn.click(
            add_new_bio,
            inputs=[add_bio_text, add_bio_importance],
            outputs=[bio_table, bio_status, add_bio_text]
        ).then(
            lambda: gr.update(choices=get_bio_choices()),
            outputs=[bio_dropdown]
        )
        
        bio_dropdown.change(
            load_bio_for_edit,
            inputs=[bio_dropdown],
            outputs=[edit_bio_text, edit_bio_importance]
        ).then(
            lambda selected: selected, # 선택된 ID를 bio_id_state에 저장
            inputs=[bio_dropdown],
            outputs=[bio_id_state]
        )
        
        update_bio_btn.click(
            update_existing_bio,
            inputs=[bio_id_state, edit_bio_text, edit_bio_importance],
            outputs=[bio_table, bio_status]
        ).then(
            lambda: gr.update(choices=get_bio_choices()),
            outputs=[bio_dropdown]
        )
        
        delete_bio_btn.click(
            delete_selected_bio,
            inputs=[bio_id_state],
            outputs=[bio_table, bio_status, bio_dropdown, edit_bio_text, edit_bio_importance]
        ).then(
            lambda: (gr.update(choices=get_bio_choices()), None), # 드롭다운 갱신, bio_id_state 초기화
            outputs=[bio_dropdown, bio_id_state]
        )
        
        refresh_bio_btn.click(
            lambda: (load_bio_list(), gr.update(choices=get_bio_choices())),
            outputs=[bio_table, bio_dropdown]
        )

        # ==================== 초기 로드 ====================
        
        # 채팅 관련 초기 로드
        interface.load(
            lambda: gr.update(choices=get_chat_list()),
            outputs=[chat_dropdown]
        )
        interface.load(
            start_new_chat,
            outputs=[thread_id_state, chatbot, msg, chat_dropdown, current_chat_info]
        )
        
        # Bio 관련 초기 로드
        interface.load(
            lambda: (load_bio_list(), gr.update(choices=get_bio_choices())),
            outputs=[bio_table, bio_dropdown]
        )

    return interface
