import sqlite3
from typing import Sequence, Dict, List, Optional
from typing_extensions import Annotated, TypedDict

from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from src import config
from src.core import models
from src.core.tools import TOOLS_LIST
from src.utils.parsers import makeJSONToToolCall




# 모델, 토크나이저, 트리머 인스턴스화
llm = models.get_llm()
tokenizer = models.get_tokenizer()
trimmer = models.get_trimmer(tokenizer)
tools_node = ToolNode(TOOLS_LIST)

class State(TypedDict):
    variables: Dict[str, str]
    system_prompt: str
    history: Annotated[Sequence[BaseMessage], add_messages]
    messages: Optional[List[BaseMessage]]
    tools_result: Optional[List[ToolMessage]]
    query: HumanMessage
    final_answer: Optional[AIMessage]


def query_or_respond(state: State):
    filled_system_prompt = state["system_prompt"].format(**state["variables"])
    trimmed_messages = trimmer.invoke([SystemMessage(filled_system_prompt)] + state["history"] + [state["query"]])
    llm_with_tools = llm.bind_tools(TOOLS_LIST)
    response = llm_with_tools.invoke(trimmed_messages)
    content = response.content.strip()

    if content.startswith("<tool_call>"):
        tools_call = AIMessage(content="", tool_calls=makeJSONToToolCall(content))
        return {"messages": [tools_call], "final_answer": None}
    else:
        add_messages = [state["query"]] + [response]
        return {"history": add_messages, "messages": None, "final_answer": response}


def check_for_tools(state: State):
    return "tools" if state.get("messages") else END


def run_tools_and_pass_through_state(state: State):
    tools_result = tools_node.invoke(state["messages"])
    return {"tools_result": tools_result}


def generate(state: State):
    filled_system_prompt = state["system_prompt"].format(**state["variables"])
    conversation_messages = [
        message for message in state["history"]
        if message.type in ("human") or (message.type == "ai" and not message.tool_calls)
    ]
    trimmed_messages = trimmer.invoke([SystemMessage(filled_system_prompt)] + conversation_messages + state["tools_result"] + [state["query"]])
    response = llm.invoke(trimmed_messages)
    add_messages = [state["query"]] + state["messages"] + state["tools_result"] + [response]
    return {"history": add_messages, "final_answer": response}


def create_graph():
    """LangGraph 워크플로우를 생성하고 컴파일합니다."""
    workflow = StateGraph(state_schema=State)

    workflow.add_node("query_or_respond", query_or_respond)
    workflow.add_node("run_tools_and_pass_through_state", run_tools_and_pass_through_state)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "query_or_respond")
    workflow.add_conditional_edges("query_or_respond", check_for_tools, {END: END, "tools": "run_tools_and_pass_through_state"})
    workflow.add_edge("run_tools_and_pass_through_state", "generate")
    workflow.add_edge("generate", END)
    
    memory = SqliteSaver(conn=sqlite3.connect(config.SQLITE_DB_FILE, check_same_thread=False))
    
    return workflow.compile(checkpointer=memory)


# 그래프 인스턴스 생성
langgraph_app = create_graph()