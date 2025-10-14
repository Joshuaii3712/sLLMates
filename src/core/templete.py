import json
from typing import Sequence
from langchain.schema import BaseMessage

from src.core.tools import tool_dict_bind




def convertMessageToText(messages : Sequence[BaseMessage], tool_bind = False) -> str:
    context = "" # <|begin_of_text|> 토큰은 llm 시작 시 자동 선언됨

    for message in messages:
        if message.type == 'system':
            context += f"<|start_header_id|>system<|end_header_id|>\n{message.content}"
            if tool_bind:
                context += tool_dict_bind()
                context += "<|eot_id|>"
            else:
                context += "<|eot_id|>"

        elif message.type == 'human':
            context += f"<|start_header_id|>user<|end_header_id|>\n\n{message.content}<|eot_id|>"

        elif message.type == 'ai':
            if message.tool_calls:
                context += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                for tool_call in message.tool_calls:
                    function_name = tool_call.get('name')
                    function_args = json.dumps(tool_call.get('args'))
                    context += "{\"name\": \"" + function_name + "\", \"parameters\": " + function_args + "}<|eot_id|>"
            else:
                context += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message.content}<|eot_id|>"

        elif message.type == 'tool':
            context += "<|start_header_id|>ipython<|end_header_id|>\n\n<|python_tag|>{\n    \"tool_call_id\": \"" + message.tool_call_id + "\"\n    \"output\": \"" + message.content + "\"\n}<|eot_id|>"

    context += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return context