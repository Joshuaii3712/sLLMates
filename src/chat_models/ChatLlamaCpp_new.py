from typing import List
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import BaseMessage


class ChatLlamaCpp_new(ChatLlamaCpp):
    """ChatLlamaCpp를 상속하여 ChatLlamaCpp에 존재하던 오류를 수정한 새로운 클래스"""
    
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        if not self.client:
            raise ValueError(
                "Llama.cpp client (self.client) is not initialized. "
                "Ensure the model_path is correct and the model is loaded."
            )
        
        total_tokens = 0
        for message in messages:
            content_str = ""

            if isinstance(message.content, str):
                content_str = message.content
            elif isinstance(message.content, list):
                for part in message.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_str += part.get("text", "")

            if content_str:
                try:
                    message_bytes = content_str.encode("utf-8")
                    # llama-cpp-python의 네이티브 tokenize 메서드 사용
                    tokens = self.client.tokenize(message_bytes)
                    total_tokens += len(tokens)
                except Exception as e:
                    print(f"Warning: Could not tokenize message content: {e}")
                    total_tokens += len(content_str) // 4
        
        return total_tokens