import os

from src.ui.app import create_simple_ui
from src.db.metadata import init_chat_metadata_db




os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 




def initialize():
    init_chat_metadata_db()


def launch_app():
    interface = create_simple_ui()
    interface.launch(share=True, inbrowser=True, show_error=True)




if __name__ == "__main__":

    print("초기 설정 시작...")
    initialize()

    print("RAG 챗봇 UI 시작...")
    launch_app()