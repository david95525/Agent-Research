# /tests/conftest.py
import pytest
import os
import logging
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def pytest_configure(config):
    """在測試開始前執行，禁用檔案 Log 避免 Windows 權限衝突"""
    # 讓 Logger 不要去開檔案，只輸出到 Console
    os.environ["DISABLE_LOG_FILE"] = "true"
    # 如果你的系統中有用快取，也可以在這裡指定測試用的路徑
    os.environ["DB_PATH"] = ":memory:"


@pytest.fixture(autouse=True)
def disable_logging_handlers():
    """自動移除所有 logger 的 FileHandler，防止 Windows 鎖定"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)


@pytest.fixture
def fake_llm_factory():
    """
    提供一個工廠函數，讓測試案例可以自定義 AI 的回覆內容。
    用法：fake_llm = fake_llm_factory(["意圖內容", "第二個回覆"])
    """

    def _create_fake_llm(responses: list):
        # 將字串列表轉換為 AIMessage 列表並轉成迭代器
        messages = iter([
            AIMessage(content=r) if isinstance(r, str) else r
            for r in responses
        ])
        return GenericFakeChatModel(messages=messages)

    return _create_fake_llm


@pytest.fixture
def mock_state():
    """提供標準的初始狀態模版"""
    return {
        "user_id": "test_user",
        "input_message": "",
        "messages": [],
        "intent": "general",
        "is_emergency": False,
        "context_data": None,
        "final_response": ""
    }
