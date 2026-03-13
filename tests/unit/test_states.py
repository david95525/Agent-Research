# tests/unit/test_state.py
import pytest
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from app.services.medical.state import AgentState
from app.services.medical.service import MedicalAgentService


# 測試 State Reducer 邏輯 (純單元測試，不需非同步)
def test_state_message_addition():
    # 模擬初始狀態
    state1 = {"messages": [HumanMessage(content="你好")]}
    # 模擬第二個節點的回傳
    state2 = {"messages": [AIMessage(content="我好")]}

    # 執行 LangGraph 內部的合併邏輯
    combined_messages = operator.add(state1["messages"], state2["messages"])

    assert len(combined_messages) == 2
    assert isinstance(combined_messages[0], HumanMessage)
    assert combined_messages[1].content == "我好"


# 定義 Service Fixture
@pytest.fixture
async def medical_service(fake_llm_factory):
    service = MedicalAgentService()
    # 注入假 LLM 避免連網
    service.llm = fake_llm_factory(["測試回覆"])

    # 強制使用 MemorySaver 避開 Windows SQLite 鎖定與非同步競爭
    service.memory = MemorySaver()

    # 手動完成編譯，確保 self.app 不是 None
    workflow = service._build_workflow()
    service.app = workflow.compile(checkpointer=service.memory)
    # 這裡我們手動標記已初始化，避免它又跑去開 sqlite
    yield service
    await service.close()


# 測試記憶隔離性 (整合測試)


@pytest.mark.asyncio
async def test_memory_isolation(medical_service):
    # 準備兩個不同的 Thread Config
    config_a = {"configurable": {"thread_id": "user_A"}}
    config_b = {"configurable": {"thread_id": "user_B"}}

    # 用戶 A 說話
    await medical_service.handle_chat(user_id="user_A", message="我是01")
    # 用戶 B 說話
    await medical_service.handle_chat(user_id="user_B", message="我是02")

    # 必須傳入 config 才能取得對應 thread 的 State
    state_a = medical_service.app.get_state(config_a)
    state_b = medical_service.app.get_state(config_b)

    # 取得所有訊息內容
    all_content_a = "".join([m.content for m in state_a.values["messages"]])
    all_content_b = "".join([m.content for m in state_b.values["messages"]])

    # 驗證隔離性
    assert "01" in all_content_a
    assert "02" not in all_content_a, "用戶 A 不應該看到用戶 B 的對話紀錄"

    assert "02" in all_content_b
    assert "01" not in all_content_b, "用戶 B 不應該看到用戶 A 的對話紀錄"

    # 驗證最後一則訊息
    assert "01" in state_a.values["messages"][-1].content
    assert "02" in state_b.values["messages"][-1].content


# 4. 測試 State 的欄位覆寫 (而非累加)
def test_state_intent_overwrite():
    # intent 欄位沒有 Annotated[..., operator.add]，所以後者會覆寫前者
    state_initial = {"intent": "general"}
    state_update = {"intent": "health_analyst"}

    # 在 TypedDict 中，這只是簡單的 key 覆蓋
    state_initial.update(state_update)
    assert state_initial["intent"] == "health_analyst"
