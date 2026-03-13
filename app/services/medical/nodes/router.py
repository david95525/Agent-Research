from langchain_core.messages import AIMessage
from app.services.medical.state import AgentState
from app.utils.logger import setup_logger

logger = setup_logger("AgentService")


class RouterNode:

    def __init__(self, llm, manifest: str, valid_ids: list):
        self.llm = llm
        self.manifest = manifest
        self.valid_ids = valid_ids

    async def node_router(self, state: AgentState):
        """意圖路由：強化了隱私攔截與無意義輸入處理"""
        user_input = state["input_message"].strip().lower()

        #  安全性攔截：隱私與越權存取 (Hard-coded Guardrail)
        privacy_keywords = ["別人的", "上一個測試員", "其他人的", "他的血壓", "誰的紀錄"]
        if any(k in user_input for k in privacy_keywords):
            logger.warning(f"[Security] 偵測到潛在隱私存取請求: {user_input}")
            return {"intent": "general"}

        #  硬編碼攔截邏輯：繪圖確認
        last_ai_message = ""
        if state.get("messages"):
            for m in reversed(state["messages"]):
                if isinstance(m, AIMessage):
                    last_ai_message = m.content
                    break

        confirm_keywords = ["好", "要", "畫", "ok", "yes", "確認", "畫吧", "顯示", "可以"]
        is_asking_to_plot = "繪製趨勢分析圖表嗎" in last_ai_message
        if is_asking_to_plot and any(k in user_input
                                     for k in confirm_keywords):
            return {"intent": "visualizer"}

        #  LLM 判斷邏輯
        last_intent = state.get("intent", "general")
        prompt = (
            "你是一個專業的任務分發中心。請根據對話歷史判斷意圖：\n\n"
            f"【當前技能清單】\n{self.manifest}\n\n"
            f"【上一回合意圖】：{last_intent}\n"
            f"【用戶訊息】：{state['input_message']}\n\n"
            "【判定意圖類別（ID）說明】：\n"
            "1. 'device_expert': 設備硬體、故障碼、設定問題。\n"
            "2. 'health_query': 純數據查詢。特徵是沒有詢問『為什麼』或評估。例如：『查紀錄』、『列出數據』。\n"
            "3. 'health_analyst': 涉及評估與分析。例如：『我這樣正常嗎』、『幫我分析』、『最近血壓為什麼高』。\n"
            "4. 'visualizer': 要求畫圖或調整圖表。\n"
            "5. 'general': 閒聊、問候、詢問他人隱私、要求開藥、或是無法理解的亂碼。\n\n"
            "【決策準則】：\n"
            "- 如果用戶詢問非本人的數據或他人隱私，必須判斷為 'general'。\n"
            "- 如果用戶要求開藥、診斷或治療建議，必須判斷為 'general'。\n"
            "- 如果用戶輸入是無意義的亂碼（如 asdf），判斷為 'general'。\n"
            "- 如果只是純查詢資料，優先判斷為 'health_query'。\n"
            "【指令】僅回傳 ID，嚴禁解釋。")

        res = await self.llm.ainvoke(prompt)
        raw_intent = res.content.strip().lower()

        #  ID 匹配邏輯
        final_intent = "general"
        for vid in sorted(self.valid_ids, key=len, reverse=True):
            if vid.lower() in raw_intent:
                final_intent = vid
                break

        logger.info(f"[Router Decision] 識別意圖: {final_intent}")
        return {"intent": final_intent}
