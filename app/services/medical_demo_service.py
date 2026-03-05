# app/services/medical_demo_service.py
from app.services.base import BaseAgent
from app.utils.logger import setup_logger
from app.services.tools.medical_tools import get_user_health_data
from app.services.tools.system_tools import load_specialized_skill
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = setup_logger("MedicalDemoService")


class MedicalDemoService(BaseAgent):
    def __init__(self):
        super().__init__("MedicalDemoService")
        # 定義簡單的 LangChain 流程
        self.output_parser = StrOutputParser()

    async def run_demo_chat(self, user_id: str, message: str) -> str:
        """
        純 LangChain 實作：獲取數據 -> 組合 Prompt -> 調用 LLM
        """
        logger.info(f"[Demo Service] 處理用戶 {user_id} 的請求: {message}")

        try:
            # 直接調用工具獲取資料 (Mock 資料)
            # 因為 get_user_health_data 是 @tool，我們直接調用其原函數或 .invoke
            health_data = get_user_health_data.invoke({"user_id": user_id})

            # 載入專業技能設定 (skill.md)
            skill_config = load_specialized_skill.invoke(
                {"skill_name": "medical_expert"}
            )

            # 建立 ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "你是一個專業的醫護助理，請根據以下專業規範回答問題：\n\n{skill_config}",
                    ),
                    (
                        "human",
                        "【用戶健康數據】\n{health_data}\n\n【用戶問題】\n{user_message}",
                    ),
                ]
            )

            # 組合成 Chain (LCEL 語法)
            chain = prompt | self.llm | self.output_parser

            # 回傳內容
            response = await chain.ainvoke(
                {
                    "skill_config": skill_config,
                    "health_data": health_data,
                    "user_message": message,
                }
            )

            return response

        except Exception as e:
            logger.error(f"[Demo Service Error] 執行失敗: {str(e)}", exc_info=True)
            return "抱歉，分析您的健康數據時發生錯誤。請稍後再試。"
