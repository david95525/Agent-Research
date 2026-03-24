# Agent-Research: AI Deep Research Platform (Finance & Medical)

基於 FastAPI + Gemini 2.5 Flash + LangGraph 的自主代理平台
本專案是一個集成了 DeepAgents (官方自主代理) 與 LangGraph (自定義思考鏈) 的先進 AI 系統。它不僅能處理醫療知識庫查詢與生理數據分析，還能針對金融標的進行即時深度研究。

# 🚀 核心進化與技術亮點

- **雙模式研究架構**：
  - **官方模式 (Official DeepAgents)**：利用 Gemini 的原生 Tool Calling 進行自主規劃與工具調用。
  - **手動模式 (Manual LangGraph)**：透過顯式定義「研究-風險分析-決策」節點，實現可控且透明的思考鏈（金融研究專用）。
- **醫療思考鏈 (Medical Workflow)**：採用 LangGraph 構建，包含意圖路由 (Router)、數據抓取 (Fetch)、健康分析 (Analyst) 與動態視覺化 (Visualizer) 節點。
- **動態技能注入 (Skill Injection)**：透過工具 `load_specialized_skill` 讀取 `skills/{skill_name}/SKILL.md`，動態賦予 Agent 不同領域（如：financial_expert 或 health_analyst）的專業人格與輸出規範。
- **混合持久化機制**：
  - **SQLite (AsyncSqliteSaver)**：負責 LangGraph 的狀態保存與對話記憶 (Thread-based Memory)。
  - **PostgreSQL (pgvector)**：專用於 RAG (檢索增強生成)，存儲 PDF 說明書的向量數據（由 `ingest_pdf.py` 處理）。

# 🛠️ 功能模組

### 1. 金融深度研究 (Financial Research)
- **即時數據**：自動校正股票代號（如 2330 -> 2330.TW）並抓取 yfinance 即時行情。
- **市場情緒**：透過 DuckDuckGo API 定位 tw-tzh 區域，獲取台股最新財經新聞。
- **深度分析**：手動模式下會歷經「數據採集 -> 專家解讀 -> 風險評估」的完整鏈條。

### 2. 醫療健康助理 (Medical AI)
- **設備知識檢索 (get_device_knowledge)**：目前透過模擬知識庫 (Mock KB) 針對 Microlife 醫療器材（血壓計、耳溫槍等）的錯誤代碼 (Err) 與故障排除流程進行匹配。
- **健康數據分析 (get_user_health_data)**：從遠端 API 獲取用戶歷史血壓與心率，並對照醫學標準提供趨勢分析。
- **緊急狀態識別**：若偵測到危險血壓值 (≥160/100)，系統會自動觸發 `[EMERGENCY]` 標記並給予急救指引。
- **動態視覺化 (plot_health_chart)**：自動將健康數據轉化為專業圖表（折線圖、長條圖、散佈圖）。
- **RAG 預研 (search_device_manual)**：已具備 PDF 向量化腳本 `ingest_pdf.py`，支援未來切換至完全動態的知識庫檢索。

# 快速開始

1. **環境準備**
   建立 `.env` 檔案並設定以下變數：
   ```env
   PORT=8000
   ENVIRONMENT=production
   GEMINI_API_KEY=你的Gemini金鑰
   # PostgreSQL 用於 pgvector (RAG 存儲)
   DATABASE_URL=postgresql+psycopg://postgres:密碼@db:5432/postgres
   # 遠端健康數據 API
   API_DOMAIN=https://api.example.com
   API_TOKEN=你的Token
   ```

2. **使用 Docker Compose 一鍵部署 (推薦)**
   ```bash
   # 建置鏡像並啟動服務
   docker compose up -d --build

   # (選配) 導入 PDF 資料至 PostgreSQL (pgvector)
   docker compose exec agent python ingest_pdf.py
   ```

3. **開發環境執行**
   ```bash
   # 安裝依賴
   uv sync

   # 啟動開發伺服器
   uv run fastapi dev main.py --host 0.0.0.0
   ```

# 📝 測試案例

| 測試場景 | 詢問範例 | 預期 Agent 行為 |
| :--- | :--- | :--- |
| **金融分析** | "分析股票 2330" | 自動補全代號，抓取股價與新聞，產出格式化投資快報。 |
| **設備故障** | "Err 3 是什麼意思？" | 識別為 device_expert，回答「袖帶充氣異常」並提供步驟。 |
| **數據趨勢** | "幫我分析最近的血壓" | 調用 API 抓取數據，給予趨勢總結。 |
| **動態繪圖** | "幫我畫出血壓圖表" | 解析日期範圍，調用繪圖工具產生 Base64 趨勢圖。 |
| **緊急攔截** | (數據異常時) | 自動觸發 `[EMERGENCY]` 邏輯，提示立即就醫。 |

# 專案結構

- `app/services/medical/nodes/`：LangGraph 核心節點（Router, Analyst, Expert...）。
- `app/services/tools/`：核心工具集（金融、醫療、系統工具）。
- `skills/`：存放專業領域的 Markdown 規範（人格設定）。
- `static/`：多功能前端介面（包含測試、Demo、研究區）。
- `ingest_pdf.py`：PDF 向量化存儲至 PostgreSQL 的腳本。

# 🛠️ 如何執行單元測試
本專案提供自動化測試，驗證 AI 節點邏輯（不產生 API 費用）：
- 指令行：輸入 `pytest tests/unit/test_nodes.py`。
- 檢查重點：`test_router_logic` (路由精準度)、`test_health_analyst_emergency` (緊急攔截機制)。
