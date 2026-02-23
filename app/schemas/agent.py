from pydantic import BaseModel, Field
from typing import Literal, List  # 建議還是匯入 List


class ChartParams(BaseModel):
    """
    健康數據視覺化參數模型。
    用於規範 LLM 產出繪圖工具所需的結構化參數。
    """

    chart_type: Literal["line", "bar", "scatter"] = Field(
        ...,
        description="圖表類型：'line' 適合長短期趨勢, 'bar' 適合不同日期間的數值對比, 'scatter' 適合觀察數據離散程度",
    )
    # 這裡改用 List[str] 或保持 list[str] 都可以，但務必確保 typing 匯入正確
    columns: List[str] = Field(
        ...,
        min_items=1,
        description="要繪製的數據欄位 Key (例如: ['sys', 'dia'] 或 ['weight'])",
    )
    labels: List[str] = Field(
        ...,
        min_items=1,
        description="對應欄位的中文標題，必須與 columns 數量一致 (例如: ['收縮壓', '舒張壓'])",
    )
    unit: str = Field(..., description="Y 軸的數據單位 (例如: 'mmHg', 'kg', 'mg/dL')")
    title: str = Field(..., description="圖表標題，應包含數據種類與用戶特徵")
