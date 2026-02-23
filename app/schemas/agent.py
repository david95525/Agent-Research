from pydantic import BaseModel, Field
from typing import List, Literal


class ChartParams(BaseModel):
    """
    健康數據視覺化參數模型。
    用於規範 LLM 產出繪圖工具所需的結構化參數。
    """

    chart_type: Literal["line", "bar", "scatter"] = Field(
        ...,
        description="圖表類型：'line' 適合趨勢, 'bar' 適合數值對比, 'scatter' 適合離散分佈",
    )
    columns: List[str] = Field(
        ..., description="要從資料庫提取的欄位 Key (例如: ['sys', 'dia'] 或 ['weight'])"
    )
    labels: List[str] = Field(
        ..., description="對應欄位的中文名稱標籤 (例如: ['收縮壓', '舒張壓'])"
    )
    unit: str = Field(..., description="Y 軸的數據單位 (例如: 'mmHg', 'kg', 'mg/dL')")
    title: str = Field(..., description="圖表最上方的標題內容")
