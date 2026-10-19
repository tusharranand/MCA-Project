from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    Customer_ID: str
    Tenure_Months: Optional[float] = None
    Contract_Type: Optional[str] = None
    Plan_Type: Optional[str] = None
    Monthly_Spend: Optional[float] = None
    Total_Spend: Optional[float] = None
    Support_Tickets: Optional[int] = None
    Satisfaction_Score: Optional[int] = None
    Last_Interaction_Days: Optional[int] = None
    Payment_Method: Optional[str] = None
    Region: Optional[str] = None
    Age: Optional[int] = None
    Churned: Optional[int] = None


class PredictionRequest(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    features: Dict[str, Any] = Field(..., description="Feature values for the customer")


class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    risk_band: str
    model_version: str
    explanation: List[str]
    validation_errors: List[str] = []
