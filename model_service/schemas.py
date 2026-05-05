from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# Request — raw bank call record (19 original features, no duration)
# Feature engineering (pdays → flags) happens inside the service, not here.
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int = Field(..., description="999 means never contacted before")
    previous: int
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class PredictResponse(BaseModel):
    prediction: int          # 0 or 1
    probability: float       # raw model score for class 1
    threshold: float         # operating threshold used


class DriftFeatureDetail(BaseModel):
    feature: str
    statistic: float         # PSI value or chi² statistic
    p_value: float | None    # None for PSI (no p-value)
    drifted: bool


class DriftReport(BaseModel):
    severity: Literal["none", "warning", "critical"]
    window_size: int
    numeric_drift: list[DriftFeatureDetail]
    categorical_drift: list[DriftFeatureDetail]
    output_drift: float      # PSI on prediction probabilities


class PromoteResponse(BaseModel):
    version: int
    stage: str
    message: str
