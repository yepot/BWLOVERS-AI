from pydantic import BaseModel
from typing import Optional

class PregnancyInfo(BaseModel):
    age: int
    height: int
    weight_pre: int
    weight_current: int
    is_firstbirth: bool
    gestational_week: int
    expected_date: str
    is_multiple_pregnancy: bool
    miscarriage_history: Optional[int]

class HealthStatus(BaseModel):
    past_history_json: str
    medicine_json: str
    current_condition: str
    chronic_conditions_json: str
    pregnancy_complications_json: str

class User(BaseModel):
    user_id: int
    name: str
    email: str

class MaternityProfile(BaseModel):
    user: User
    pregnancyInfo: PregnancyInfo      
    healthStatus: HealthStatus        
