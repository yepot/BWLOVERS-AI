from fastapi import FastAPI
from models.maternity import MaternityProfile

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI AI server is running"}

@app.post("/ai/recommend")
async def recommend(profile: MaternityProfile):
    """
    Spring에서 보내준 산모 데이터를 그대로 다시 돌려주기 (디버깅용)
    """
    print("[FastAPI] 받은 데이터:")
    print(profile.dict())
    
    data = profile.model_dump()

    # --- 1) 최상위 구조 분리 ---
    user = data["user"]
    preg = data["pregnancyInfo"]
    health = data["healthStatus"]

    # --- 2) User 정보 분해 ---
    name = user["name"]
    email = user["email"]
    user_id = user["user_id"]

    # --- 3) 임신 기본 정보(PregnancyInfo) 분해 ---
    age = preg["age"]
    height = preg["height"]
    weight_pre = preg["weight_pre"]
    weight_current = preg["weight_current"]
    is_firstbirth = preg["is_firstbirth"]
    gestational_week = preg["gestational_week"]
    expected_date = preg["expected_date"]
    is_multiple_pregnancy = preg["is_multiple_pregnancy"]
    miscarriage_history = preg["miscarriage_history"]

    # --- 4) 건강 정보(HealthStatus) 분해 ---
    past_history = health["past_history_json"]
    medicine = health["medicine_json"]
    current_condition = health["current_condition"]
    chronic_conditions = health["chronic_conditions_json"]
    pregnancy_complications = health["pregnancy_complications_json"]

    prompt = f"""
        당신은 태아 보험 전문가입니다.
        아래 산모 정보를 기반으로 맞춤 보험 특약을 추천하세요.

        [산모 기본 정보]
        - 이름: {name}
        - 나이: {age}
        - 키: {height}cm
        - 임신 전 체중: {weight_pre}kg
        - 현재 체중: {weight_current}kg
        - 초산 여부: {"초산" if is_firstbirth else "경산"}
        - 다태아 여부: {"단태아" if not is_multiple_pregnancy else "다태아"}
        - 임신 주차: {gestational_week}주차
        - 출산 예정일: {expected_date}
        - 유산 경험: {miscarriage_history}회

        [병력]
        {past_history}

        [복용 중인 약물]
        {medicine}

        [만성 질환]
        {chronic_conditions}

        [임신 중 위험요인]
        {pregnancy_complications}

        [현재 증상]
        {current_condition}

        위 정보를 기반으로 맞춤 보험 특약 10개를 JSON 형식으로 추천하세요.
        """

    
    
    return {
        "success": True,
        "received_profile": profile.dict()
    }
