from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import requests
import os
import json
import uuid
from datetime import datetime

app = FastAPI(
    title="BWLOVERS AI",
    description="산모 맞춤형 보험 추천 AI 서비스",
    version="1.0.0"
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

class PregnancyInfo(BaseModel):
    weight_pre: int           # BIGINT
    weight_current: int       # BIGINT
    is_firstbirth: bool       # BOOLEAN
    gestational_week: int     # BIGINT
    expected_date: str        # DATETIME (ISO format string)
    is_multiple_pregnancy: bool  # BOOLEAN
    miscarriage_history: int  # BIGINT

class UserProfile(BaseModel):
    user_id: int              # BIGINT
    birth_date: int           # BIGINT
    job_name: str             # VARCHAR
    height: int               # BIGINT
    pregnancy_info: PregnancyInfo

class PastDisease(BaseModel):
    status_id: int            # BIGINT
    past_disease_type: str    # ENUM (string으로 받음)
    past_cured: bool          # BOOLEAN
    past_last_treated_at: str # DATE (string format: "YYYY-MM")

class ChronicDisease(BaseModel):
    status_id: int            # BIGINT
    chronic_disease_type: str # ENUM (string으로 받음)
    chronic_on_medication: bool  # BOOLEAN

class PregnancyComplication(BaseModel):
    status_id: int            # BIGINT
    complication_type: str    # ENUM (string으로 받음)

class HealthStatus(BaseModel):
    user_id: int              # BIGINT
    created_at: str           # DATETIME
    past_diseases: Optional[List[PastDisease]] = []
    chronic_diseases: Optional[List[ChronicDisease]] = []
    pregnancy_complications: Optional[List[PregnancyComplication]] = []

class BackendRequest(BaseModel):
    user_profile: UserProfile
    health_status: HealthStatus

@app.get("/")
async def root():
    """AI 서버 상태 확인"""
    return {
        "message": "BWLOVERS AI 서버가 실행 중입니다.",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/ai/recommend")
async def recommend(request: BackendRequest):
    """
    백엔드에서 데이터베이스 타입에 맞는 산모 정보 받아서 AI 추천 수행
    """
    try:
        print("[AI] 백엔드에서 데이터베이스 타입 데이터 수신:")
        print(f"사용자 ID: {request.user_profile.user_id} (BIGINT)")
        print(f"생년월일: {request.user_profile.birth_date} (BIGINT)")
        print(f"직업: {request.user_profile.job_name} (VARCHAR)")
        print(f"키: {request.user_profile.height}cm (BIGINT)")
        print(f"임신 주차: {request.user_profile.pregnancy_info.gestational_week} (BIGINT)")
        print(f"초산 여부: {request.user_profile.pregnancy_info.is_firstbirth} (BOOLEAN)")
        print(f"다태아 여부: {request.user_profile.pregnancy_info.is_multiple_pregnancy} (BOOLEAN)")
        print(f"유산 횟수: {request.user_profile.pregnancy_info.miscarriage_history} (BIGINT)")
        print(f"과거 질환: {len(request.health_status.past_diseases)}개")
        print(f"만성 질환: {len(request.health_status.chronic_diseases)}개")
        print(f"임신 합병증: {len(request.health_status.pregnancy_complications)}개")
        
        # AI 추천 로직 (데이터베이스 타입에 맞게)
        recommendations = generate_recommendations_db(request)
        
        # 결과 ID 생성
        result_id = str(uuid.uuid4())
        
        # 백엔드로 결과 전송 
        backend_response = await send_to_backend(result_id, recommendations)
        
        return {
            "success": True,
            "resultId": result_id,
            "processed_data": {
                "userId": request.user_profile.user_id,
                "gestationalWeek": request.user_profile.pregnancy_info.gestational_week,
                "complications_count": len(request.health_status.pregnancy_complications)
            },
            "recommendations": recommendations,
            "backend_response": backend_response
        }
        
    except Exception as e:
        print(f"[AI] 추천 처리 중 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "timestamp": datetime.now().isoformat(),
                "status": 500,
                "error": "INTERNAL_SERVER_ERROR",
                "message": f"AI 처리 중 내부 오류가 발생했습니다: {str(e)}",
                "path": "/ai/recommend"
            }
        )

async def send_to_backend(result_id: str, recommendations: list) -> Dict[str, Any]:
    """추천 결과를 백엔드로 전송"""
    try:
        url = f"{BACKEND_URL}/ai/callback/recommend"
        payload = {
            "resultId": result_id,
            "expiresInSec": 600,
            "items": recommendations
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "BWLOVERS-AI/1.0"
        }
        
        # 인증 토큰이 있으면 추가
        access_token = os.getenv("BACKEND_ACCESS_TOKEN")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        
        print(f"[AI] 백엔드로 결과 전송 시도...")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        result = {
            "status_code": response.status_code,
            "success": response.status_code == 200
        }
        
        if response.status_code == 200:
            print(f"[AI] 백엔드 전송 성공")
        else:
            print(f"[AI] 백엔드 전송 실패: {response.status_code}")
            result["error"] = response.text
            
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"[AI] 백엔드 연결 실패: {str(e)}")
        return {
            "status_code": None,
            "success": False,
            "error": str(e)
        }

def generate_recommendations_db(request: BackendRequest) -> List[Dict[str, Any]]:
    """
    데이터베이스 타입을 고려한 보험 추천 생성
    """
    recommendations = []
    pregnancy_info = request.user_profile.pregnancy_info
    health_status = request.health_status
    
    gestational_week = pregnancy_info.gestational_week
    is_firstbirth = pregnancy_info.is_firstbirth
    is_multiple = pregnancy_info.is_multiple_pregnancy
    miscarriage_count = pregnancy_info.miscarriage_history
    
    # 유산 경력이 있는 경우 고려
    has_miscarriage_history = miscarriage_count > 0
    
    # 기본 추천 로직
    if gestational_week < 20:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "교보라이프플래닛",
            "product_name": "무배당 교보라플 어린이보험",
            "is_long_term": True,
            "monthly_cost": 1000,
            "summary_reason": f"임신 {gestational_week}주차 초기 단계에 적합한 기본 보장"
        })
    elif gestational_week < 30:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "삼성화재",
            "product_name": "무배당 삼성화재 다이렉트 임산부ㆍ아기보험",
            "is_long_term": True,
            "monthly_cost": 1200,
            "summary_reason": f"임신 {gestational_week}주차 임산부 특화 보장"
        })
    else:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "KB손해보험",
            "product_name": "KB 다이렉트 자녀보험",
            "is_long_term": True,
            "monthly_cost": 1100,
            "summary_reason": "출산 준비 및 산후 보장"
        })
    
    # 다태아 임신인 경우 추가 추천
    if is_multiple:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "현대해상",
            "product_name": "굿앤굿어린이종합보험Q",
            "is_long_term": True,
            "monthly_cost": 1400,
            "summary_reason": "다태아 임신 특화 보장 및 추가 건강 혜택"
        })
    
    # 유산 경력이 있는 경우 추가 추천
    if has_miscarriage_history:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "메리츠화재",
            "product_name": "메리츠 다이렉트 어린이보험",
            "is_long_term": True,
            "monthly_cost": 1150,
            "summary_reason": f"유산 경력({miscarriage_count}회) 고려한 추가 보장"
        })
    
    # 합병증이 있는 경우 추가 추천
    if health_status.pregnancy_complications and len(health_status.pregnancy_complications) > 0:
        recommendations.append({
            "itemId": f"rec-{uuid.uuid4().hex[:8]}",
            "insurance_company": "한화손해보험",
            "product_name": "한화손해보험 어린이보험",
            "is_long_term": True,
            "monthly_cost": 1300,
            "summary_reason": "고위험 임신 특화 보장 및 추가 건강 혜택"
        })
    
    return recommendations[:3]  # 최대 3개 추천

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )