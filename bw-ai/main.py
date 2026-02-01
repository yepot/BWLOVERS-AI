from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, Optional, List, Union
import os
import uuid
from datetime import datetime, date
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import json
import logging
import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bw-ai")

app = FastAPI(title="BWLOVERS AI", version="1.0.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    log.error("[422] url=%s errors=%s body=%s", request.url, exc.errors(), body.decode("utf-8", "ignore"))
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def yyyymmdd_to_date(v: int) -> date:
    s = str(v)
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def any_to_date(v):
    if v is None:
        return None
    if isinstance(v, date):
        return v
    if isinstance(v, int):
        return yyyymmdd_to_date(v)
    if isinstance(v, list) and len(v) == 3:
        return date(int(v[0]), int(v[1]), int(v[2]))
    return v


class JobIn(BaseModel):
    jobId: Optional[int] = None
    jobName: Optional[str] = None
    riskLevel: Optional[int] = None


class PregnancyInfo(BaseModel):
    weightPre: Optional[int] = None
    weightCurrent: Optional[int] = None
    gestationalWeek: Optional[int] = None
    isFirstbirth: Optional[bool] = None
    isMultiplePregnancy: Optional[bool] = None
    miscarriageHistory: Optional[int] = 0
    expectedDate: Optional[Union[int, str, date, List[int]]] = None

    @field_validator("expectedDate", mode="before")
    def parse_expected_date(cls, v):
        return any_to_date(v)


class UserProfile(BaseModel):
    userId: Union[int, str]
    birthDate: Optional[Union[int, str, date, List[int]]] = None
    height: Optional[int] = None

    job: Optional[Union[str, JobIn]] = None
    jobName: Optional[str] = None

    weightPre: Optional[int] = None
    weightCurrent: Optional[int] = None
    gestationalWeek: Optional[int] = None
    isFirstbirth: Optional[bool] = None
    isMultiplePregnancy: Optional[bool] = None
    miscarriageHistory: Optional[int] = None
    expectedDate: Optional[Union[int, str, date, List[int]]] = None

    pregnancyInfo: Optional[PregnancyInfo] = None

    @field_validator("birthDate", mode="before")
    def parse_birth_date(cls, v):
        return any_to_date(v)

    @field_validator("job", mode="before")
    def normalize_job(cls, v):
        if isinstance(v, str):
            return JobIn(jobName=v)
        return v

    @model_validator(mode="after")
    def build_nested_pregnancy_info(self):
        if self.pregnancyInfo is not None:
            return self

        if any([
            self.weightPre is not None,
            self.weightCurrent is not None,
            self.gestationalWeek is not None,
            self.isFirstbirth is not None,
            self.isMultiplePregnancy is not None,
            self.miscarriageHistory is not None,
            self.expectedDate is not None,
        ]):
            self.pregnancyInfo = PregnancyInfo(
                weightPre=self.weightPre,
                weightCurrent=self.weightCurrent,
                gestationalWeek=self.gestationalWeek,
                isFirstbirth=self.isFirstbirth,
                isMultiplePregnancy=self.isMultiplePregnancy,
                miscarriageHistory=self.miscarriageHistory if self.miscarriageHistory is not None else 0,
                expectedDate=any_to_date(self.expectedDate),
            )
        return self

    def resolved_job_name(self) -> Optional[str]:
        if self.jobName:
            return self.jobName
        if isinstance(self.job, JobIn):
            return self.job.jobName
        if isinstance(self.job, str):
            return self.job
        return None


class PastDisease(BaseModel):
    statusId: Optional[int] = None
    pastDiseaseType: str
    pastCured: bool
    pastLastTreatedAt: Optional[str] = None


class ChronicDisease(BaseModel):
    statusId: Optional[int] = None
    chronicDiseaseType: str
    chronicOnMedication: bool


class HealthStatus(BaseModel):
    userId: Union[int, str]
    createdAt: Optional[Union[str, datetime]] = None
    pastDiseases: List[PastDisease] = Field(default_factory=list)
    chronicDiseases: List[ChronicDisease] = Field(default_factory=list)
    pregnancyComplications: List[str] = Field(default_factory=list)


class BackendRequest(BaseModel):
    user_profile: UserProfile
    health_status: HealthStatus


# =========================
# 응답 형태 스키마
# =========================

class EvidenceSourceOut(BaseModel):
    page_number: int
    text_snippet: str


class SpecialContractOut(BaseModel):
    contract_name: str
    contract_description: str
    contract_recommendation_reason: str
    key_features: List[str]
    page_number: int


class ItemOut(BaseModel):
    itemId: str

    insurance_company: str
    product_name: str
    is_long_term: bool
    monthly_cost: int

    insurance_recommendation_reason: Optional[str] = None
    special_contracts: Optional[List[SpecialContractOut]] = None
    evidence_sources: Optional[List[EvidenceSourceOut]] = None


class RecommendListResponseOut(BaseModel):
    resultId: str
    expiresInSec: int = 600
    items: List[ItemOut]


@app.get("/")
async def root():
    return {"message": "BWLOVERS AI 서버 실행 중", "version": "1.0.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/ai/recommend", response_model=RecommendListResponseOut)
async def recommend(request: BackendRequest):
    """
    백엔드에서 산모 정보 + 건강 상태를 받아 보험 추천 수행
    - 응답은 resultId/expiresInSec/items 형태로만 반환함 (백엔드 DTO 맞춤)
    """
    try:
        log.info("[AI] 요청 수신 RAW(JSON)\n%s", json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2))

        up = request.user_profile
        hs = request.health_status
        pi = up.pregnancyInfo

        # 추천 생성
        recommendations = generate_recommendations_db(request)

        items: List[Dict[str, Any]] = []
        for idx, rec in enumerate(recommendations, start=1):
            rec = dict(rec)
            rec["itemId"] = rec.get("itemId") or f"item-{idx}"
            items.append(rec)

        result_id = str(uuid.uuid4())
        ttl = 600

        # 콜백은 그대로 보내되, 응답에는 절대 섞지 않음
        backend_response = await send_to_backend(result_id, items)
        log.info("[AI] backend callback result=%s", backend_response)

        # 백엔드가 원하는 최종 응답
        return {
            "resultId": result_id,
            "expiresInSec": ttl,
            "items": items
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("[AI] 추천 처리 중 오류: %s", str(e))
        raise HTTPException(status_code=500, detail="AI 처리 중 내부 오류가 발생함")


async def send_to_backend(result_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    추천 결과를 백엔드 콜백으로 전송
    """
    url = f"{BACKEND_URL}/ai/callback/recommend"
    payload = {
        "resultId": result_id,
        "expiresInSec": 600,
        "items": items
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "BWLOVERS-AI/1.0"
    }

    access_token = os.getenv("BACKEND_ACCESS_TOKEN")
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            log.info("[AI] 백엔드로 결과 전송 시도... url=%s", url)
            resp = await client.post(url, json=payload, headers=headers)

        result: Dict[str, Any] = {"status_code": resp.status_code, "success": resp.status_code == 200}
        if resp.status_code != 200:
            result["error"] = resp.text
        return result

    except httpx.RequestError as e:
        return {"status_code": None, "success": False, "error": str(e)}


def generate_recommendations_db(request: BackendRequest) -> List[Dict[str, Any]]:
    """
    테스트용 예시 추천 데이터 반환
    """
    return [
        {
            "insurance_company": "교보라이프플래닛",
            "is_long_term": True,
            "product_name": "무배당 교보라플 어린이보험",
            "insurance_recommendation_reason": "일반보험 추천 이유 (사용자에 따라 달라지는 것)",
            "monthly_cost": 1000,
            "special_contracts": [
                {
                    "contract_name": "태아 검사비 지원 특약",
                    "contract_description": "특약 설명",
                    "contract_recommendation_reason": "특약 추천 이유 (사용자에 따라 달라지는 것)",
                    "key_features": ["다이렉트 보험으로 보험료 저렴", "태아 가입 시 선천성이상 및 저체중아 보장 특약 선택 가능"],
                    "page_number": 12
                }
            ],
            "evidence_sources": [
                {"page_number": 12, "text_snippet": "제 5조(보상하는 손해) ... 태아 검사비 지원에 관한 사항"}
            ]
        },
        {
            "insurance_company": "교보라이프플래닛2",
            "is_long_term": True,
            "product_name": "무배당 교보라플 어린이보험",
            "insurance_recommendation_reason": "일반보험 추천 이유 (사용자에 따라 달라지는 것)",
            "monthly_cost": 1000,
            "special_contracts": [
                {
                    "contract_name": "임신 중독증 진단비",
                    "contract_description": "특약 설명",
                    "contract_recommendation_reason": "특약 추천 이유 (사용자에 따라 달라지는 것)",
                    "key_features": ["다이렉트 보험으로 보험료 저렴", "태아 가입 시 선천성이상 및 저체중아 보장 특약 선택 가능"],
                    "page_number": 45
                }
            ],
            "evidence_sources": [
                {"page_number": 45, "text_snippet": "특약 별표 2 ... 임신 중독증 진단 확정 기준"}
            ]
        }
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
