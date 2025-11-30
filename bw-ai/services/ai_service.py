from models.maternity import MaternityProfile
import json


def run_maternity_analysis(profile: MaternityProfile):
    """
    FastAPI 엔드포인트에서 호출할 실제 분석 함수
    지금은 디버깅용: 받은 데이터를 그대로 정리해서 응답
    """

    # JSON 문자열인 것들을 Python dict로 변환
    past_history = json.loads(profile.health_status.past_history_json)
    medicine = json.loads(profile.health_status.medicine_json)
    chronic = json.loads(profile.health_status.chronic_conditions_json)
    complications = json.loads(profile.health_status.pregnancy_complications_json)

    # 여기서 나중에 OpenAI API 호출 가능

    return {
        "user_name": profile.user.name,
        "user_email": profile.user.email,

        "pregnancy_summary": {
            "age": profile.pregnancy_info.age,
            "gestational_week": profile.pregnancy_info.gestational_week,
            "multiple": profile.pregnancy_info.is_multiple_pregnancy
        },

        "health_summary": {
            "past_history": past_history,
            "medicine": medicine,
            "chronic_conditions": chronic,
            "pregnancy_complications": complications
        },

        "status": "profile_received"
    }
