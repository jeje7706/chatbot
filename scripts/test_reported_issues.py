"""사용자가 제보한 세 질문을 실제 OpenAI API로 검사합니다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import app


def main():
    docs, _, _ = app.load_documents()
    key = app.read_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다.")
    store = app.build_store(docs, key)
    questions = [
        "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?",
        "제주 2박 3일 출장에서 숙박비가 5만2천 원, 4만7천 원이면 얼마를 받을 수 있나요?",
        "민간기업 직원 출장비 기준도 알려주세요.",
    ]
    for index, question in enumerate(questions):
        response = app.answer_question(question, store, key)
        print("QUESTION:", question)
        print("ANSWER:", response["answer"])
        print("SOURCES:", [(source["source"], source["page"]) for source in response["sources"]])
        assert response["answer"] != app.UNKNOWN
        assert "검증하지 못" not in response["answer"]
        if index == 0:
            assert "세종" in response["answer"] and "대전" not in response["answer"]
            assert response["sources"]
        elif index == 1:
            compact = app.normalize(response["answer"])
            assert any(amount in compact for amount in ("99,000", "99000", "9만9천")), response
            assert response["sources"]
        else:
            assert "회사" in response["answer"] and "규정" in response["answer"]
            assert "공무" in response["answer"]
    print("PASS: 지명 보존, 동일 숙박비 사례 계산, 민간기업 적용 범위 안내")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAILED:", app.error_message(error))
        sys.exit(1)
