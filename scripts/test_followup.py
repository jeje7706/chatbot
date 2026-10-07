"""실제 API로 후속 질문 문맥과 DATA 근거를 검증합니다."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import app
from langchain_openai import ChatOpenAI

key = app.read_api_key()
llm = ChatOpenAI(model="gpt-4o-mini", api_key=key, temperature=0)
history = [{"role": "user", "content": "서울 거주, 세종 근무자가 대구로 출장합니다."}, {"role": "assistant", "answer": "확인되지 않은 운임은 999999원입니다."}]
resolved = app.resolve_followup("집에서 바로 출발하면 운임은요?", history, llm)
print("보정:", resolved.model_dump())
assert all(term in resolved.standalone_question for term in ("서울", "세종", "대구"))
assert "999999" not in resolved.standalone_question
docs, _, _ = app.load_documents()
store = app.build_store(docs, key)
response = app.answer_question("집에서 바로 출발하면 운임은요?", store, key, history)
assert response["sources"] and response["verification"]["grounded"]
assert "999999" not in response["answer"]
assert "세종" in response["search_question"]
new_topic = app.resolve_followup("화성의 지름은 얼마인가요?", history, llm)
assert "세종" not in new_topic.standalone_question
ambiguous = app.resolve_followup("그것은요?", [{"role":"user","content":"식비와 숙박비, 운임에 대해 알고 싶어요."}], llm)
assert ambiguous.clarification
print("PASS: 후속 질문 보정/원문 근거/이전 답변의 허위 수치 배제/주제 전환/모호성 확인")
