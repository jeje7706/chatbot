"""로컬 검증 및 --live 옵션으로 실제 OpenAI 연결을 검사합니다."""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch

# scripts 폴더에서 실행해도 최상단의 app.py를 찾습니다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import app
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.app_test import TMP_DIR


class TestContextFilter(logging.Filter):
    def filter(self, record):
        # 테스트 스레드의 세션 누락 안내만 제외하며 deprecated 경고는 검사합니다.
        return "Thread 'MainThread': missing ScriptRunContext!" not in record.getMessage()


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(TestContextFilter())


class TestEmbeddings(Embeddings):
    """외부 API 없이 벡터 저장소의 실제 검색 경로를 테스트합니다."""

    def embed_documents(self, texts):
        return [[float("출장" in text), float("식비" in text), 1.0] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def local_checks():
    assert sys.version_info[:2] == (3, 11)
    docs, files, notices = app.load_documents()
    assert len(files) == len(app.data_signature())
    assert all(doc.metadata["page"] >= 1 for doc in docs)
    assert not notices, notices
    chunks = app.split_documents(docs)
    print(f"PASS: PDF {len(files)}개, 텍스트 {len(docs)}페이지, {len(chunks)}개 조각")

    document = Document(page_content="출장 식비는 하루 25,000원이다.", metadata={"source": "test.pdf", "page": 1})
    valid = app.GroundedAnswer(supported=True, claims=[app.Claim(
        text="식비는 하루 25,000원입니다.",
        evidence=[app.Evidence(document_id=1, quote=document.page_content)],
    )])
    assert app.validate_answer(valid, [document])["sources"][0]["page"] == 1
    invalid = valid.model_copy(deep=True)
    invalid.claims[0].evidence[0].quote = "문서에 존재하지 않는 문장입니다."
    assert app.validate_answer(invalid, [document])["answer"] == app.UNKNOWN
    invalid.claims[0].evidence[0].document_id = 99
    assert app.validate_answer(invalid, [document])["sources"] == []
    assert app.validate_answer(app.GroundedAnswer(supported=False, claims=[]), [document])["answer"] == app.UNKNOWN
    guarded = app.require_calculation_conditions("2박3일 출장 여비 계산", valid.model_copy(deep=True))
    assert guarded.missing_information
    assert "25,000" not in app.validate_answer(guarded, [document])["answer"]

    store = InMemoryVectorStore(TestEmbeddings())
    store.add_documents([document])
    # 실제 Runnable 조합, 검색, 근거 검증을 통과시키되 LLM 응답만 대체합니다.
    with patch.object(app.ChatOpenAI, "with_structured_output", side_effect=lambda schema, **_: RunnableLambda(lambda _: app.SearchPlan(queries=["출장 식비"]) if schema is app.SearchPlan else app.AnswerAudit(grounded=True, matches_question=True, issues=[]) if schema is app.AnswerAudit else app.ReferencedAnswer(supported=True, claims=[app.ReferencedClaim(text="식비는 하루 25,000원입니다.", evidence_ids=[1])], missing_information=[]))):
        assert app.answer_question("출장 식비?", store, "test-placeholder")["sources"]
    print("PASS: 실제 벡터 검색/Runnable 실행, 인용 검증, 없는 근거 거부")
    with patch.object(app.ChatOpenAI, "with_structured_output", side_effect=lambda schema, **_: RunnableLambda(lambda _: app.SearchPlan(queries=["출장 식비"]) if schema is app.SearchPlan else app.AnswerAudit(grounded=False, matches_question=True, issues=["검사용 근거 불일치"]) if schema is app.AnswerAudit else app.ReferencedAnswer(supported=True, claims=[app.ReferencedClaim(text="틀린 확정 답변", evidence_ids=[1])], missing_information=[]))):
        checked = app.answer_question("출장 식비?", store, "test-placeholder")
        assert "보류" in checked["answer"]
        assert not checked["verification"]["grounded"]
        assert checked["sources"]
    print("PASS: 자동 답변 검토 실패 시 확정 답변 차단")

    history_path = Path(TMP_DIR.name) / "history.sqlite3"
    with patch.object(app, "HISTORY_DB", history_path), patch.object(app, "read_api_key", return_value=""):
        screen = AppTest.from_string("import app\napp.main()", default_timeout=60).run()
        assert not screen.exception, screen.exception
        assert screen.chat_input[0].disabled
        assert screen.warning
    with patch.object(app, "HISTORY_DB", history_path), patch.object(app, "read_api_key", return_value="test-placeholder"), patch.object(app, "build_store", return_value=store), patch.object(app, "answer_question", return_value=app.validate_answer(valid, [document])):
        screen = AppTest.from_string("import app\napp.main()", default_timeout=60).run()
        screen.button[1].click().run()
        assert not screen.exception, screen.exception
        assert not screen.chat_input[0].disabled
        screen.chat_input[0].set_value("출장 식비?").run()
        assert not screen.exception, screen.exception
        assert len(screen.chat_message) == 2
        assert any("test.pdf" in item.value for item in screen.text)
        # 새 브라우저 세션과 문서/앱 버전 변경 후에도 저장된 대화가 복원됩니다.
        restored = AppTest.from_string("import app\napp.main()", default_timeout=60).run()
        assert not restored.exception
        assert len(restored.chat_message) == 2
        restored.session_state["data_token"] = "old-version"
        restored.run()
        assert len(restored.chat_message) == 2
        screen.button[0].click().run()
        assert len(screen.chat_message) == 0
        assert app.load_history() == []
    print("PASS: Streamlit 문서/질문/출처, 새 세션 대화 복원, 문서 재준비 후 보존, 초기화")
    return docs


def live_checks(docs):
    key = app.read_api_key()
    if not key:
        raise RuntimeError(".env에 실제 OPENAI_API_KEY가 필요합니다.")
    store = app.build_store(docs, key)
    assert len(store.store) == len(app.split_documents(docs))
    response = app.answer_question("공무원 국내 출장 시 식비는 1일 얼마인가요?", store, key)
    assert response["sources"], "실제 문서 질문에 유효한 출처가 없습니다."
    print("LIVE answer:", response["answer"])
    for source in response["sources"]:
        print("LIVE source:", source["source"], source["page"], source["quote"])
    unknown = app.answer_question("화성의 지름은 몇 km인가요?", store, key)
    print("OUT-OF-SCOPE answer:", unknown["answer"])
    assert unknown["answer"] == app.UNKNOWN and not unknown["sources"]
    print("PASS: 실제 OpenAI 임베딩, gpt-4o-mini 문서 답변 및 문서 밖 질문 거부")
    questions = [
        "2박3일 출장기간 필요한 여비를 계산해주면 좋을 것 같아! 1박은 강화도에서 숙박하고 나머지 1박은 서울시 중구에서 숙박할 예정이야",
        "3박4일 국외출장을 하려고 하는데 일본의 경우 필요여비에 대해서 알려줘",
    ]
    for question in questions:
        response = app.answer_question(question, store, key)
        print("REGRESSION question:", question)
        print("REGRESSION answer:", response["answer"])
        assert response["answer"] != app.UNKNOWN, response
        assert response["sources"], response
        assert "확인" in response["answer"]
        assert "×" not in response["answer"], "조건 미확정 질문에 확정 계산식을 노출했습니다."
        print("REGRESSION sources:", [(source["source"], source["page"]) for source in response["sources"]])
    print("PASS: 화면에 나온 국내/일본 출장 질문 회귀 검사")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    args = parser.parse_args()
    try:
        documents = app.load_documents()[0] if args.live_only else local_checks()
        if args.live or args.live_only:
            try:
                live_checks(documents)
            except Exception as error:
                # API 키를 포함할 수 있는 원본 오류 메시지는 출력하지 않습니다.
                print("LIVE FAILED:", app.error_message(error))
                sys.exit(1)
    finally:
        TMP_DIR.cleanup()
