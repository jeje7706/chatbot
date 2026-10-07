"""API 호출 없이 문서·저장 자료·앱 동작을 한 번에 검사합니다."""

import argparse
import ipaddress
import json
import os
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
TEMP_ROOT = ROOT / ".offline-validation-tmp"
TEMP_ROOT.mkdir(exist_ok=True)
tempfile.tempdir = str(TEMP_ROOT)

results = []
blocked_requests = []


def record(name, status, detail):
    results.append({"name": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}")


def no_api(*args, **kwargs):
    blocked_requests.append("OpenAI SDK request")
    raise RuntimeError("오프라인 검증 중 API 요청을 차단했습니다.")


original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex


def check_address(address):
    # 테스트 도구 내부의 루프백 통신만 허용합니다. 외부 서버는 접속 전에 차단합니다.
    if isinstance(address, tuple):
        host = str(address[0])
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host.lower() == "localhost"
        if allowed:
            return
    blocked_requests.append("external socket")
    raise RuntimeError("오프라인 검증 중 외부 네트워크 요청을 차단했습니다.")


def local_connect(self, address):
    check_address(address)
    return original_connect(self, address)


def local_connect_ex(self, address):
    check_address(address)
    return original_connect_ex(self, address)


def validate_artifacts(app, documents, chunks):
    from validate_rag import chunk_id, extraction_report
    extraction_report(documents)
    report = json.loads((app.VALIDATION_DIR / "extraction_report.json").read_text(encoding="utf-8"))
    if not report["coverage_pass"]:
        record("페이지 추출", "FAIL", "누락/빈 페이지가 있습니다. extraction_report.json 확인")
    else:
        record("페이지 추출", "PASS", f"{len(documents)}페이지, {len(chunks)}청크")
    if report["quality_requires_review"]:
        record("추출 품질", "WARN", "문자가 적거나 깨진 페이지가 있습니다. 원본 대조 필요")
    chunk_map = {chunk_id(doc): doc for doc in chunks}
    path = app.VALIDATION_DIR / "chunk_candidates.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [row["chunk_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(chunk_map):
            record("후보 범위", "FAIL", "청크 중복/누락 또는 DATA 변경 후 오래된 후보가 있습니다.")
        else:
            record("후보 범위", "PASS", f"현재 청크 {len(ids)}개와 1:1 일치")
        flagged = 0
        for row in rows:
            doc = chunk_map.get(row["chunk_id"])
            quotes_match = doc is not None and bool(row.get("evidence_quotes")) and all(len(app.normalize(quote)) >= 8 and app.normalize(quote) in app.normalize(doc.page_content) for quote in row.get("evidence_quotes", []))
            metadata_match = doc is not None and row["chunk_text"] == doc.page_content and row["source"] == doc.metadata["source"] and row["page"] == doc.metadata.get("page")
            if not metadata_match or (row.get("generation_pass") and not quotes_match):
                record("후보 정합성", "FAIL", f"원문/출처 불일치 또는 잘못된 통과 표시: {row['chunk_id']}")
            if not row.get("generation_pass") or not quotes_match:
                flagged += 1
            if row.get("human_approved") and not quotes_match:
                record("검토 승인", "FAIL", f"인용 미일치 후보에 승인 표시: {row['chunk_id']}")
        record("후보 인용 검사", "WARN" if flagged else "PASS", f"{len(rows)}개 중 검토 필요 {flagged}개. 후보 답안의 의미 정확성은 사람 검토 필요")
    else:
        record("참고답안 후보", "WARN", "후보 파일이 없습니다. 이 도구는 API로 후보를 생성하지 않습니다.")
    questions_path = app.VALIDATION_DIR / "questions.json"
    if questions_path.exists():
        cases = json.loads(questions_path.read_text(encoding="utf-8"))
        available = {(doc.metadata["source"], doc.metadata.get("page")) for doc in documents}
        if len({case["id"] for case in cases}) != len(cases):
            record("질문 데이터", "FAIL", "질문 ID가 중복됩니다.")
        elif any((page["source"], page["page"]) not in available for case in cases for page in case["expected_pages"]):
            record("질문 데이터", "FAIL", "기대 근거 파일/페이지가 현재 DATA에 없습니다.")
        else:
            record("질문 데이터", "PASS", f"등록 질문 {len(cases)}개의 기대 페이지 존재 확인")
    # 과거 실제 API 보고서는 읽기만 하고 새 실제 성능 검사라고 표시하지 않습니다.
    previous = app.VALIDATION_DIR / "evaluation_report.json"
    if previous.exists():
        saved = json.loads(previous.read_text(encoding="utf-8"))
        count = len(saved["cases"])
        if saved["dataset_cases"] != count or not all(0 <= saved[key] <= 1 for key in ("retrieval_hit_rate", "answer_pass_rate")):
            record("저장된 평가 보고서", "FAIL", "항목 수 또는 점수 범위가 잘못됐습니다.")
        else:
            record("저장된 평가 보고서", "PASS", f"{count}개 항목 구조 확인. 이번 실행에서 실제 OpenAI 성능을 재평가하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="WARN도 실패 종료 코드로 처리")
    args = parser.parse_args()
    started = time.monotonic()
    # 실제 키를 읽을 필요가 없습니다. 추적 서비스의 외부 호출도 방지합니다.
    settings = {"OPENAI_API_KEY": "offline-placeholder", "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"}
    with patch.dict(os.environ, settings), patch.object(socket.socket, "connect", local_connect), patch.object(socket.socket, "connect_ex", local_connect_ex):
        import app
        import openai._base_client as sdk
        with patch.object(sdk.SyncAPIClient, "request", no_api), patch.object(sdk.AsyncAPIClient, "request", no_api), patch.object(app, "read_api_key", return_value=""):
            tests = None
            try:
                import test_rag as tests
                docs = tests.local_checks()
                record("모의 검색·답변·화면·대화 저장", "PASS", "실제 InMemoryVectorStore와 모의 LLM으로 검사. 실제 OpenAI 임베딩 검색 품질 검사는 포함하지 않습니다.")
                with patch.object(socket.socket, "connect", local_connect), patch.object(socket.socket, "connect_ex", local_connect_ex):
                    validate_artifacts(app, docs, app.split_documents(docs))
            except Exception as error:
                record("자동 검사 실행", "FAIL", f"{type(error).__name__}: {error}")
            finally:
                if tests is not None:
                    tests.TMP_DIR.cleanup()
    record("API·외부 접속", "FAIL" if blocked_requests else "PASS", f"API 및 외부 접속 시도 {len(blocked_requests)}회. 모든 시도는 차단됩니다.")
    fail_count = sum(item["status"] == "FAIL" for item in results)
    warn_count = sum(item["status"] == "WARN" for item in results)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": "offline", "api_requests": 0, "blocked_attempts": len(blocked_requests), "app_version": app.DISPLAY_VERSION, "duration_seconds": round(time.monotonic() - started, 2), "failures": fail_count, "warnings": warn_count, "status": "FAIL" if fail_count else "PASS_WITH_WARNINGS" if warn_count else "PASS", "results": results, "limitations": "자동 인용/구조/로직 검사입니다. 실제 OpenAI 답변 품질, 임베딩 검색 성능, 원본 표의 의미 정확성은 검증하지 않습니다."}
    app.VALIDATION_DIR.mkdir(exist_ok=True)
    destination = app.VALIDATION_DIR / "offline_report.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"완료: 실패 {fail_count}개 / 경고 {warn_count}개 / API 호출 0회")
    print(f"보고서: {destination}")
    return 1 if fail_count else 2 if args.strict and warn_count else 0


if __name__ == "__main__":
    sys.exit(main())
