"""추출 검사, 제보 질문 평가, 청크별 참고답안 후보를 생성합니다."""

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import app
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient


class Candidate(BaseModel):
    question: str
    reference_answer: str
    evidence_quotes: list[str] = Field(description="청크에서 연속된 원문을 그대로 복사")
    answerable: bool
    limitations: list[str]


def write_json(filename, payload):
    app.VALIDATION_DIR.mkdir(exist_ok=True)
    destination = app.VALIDATION_DIR / filename
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def chunk_id(doc):
    payload = json.dumps([doc.metadata["source"], doc.metadata.get("page"), doc.page_content], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def extraction_report(documents):
    files = []
    for relative, _, _ in app.data_signature():
        path = app.DATA_DIR / relative
        docs = [doc for doc in documents if doc.metadata["source"] == path.relative_to(app.DATA_DIR).as_posix()]
        total = len(PdfReader(path).pages) if path.suffix.lower() == ".pdf" else 1
        pages = {doc.metadata.get("page") or 1 for doc in docs}
        files.append({"source": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "total_pages": total, "extracted_pages": len(pages), "empty_or_missing_pages": sorted(set(range(1, total + 1)) - pages), "characters": sum(len(doc.page_content) for doc in docs), "suspicious_pages": [doc.metadata.get("page") for doc in docs if len(app.normalize(doc.page_content)) < 30 or "\ufffd" in doc.page_content]})
    chunks = app.split_documents(documents)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "app_version": app.DISPLAY_VERSION, "files": files, "chunks": len(chunks), "coverage_pass": all(not item["empty_or_missing_pages"] for item in files), "quality_requires_review": any(item["suspicious_pages"] for item in files), "limitations": "표 구조와 읽는 순서는 사람이 원본과 대조해야 합니다."}
    report["limitations"] = "페이지 누락/빈 텍스트/깨진 문자 검사는 자동화했습니다. 표의 열 정렬·읽는 순서·이미지 내용은 PDF 원본과 사람이 대조해야 합니다."
    write_json("extraction_report.json", report)
    print("추출 검사:", len(files), "파일", len(documents), "페이지", len(chunks), "청크")
    return chunks


async def generate_candidates(chunks, key):
    output = app.VALIDATION_DIR / "chunk_candidates.jsonl"
    current_ids = {chunk_id(doc) for doc in chunks}
    completed = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["chunk_id"] in current_ids and row.get("generation_pass"):
                completed[row["chunk_id"]] = row
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in completed.values()), encoding="utf-8")
    semaphore = asyncio.Semaphore(3)
    # 전용 HTTP 연결을 사용하여 다른 모델 호출의 공유 연결을 닫지 않게 합니다.
    async_client = DefaultAsyncHttpxClient()
    sync_client = DefaultHttpxClient()
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=key, temperature=0, timeout=60, max_retries=2, http_async_client=async_client, http_client=sync_client)
    model = llm.with_structured_output(Candidate, method="json_schema", strict=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "청크별 평가용 질문과 참고답안 후보를 하나 만드세요. 주어진 청크만 근거로 답하세요. 청크에 없는 수치/조건/다른 페이지를 보충하지 마세요. 표의 일부라서 답할 수 없으면 answerable=false로 표시하고 필요한 문맥을 limitations에 적으세요. 질문은 이 청크에서 확인할 수 있는 규정을 구체적으로 물어야 합니다. reference_answer에 적용 대상과 예외를 명시하세요. evidence_quotes는 최소 하나의 연속된 원문을 그대로 복사하세요. 이것은 자동 생성 후보이며 인증된 정답이 아닙니다."),
        ("human", "파일 {source}, PDF 페이지 {page}\n청크:\n{text}"),
    ])
    chain = prompt | model

    async def generate(doc):
        identity = chunk_id(doc)
        if identity in completed:
            return
        async with semaphore:
            row = {"chunk_id": identity, "source": doc.metadata["source"], "page": doc.metadata.get("page"), "chunk_text": doc.page_content, "model": "gpt-4o-mini", "app_version": app.DISPLAY_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "review_status": "unreviewed", "human_approved": False}
            try:
                for attempt in range(2):
                    candidate = await chain.ainvoke({"source": row["source"], "page": row["page"], "text": doc.page_content})
                    quote_pass = bool(candidate.evidence_quotes) and all(len(app.normalize(quote)) >= 8 and app.normalize(quote) in app.normalize(doc.page_content) for quote in candidate.evidence_quotes)
                    if quote_pass:
                        break
                row.update(candidate.model_dump())
                row["generation_pass"] = quote_pass and bool(candidate.question.strip()) and bool(candidate.reference_answer.strip())
                row["quote_check_pass"] = quote_pass
                if not row["generation_pass"]:
                    row["review_status"] = "generation_needs_review"
            except Exception as error:
                row.update(generation_pass=False, review_status="generation_failed", error_type=type(error).__name__)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[identity] = row
            if len(completed) % 10 == 0 or len(completed) == len(chunks):
                print(f"참고답안 후보 {len(completed)}/{len(chunks)}", flush=True)

    try:
        await asyncio.gather(*(generate(doc) for doc in chunks))
    finally:
        # Windows에서는 이벤트 루프가 닫히기 전에 API 연결을 명시적으로 닫습니다.
        await llm.root_async_client.close()
        llm.root_client.close()
        await asyncio.sleep(0.25)
    # 원문 순서로 정렬하며 중단 후 재개할 때도 중복을 방지합니다.
    output.write_text("".join(json.dumps(completed[chunk_id(doc)], ensure_ascii=False) + "\n" for doc in chunks), encoding="utf-8")
    failures = sum(not row.get("generation_pass") for row in completed.values())
    write_json("candidate_summary.json", {"chunks": len(chunks), "generated": len(completed), "generation_failures": failures, "human_approved": 0, "note": "모두 사람의 검토가 필요한 후보입니다."})
    print("후보 생성 완료:", len(completed), "검사 실패:", failures)


def evaluate(documents, key):
    path = app.VALIDATION_DIR / "questions.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    store = app.build_store(documents, key)
    rows = []
    for case in cases:
        response = app.answer_question(case["question"], store, key)
        expected = {(item["source"], item["page"]) for item in case["expected_pages"]}
        retrieved = {(item["source"], item["page"]) for item in response.get("retrieval", [])}
        cited = {(item["source"], item["page"]) for item in response["sources"]}
        compact = app.normalize(response["answer"])
        rules_pass = all(any(app.normalize(term) in compact for term in alternatives) for alternatives in case["required_any"]) and not any(term in response["answer"] for term in case["forbidden"])
        audit = response.get("verification", {})
        answer_pass = rules_pass and (all(audit.get(field, False) for field in ("grounded", "matches_question", "citations_match")) if case["expected_pages"] else response["answer"] == app.UNKNOWN)
        row = {"id": case["id"], "question": case["question"], "retrieval_expected_page_hit": bool(expected & retrieved) if expected else None, "retrieval_expected_page_recall": len(expected & retrieved) / len(expected) if expected else None, "citation_expected_page_hit": bool(expected & cited) if expected else None, "answer_rules_pass": rules_pass, "answer_pass": answer_pass, "response": response}
        rows.append(row)
        print("질문 평가:", case["id"], "검색", row["retrieval_expected_page_hit"], "답변", answer_pass)
    applicable = [row for row in rows if row["retrieval_expected_page_hit"] is not None]
    write_json("evaluation_report.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "app_version": app.DISPLAY_VERSION, "dataset_cases": len(rows), "retrieval_hit_rate": sum(row["retrieval_expected_page_hit"] for row in applicable) / len(applicable), "answer_pass_rate": sum(row["answer_pass"] for row in rows) / len(rows), "note": "문서 근거를 지정한 소수 질문의 검사 결과입니다. LLM 판정을 포함하며 모든 질문의 정확성을 보장하지 않습니다.", "cases": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    docs, _, _ = app.load_documents()
    chunks = extraction_report(docs)
    key = app.read_api_key()
    if args.generate or args.evaluate:
        if not key:
            raise RuntimeError(".env에 OPENAI_API_KEY가 필요합니다.")
    if args.generate:
        asyncio.run(generate_candidates(chunks, key))
    if args.evaluate:
        evaluate(docs, key)

