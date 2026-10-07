# DATA 문서 RAG 챗봇

Python 3.11, Streamlit, OpenAI, 최신 LangChain Runnable로 만든 문서 기반 챗봇입니다.
현재 화면 버전은 **ver5**이며, 답변 아래 출처 파일명·PDF 페이지·근거 원문을 표시합니다.

## 주요 기능

- DATA 하위 폴더의 PDF/TXT/MD/CSV/JSON 읽기
- `text-embedding-3-small` 임베딩과 `InMemoryVectorStore` 검색
- `gpt-4o-mini` 답변 생성 및 별도 자동 검토
- 최근 대화 10개를 반영한 후속 질문 검색어 보정
- 이전 답변은 사실 근거로 사용하지 않고 DATA 원문만 인용
- 조건이 부족하거나 질문 대상이 모호하면 확인 질문 반환
- 대화 초기화 버튼, 로컬 대화 저장, Cloud 이용자별 세션 분리
- API 호출 없는 자동검증과 청크별 참고답안 후보

`LLMChain`, `ConversationChain`, `RetrievalQA`를 사용하지 않습니다.
`ChatPromptTemplate | model`과 `invoke`를 사용하는 Runnable 방식입니다.

## 로컬 실행

먼저 Python 3.11과 uv를 준비하고 프로젝트 폴더에서 실행하세요.

```powershell
uv sync --locked
```

`.env.example`을 `.env`로 복사하고 실제 키를 입력합니다.

```dotenv
OPENAI_API_KEY=여기에_실제_API_키
```

```powershell
uv run streamlit run app.py
```

[http://localhost:8501](http://localhost:8501)을 열고 **문서 준비**를 누른 뒤 질문하세요.
문서 준비, 질문 보정, 임베딩, 답변 생성·검토에는 OpenAI API 사용료가 발생합니다.
벡터는 메모리에 저장되므로 서버/세션 재시작이나 DATA 변경 후 다시 준비해야 합니다.

uv가 PATH에 없는 Windows PC에서는 다음처럼 실행할 수 있습니다.

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run streamlit run app.py
```

이 프로젝트를 작업한 PC에서 가상환경을 다시 생성하려면 아래 설정을 사용할 수 있습니다.

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --locked
```

## 후속 질문 예시

1. “서울 거주, 세종 근무자가 대구로 출장합니다.”
2. “집에서 바로 출발하면 운임은요?”

확인한 사용자 조건을 이어받아 독립적인 검색 질문으로 보정합니다.
최신 사용자 조건을 우선하며, 새 주제에는 이전 조건을 임의로 붙이지 않습니다.
이전 답변의 수치나 주장은 DATA에서 다시 확인해야 합니다.

## 대화 기록

- 로컬 `.env`가 있는 환경: `.chat_history.sqlite3`에 자동 저장하며 같은 PC의 탭들이 기록을 공유합니다.
- `.env`를 업로드하지 않는 Cloud: 이용자별 Streamlit 세션에 보관하며 새로고침·세션 종료 시 사라질 수 있습니다.
- **대화 초기화** 버튼은 해당 저장 방식의 대화 기록을 지웁니다.
- Cloud 장기 보관에는 이용자 인증과 외부 저장소가 추가로 필요합니다.

## Streamlit Community Cloud 배포

[Streamlit Cloud](https://share.streamlit.io/)에서 다음을 선택하세요.

| 설정 | 값 |
|---|---|
| Repository | `jeje7706/chatbot` |
| Branch | `main` |
| Main file path | `app.py` |
| Advanced settings → Python version | `3.11` |

**Advanced settings → Secrets**에만 실제 키를 입력하세요.
배포 후에는 **App settings → Secrets**에서 변경할 수 있습니다.

```toml
OPENAI_API_KEY = "여기에 실제 API 키"
```

키 읽기 순서는 환경 변수 → Streamlit Secrets → 로컬 `.env`입니다.
`.env`, `.streamlit/secrets.toml`, 대화 DB, 가상환경·캐시는 Git 제외 대상입니다.
실제 키를 코드·README·GitHub에 입력하지 마세요.
Cloud는 `uv.lock`을 인식하므로 별도의 requirements.txt는 추가하지 않습니다.
공개 앱의 방문자 사용 비용은 설정한 API 키에 청구됩니다.

[Cloud Secrets 공식 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## API 호출 없는 자동검증

확인 질문이나 패키지 다운로드 없이 현재 가상환경으로 실행합니다. API 키도 필요하지 않습니다.

```powershell
.\verify.ps1
# 경고까지 실패로 취급
.\verify.ps1 -Strict
```

PowerShell 실행 정책으로 제한되면 Python으로 직접 실행하세요.

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error scripts\verify_offline.py
```

문서 페이지 추출, 청크·후보 정합성, 인용 원문 일치, 기대 페이지 존재,
모의 검색·답변, 잘못된 답변 차단, 화면, 대화 복원·초기화를 검사합니다.
OpenAI SDK 요청과 외부 TCP 접속을 차단하고 실제 대화 DB와 분리된 테스트 DB를 사용합니다.

결과: `validation/offline_report.json`

종료 코드: `0` 기술 검사 통과, `1` 실패, `2` Strict 모드의 경고 발견.
기술 검사 통과와 별도로 원본·후보 검토 경고가 남을 수 있습니다.
실제 OpenAI 검색·답변 품질이나 PDF 표의 의미 정확성을 재평가하지는 않습니다.

## 실제 API 검증 및 참고답안 후보

아래 명령은 API 사용료가 발생합니다.

```powershell
uv run python -W error scripts/test_rag.py --live
uv run python -W error scripts/test_reported_issues.py
uv run python -W error scripts/test_followup.py
uv run python -W error scripts/validate_rag.py --evaluate
uv run python -W error scripts/validate_rag.py --generate
```

Cloud 설정 읽기는 API 없이 검사할 수 있습니다.

```powershell
uv run python -W error scripts/test_cloud_config.py
```

| 파일 | 내용 |
|---|---|
| `validation/extraction_report.json` | 페이지·문자 수, 누락·의심 페이지, 원본 해시 |
| `validation/questions.json` | 제보 질문, 기대 근거 페이지, 필수·금지 표현 |
| `validation/evaluation_report.json` | 등록 질문의 검색·답변 검증 결과 |
| `validation/chunk_candidates.jsonl` | 청크 원문, 질문, 참고답안 후보, 인용 검사 상태 |
| `validation/candidate_summary.json` | 후보 생성·검토 필요 항목 수 |
| `validation/offline_report.json` | 최근 API 없는 검사 결과 |

청크별 답안은 **자동 생성 후보**입니다. 사람 검토 전에는 확정 모범답안이 아닙니다.
원문·질문·답안을 대조해 수정한 뒤 `human_approved=true`로 승인하세요.
검증할 질문은 `questions.json`에 추가하며, 자동 생성 후보와 별도 질문도 사용하세요.
사이드바 **검증 보고서·청크 참고답안**에서 파일을 내려받을 수 있습니다.

## 검증의 범위와 한계

각 답변의 **이 답변의 검증 결과**에서 인용·사실/숫자·질문 조건 검사를 확인합니다.
같은 계열 LLM의 자동 검토이므로 사람 검토를 대체하지 않습니다.
표의 열 정렬·읽는 순서, 스캔 이미지의 OCR은 원본 PDF와 대조해야 합니다.
스캔 PDF에는 별도의 OCR 처리가 필요합니다.
문서별 버전과 단가가 다를 수 있으며, 인용 문서의 사례 결과와 현행 규정은 구분해야 합니다.
과거 검증 보고서는 해당 버전과 등록 질문의 결과로 표시합니다.

## 프로젝트 구성

```text
app.py                    Streamlit 앱
pyproject.toml / uv.lock   Python 의존성
DATA/                     검색할 문서
scripts/                  환경·API·오프라인 검증 도구
validation/               평가 질문·보고서·참고답안 후보
verify.ps1                Windows 자동검증 실행
.env.example              키 설정 예시
```

앱 버전은 `app.py`의 `DISPLAY_VERSION`으로 관리합니다.
검색 인덱스의 `APP_VERSION`과 분리되어 화면 버전 변경으로 대화를 초기화하지 않습니다.
