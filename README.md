# DATA 문서 RAG 챗봇

Python 3.11 / Streamlit / LangChain Runnable / OpenAI 기반 앱입니다.
프로젝트 최상단의 app.py를 실행합니다.

제목 옆의 화면 버전은 현재 `ver5`입니다. 기능 변경 시 app.py의
`DISPLAY_VERSION`을 `ver2`, `ver3` 순으로 올립니다. 검색 인덱스용
`APP_VERSION`과 별도로 관리하여 화면 버전 변경으로 대화를 초기화하지 않습니다.

## Streamlit Community Cloud 배포

GitHub의 `.env`와 `.streamlit/secrets.toml`은 제외 상태를 유지하세요.
현재 코드 변경을 GitHub에 반영한 뒤 Community Cloud에서 아래를 선택합니다.

- Repository: `jeje7706/chatbot`
- Branch: `main`
- Main file path: `app.py`
- Advanced settings → Python version: `3.11`
- Advanced settings → Secrets (배포 후에는 App settings → Secrets):

```toml
OPENAI_API_KEY = "실제 키를 Cloud 설정 화면에만 입력"
```

코드·README·GitHub 파일에는 실제 키를 입력하지 마세요. 앱은 환경 변수,
Streamlit Secrets, 로컬 `.env` 순으로 키를 읽습니다. 실제 키가 담긴 Secrets 파일은
새로 만들 필요 없이 Cloud 설정 화면에 직접 입력하면 됩니다.
Cloud는 저장소의 `uv.lock`을 인식하므로 별도 requirements.txt를 추가하지 않습니다.

로컬에서는 기존 SQLite 대화를 유지합니다. `.env`를 올리지 않는 Cloud에서는
이용자별 세션으로 대화를 분리하며 새로고침/세션 종료 시 사라질 수 있습니다.
Cloud에서 이용자별 장기 보관이 필요하면 인증과 외부 저장소를 별도로 도입해야 합니다.
Cloud 인스턴스의 로컬 파일을 영구 저장소로 가정하지 마세요.

공개 앱에서는 방문자의 API 사용 비용이 설정한 키에 청구됩니다. 앱의 공유 범위를
확인하고 배포용 OpenAI 프로젝트와 키를 사용하세요.

## 검증 및 청크별 참고답안

### API 없이 한 번에 검사

프로젝트 폴더의 PowerShell에서 다음 명령을 실행하세요. 확인 질문, API 호출,
패키지 다운로드 없이 기존 가상환경을 사용합니다. API 키도 필요하지 않습니다.

```powershell
.\verify.ps1
# 경고도 실패로 취급하려면
.\verify.ps1 -Strict
```

스크립트 실행 정책으로 PowerShell 파일 실행이 제한된 경우 Python으로 직접 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error scripts\verify_offline.py
```

검사 범위는 문서 페이지 추출, 청크/후보 ID 일치, 저장된 인용과 원문 비교,
기대 근거 페이지 존재, 보고서 구조, 모의 검색/답변, 잘못된 답변 차단,
Streamlit 화면, 대화 복원 및 초기화입니다. 실제 대화 DB는 테스트용 DB와 분리합니다.
OpenAI SDK 요청과 외부 TCP 접속은 검사 중 차단합니다.
결과는 `validation/offline_report.json`에 저장합니다.

종료 코드: `0` 기술 검사 통과(검토 경고는 있을 수 있음), `1` 실패,
`2` Strict 모드에서 검토 경고 발견. 실패/경고가 있는데 모두 통과했다고 표시하지 않습니다.
현재 후보의 미일치 인용 및 텍스트가 적은 페이지는 경고로 남깁니다.

**실제 OpenAI 검색·답변 성능을 새로 평가하거나 모범답안의 의미 정확성을 판정하지 않습니다.**
저장된 API 평가 보고서는 과거 결과로 구분하며, PDF 표/후보 답안의 의미는 원본과 사람이 대조해야 합니다.

화면의 각 답변 아래 **이 답변의 검증 결과**에서 인용 원문 일치, 사실/숫자/계산,
질문 조건 일치를 확인할 수 있습니다. 사실과 질문 조건은 gpt-4o-mini의 별도
검토 호출로 검사합니다. 자동 검토에서 실패한 답변은 확정 답변으로 표시하지 않습니다.
같은 계열 모델의 검토이므로 독립적인 인증이나 사람 검토의 대체물이 아닙니다.
검토 호출에는 추가 API 사용료가 발생합니다.

```powershell
# API 없이 페이지 누락·빈 텍스트·깨진 문자 검사
uv run python -W error scripts/validate_rag.py
# 제보 질문의 검색/인용/답변 검사 (실제 API 사용)
uv run python -W error scripts/validate_rag.py --evaluate
# 모든 청크의 질문·참고답안 후보 생성 (실제 API 사용, 중단 후 재개 가능)
uv run python -W error scripts/validate_rag.py --generate
```

- `validation/extraction_report.json`: PDF별 페이지/문자 수, 누락·의심 페이지, 원본 해시.
- `validation/questions.json`: 제보 질문과 원문을 대조해 작성한 기대 근거 페이지, 필수/금지 표현. 사람이 확인·수정하고 검증할 질문을 여기에 추가하세요.
- `validation/evaluation_report.json`: 질문별 검색 근거 페이지 적중 여부/회수율, 인용 페이지 일치, 규칙 검사, 자동 답변 검토.
- `validation/chunk_candidates.jsonl`: 청크 ID/원문/질문/참고답안/인용/생성 검사 결과.

**청크별 후보는 인증된 모범답안이 아닙니다.** 모두 `human_approved=false`로
표시했습니다. 사람 검토 후 질문·답안·근거를 수정하고 `human_approved=true`로
바꾼 항목만 확정 평가 자료로 사용하세요. 생성 때 쓰지 않은 별도 질문도 추가하여
평가 편향을 줄이세요. 청크 후보는 등록된 네 질문의 평가 결과와 별도입니다.
후보 전체를 정답으로 삼아 챗봇 성능을 평가하지 않습니다.

페이지 추출 성공은 PDF 표의 열/읽는 순서나 이미지 내용을 정확히 읽었다는 뜻은
아닙니다. 의심 페이지와 중요한 표는 원본 PDF를 사람이 대조해야 합니다.
검색 점수는 원문을 확인한 기대 페이지가 최종 검색 문맥에 포함되는지 기준이며,
최종 문맥의 배열 순서를 전역 검색 순위로 해석하지 않습니다.

사이드바의 **검증 보고서·청크 참고답안**에서 보고서와 후보 파일을 내려받을 수 있습니다.

## 실행

.env의 OPENAI_API_KEY에 실제 키를 입력하고 저장하세요.

```powershell
uv sync --locked
uv run streamlit run app.py
```

이 PC에서는 uv가 PATH에 없으므로 다음 명령으로 실행할 수 있습니다.

```powershell
& "$env:USERPROFILE\.local\bin\uv.exe" run streamlit run app.py
```

브라우저에서 http://localhost:8501 을 열고 **문서 준비**를 누른 뒤 질문하세요.
문서 준비 시 텍스트를 OpenAI에 보내 임베딩하므로 API 사용료가 발생합니다.
벡터는 메모리에만 보관하며 서버/세션 재시작이나 DATA 변경 후 다시 준비합니다.

## 동작

- DATA 하위 폴더의 모든 PDF/TXT/MD/CSV/JSON을 읽습니다. 다른 형식은 오류로 알립니다.
- PDF는 페이지별로 추출합니다. 스캔 이미지는 OCR이 필요하며 빈 페이지를 화면에 알립니다.
- 임베딩: text-embedding-3-small / 답변: gpt-4o-mini / 벡터DB: InMemoryVectorStore
- 복합 질문을 항목별 검색어로 확장하고 파일별 검색 결과를 합칩니다.
- 검색된 PDF 조각의 원래 페이지 전체를 제공하여 표/단가/예외가 잘리지 않게 합니다.
- ChatPromptTemplate과 구조화된 ChatOpenAI를 Runnable로 조합하고 invoke를 호출합니다.
- 각 답변 문장에 근거 번호를 요구합니다. 표시할 근거 문장은 모델이 재작성하지 않고 코드가 PDF 원문에서 직접 가져옵니다.
- 문서 주제와 무관하거나 실제 관련 근거가 없는 질문에는 근거를 찾을 수 없다고 답합니다.
- 문서 단가와 사용자 일수/박수에 근거한 계산은 허용하고 계산식과 조건을 표시합니다.
- 직급, 방문 도시, 실제 지출액 등이 부족하면 확인 가능한 규정을 설명하고 추가 정보를 요청합니다.
- 적용 조건이 부족한 답변은 확정 금액이 포함된 생성 문장을 표시하지 않고 실제 근거 원문과 확인 질문을 먼저 표시합니다.
- 유효하지 않은 근거 번호는 한 번 재시도하고, 끝내 실패하면 검증 실패라고 구분해 표시합니다.
- 답변 아래에 파일명, PDF 페이지 번호, 근거 문장을 표시합니다.
- 최근 대화 10개를 바탕으로 후속 질문을 독립적인 검색 질문으로 보정합니다. 최신 사용자 조건이 우선이며 모호한 대상은 되묻습니다.
- 답변에도 대화 문맥을 전달하되 이전 답변을 사실 근거로 재사용하지 않습니다. DATA 검색 원문만 근거로 사용하고 검증합니다.
- 대화는 `.chat_history.sqlite3`에 자동 저장합니다. 새로고침/서버 재시작/문서 재준비 후에도 복원됩니다.
- 이 PC의 탭들이 같은 대화를 사용합니다. 사이드바의 **대화 초기화**로 저장된 기록을 지울 수 있습니다. 벡터 DB는 유지됩니다.
- 대화 저장 파일은 Git에서 제외되며 API 키는 저장하지 않습니다.
- 문서 사례의 지명을 사용자 지명으로 바꿔치기하지 않도록 질문의 거주지/근무지를 별도로 표시합니다.
- 근거 번호는 JSON 스키마에서 유효한 번호만 선택하도록 제한합니다.
- 동일한 지출액의 문서 사례는 해당 문서 기준의 결과로 안내합니다.
- 민간기업 자체 출장비와 공무 수행에 관한 비공무원 지급 구분표는 적용 범위를 구분합니다.
- 문서 버전이 다르면 답이 다를 수 있습니다. 제공된 문서 기준이며 현행 규정 여부는 단정하지 않습니다.

인용문 검사는 출처 조작을 줄이지만 답변과 근거의 의미 일치까지 완벽하게 보장하지는 않습니다.
중요한 결정 전에 표시된 원문과 페이지를 확인하세요.

## 검사

```powershell
uv run python -W error scripts/check_environment.py
uv run python -W error scripts/test_rag.py
# 실제 API 호출 및 전체 문서 임베딩 포함 (사용료 발생)
uv run python -W error scripts/test_rag.py --live
```

테스트 도구의 테스트 스레드에서만 발생하는 세션 누락 안내는 테스트에서 제외합니다.
실제 앱의 경고 및 Python deprecated 경고는 숨기지 않습니다.

앱 형태에 맞춰 pyproject.toml의 [tool.uv]에 package = false를 설정했습니다.
불필요한 자체 패키지 설치와 Python 3.11의 한글 editable 경로 오류를 방지합니다.
link-mode = "copy"는 Windows 하드링크 경고를 방지합니다.

이 PC에서는 uv 기본 Python 저장소 오류로 Python을 .uv-python에 설치했습니다.
가상환경 재생성 시 아래 설정을 먼저 적용하세요.

```powershell
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.uv-python'
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --locked
```
#   c h a t b o t  
 
