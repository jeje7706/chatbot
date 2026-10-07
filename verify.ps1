# API·패키지 다운로드 없이 현재 가상환경으로 검사합니다. 확인 질문은 없습니다.
param([switch]$Strict)
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    Write-Error '가상환경이 없습니다. 먼저 README의 uv sync 방법으로 환경을 준비하세요.'
    exit 1
}
$taskScript = Join-Path $PSScriptRoot 'scripts\verify_offline.py'
if ($Strict) {
    & $taskPython -X utf8 -W error $taskScript --strict
} else {
    & $taskPython -X utf8 -W error $taskScript
}
exit $LASTEXITCODE
