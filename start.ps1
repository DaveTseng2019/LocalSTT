# 啟動本機語音辨識端點。參數原樣傳給 shim.py（例如 -- --port 8124）。
$python = Join-Path $PSScriptRoot 'repo\.venv\Scripts\python.exe'
$shim = Join-Path $PSScriptRoot 'shim.py'
if (-not (Test-Path $python)) { throw "找不到虛擬環境：$python" }
& $python $shim @args
