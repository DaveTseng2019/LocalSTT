"""把本機語音辨識模型包成 OpenAI 相容的轉寫端點，給 ZeroType 直接接。

ZeroType 設定：語音辨識服務商選「本機」。沒有 API Key 欄位，端點也不必填。

啟動：repo/.venv/Scripts/python.exe shim.py [--port 8123] [--host 127.0.0.1]
引擎：MOSS-Transcribe-Diarize。FireRed 引擎已在 2026-10-31 移除（git 歷史保留）。
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

# ZeroType 讀這個檔來顯示端點記錄。放在 shim.py 旁邊，路徑對呼叫端是固定的。
LOG_PATH = Path(__file__).with_name("shim.log")


def _hide_console() -> None:
    """把自己的主控台視窗藏起來。

    notes: 不能靠 pythonw.exe。uv 的 .venv\\Scripts\\pythonw.exe 只是跳板，
           它生出來的是基底的 python.exe（主控台版），照樣配到一個空白視窗
           （類別 PseudoConsoleWindow），使用者把它關掉就等於把端點殺掉。
           主控台是行程建立當下配置的，只有行程自己藏得掉——所以這件事必須在
           shim 這一端做，呼叫端做不到。
    notes: 放在最前面、重量級 import 之前。晚一秒藏，視窗就閃一秒。
    """
    if sys.platform != "win32":
        return
    import ctypes

    hwnd = ctypes.windll.kernel32.GetConsoleWindow()
    if hwnd:
        ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE


if "--hide-console" in sys.argv:
    _hide_console()


def _console_stream():
    """要顯示視窗時，取一條真的會出現在主控台上的輸出串流。

    notes: 不能只靠 sys.stdout。ZeroType 用 Dart 的 detached 模式啟動我們，
           那個模式把子行程的標準輸出接到 NUL——實測 GetConsoleMode(STD_OUTPUT)
           回 ERROR_INVALID_HANDLE(6)，print() 出來的字完全沒有進到螢幕緩衝區，
           所以視窗開著卻是全黑的。直接開 CONOUT$ 才寫得進那個緩衝區。
    notes: 手動用 python.exe 跑的時候標準輸出本來就是主控台，這裡回傳的 CONOUT$
           指向同一個緩衝區，行為一致；沒有主控台可開就回 None，交回原本的串流。
    """
    if sys.platform != "win32":
        return None
    try:
        return open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except OSError:
        return None


class _Tee:
    """把寫進標準輸出／標準錯誤的東西同時送到記錄檔。

    notes: 用 pythonw.exe 啟動時沒有主控台，sys.stdout 與 sys.stderr 都是 None，
           一切訊息——函式庫警告、uvicorn 的錯誤、未捕捉例外的 traceback——
           全部無聲消失。ZeroType 只讀得到這個檔，所以「看得見」等於「有寫進這裡」。
    notes: 原本的串流還是照寫（手動用 python.exe 啟動時有主控台），
           所以這是 tee 不是 redirect。
    """

    def __init__(self, path: Path, original) -> None:
        self._file = open(path, "a", encoding="utf-8", buffering=1)
        self._original = original
        self._pending = ""

    def write(self, text: str) -> int:
        if self._original is not None:
            try:
                self._original.write(text)
            except (OSError, ValueError):
                pass

        # notes: 按行收斂再寫檔。進度條（transformers 載權重）用 \r 在同一行重畫幾十次，
        #        原樣寫檔會變成一條幾百字的長行，把 ZeroType 那邊真正要看的訊息擠出畫面。
        #        \r 的語意就是「蓋掉本行前面的內容」，所以每行只留最後一次重畫的結果。
        buffered = self._pending + text
        lines = buffered.split("\n")
        self._pending = lines.pop()
        if "\r" in self._pending:
            self._pending = self._pending.rsplit("\r", 1)[-1]
        try:
            for line in lines:
                self._file.write(line.rsplit("\r", 1)[-1] + "\n")
        except OSError:
            pass
        return len(text)

    def flush(self) -> None:
        for stream in (self._file, self._original):
            if stream is None:
                continue
            try:
                stream.flush()
            except (OSError, ValueError):
                pass

    def isatty(self) -> bool:
        return False


# notes: 這段刻意放在 torch／transformers 之前。那些函式庫在 import 當下就會吐警告，
#        晚一步接管就漏掉了。E402 是知情的取捨。
# notes: 每次啟動清空。這份記錄是給 ZeroType 顯示「這一次執行發生了什麼」用的，
#        不是稽核用的長期日誌——不清就會無限成長，而且沒有輪替機制。
LOG_PATH.write_text("", encoding="utf-8")
_console = None if "--hide-console" in sys.argv else _console_stream()
sys.stdout = _Tee(LOG_PATH, _console or sys.__stdout__)
sys.stderr = _Tee(LOG_PATH, _console or sys.__stderr__)


def _disable_power_throttling() -> None:
    """關掉 Windows 對這個程序的節能節流（EcoQoS）。

    pythonw 沒有視窗，Windows 把它當背景程序節流：排到效率核心、壓低時脈。MOSS 逐字產生時
    每個 token 都要 CPU 送一批小工作給顯示卡，CPU 一慢，顯示卡就閒著、驅動再把它降頻。
    2026-11-05 實測同一段 13.7 秒錄音：節流中 6.7～8.2 秒（221～665 ms/token，十多次），
    關掉後 0.96 秒（34 ms/token，3 次）。
    """
    if sys.platform != "win32":
        return
    import ctypes

    class _State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

    # Version 1；ControlMask=EXECUTION_SPEED(1) 表示「由我指定」；StateMask=0 表示「不要節流」
    st = _State(1, 1, 0)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    ProcessPowerThrottling = 4
    ok = k32.SetProcessInformation(k32.GetCurrentProcess(), ProcessPowerThrottling, ctypes.byref(st), ctypes.sizeof(st))
    print(f"{time.strftime('%H:%M:%S')} 節能節流：{'已關閉' if ok else f'關閉失敗（錯誤碼 {ctypes.get_last_error()}），長句可能變慢'}",
          flush=True)


_disable_power_throttling()

import torch  # noqa: E402
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from opencc import OpenCC


def _arg_value(name: str, default: str) -> str:
    """在 argparse 之前取一個參數值。引擎決定了要 import 什麼，必須在模組層就知道。"""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# /health 與記錄檔用的引擎名稱。只剩 MOSS 一個引擎。
ENGINE = "moss"

from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# 實測 55 筆真實錄音，這組 prompt 的字元差異中位數 0.071，比內建預設的 0.130 好，
# 而且輸出 token 少一個量級（RTF 0.06 對 0.145）。
TRANSCRIBE_PROMPT = "請將音訊逐字轉寫為繁體中文文字。中英混用時保留原本的英文單詞，不要翻譯。"

MARKER = re.compile(r"\[\d+(?:\.\d+)?\]|\[S\d+\]")

# notes: 用 s2tw 而不是 s2twp。s2twp 會做慣用詞替換——實測把模型輸出的「项目」改成
#        「專案」、「内存」改成「記憶體」。那是改掉使用者說過的詞，對語音輸入是靜默竄改。
#        只要字形轉換，不要詞彙轉換。
cc = OpenCC("s2tw")
app = FastAPI()
state: dict = {}

def log(message: str) -> None:
    """加時間戳的訊息。寫進記錄檔這件事由 _Tee 負責，這裡只管格式。"""
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


# ---- 效能紀錄：每句記下耗時與顯示卡狀態 ----
# 2026-11-04 有 32/114 句超過 3 秒，原因是節能節流（見 _disable_power_throttling）。shim.log 每次啟動清空，
# 事後查不到，所以每句另寫一行到 perf.log（只附加、不清空）。再變慢時先看 ms/token（正常約 34～48）。
PERF_LOG = Path(__file__).with_name("perf.log")


def _gpu_snapshot() -> dict:
    """呼叫 nvidia-smi 取整張卡的負載、顯示記憶體、時脈與效能狀態（含別的程式）。

    notes: 每句多一次 nvidia-smi（約 0.1～0.3 秒），放在背景執行緒與推論同時跑，
           不加進回應時間。取到的是「推論開始那一刻」的狀態，不是推論期間的峰值。
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,clocks.sm,pstate",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip().split(", ")
        return {"util": out[0], "used": out[1], "total": out[2], "clock": out[3], "pstate": out[4]}
    except Exception as e:  # 記錄不能拖垮辨識
        return {"error": type(e).__name__}


def _audio_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


PERF_HEADER = ("時間\t耗時s\tms/token\t字數\t音長s\tGPU負載%\t顯示記憶體已用/總共MiB\t本程序MiB\t時脈MHz\t效能狀態"
               "\t輸入tokens\t產生tokens\t原始字數")


def _ms_per_token(elapsed: float, gen_tokens: int) -> str:
    """監控速度用的主要數字：正常約 34～48；超過 120 多半是又被節流了。含音訊編碼時間，短句會偏高。"""
    return f"{elapsed / gen_tokens * 1000:.0f}" if gen_tokens else ""


def _migrate_perf_log() -> None:
    """舊版 perf.log 沒有 ms/token 欄：補上標題，並用舊列的耗時與產生 tokens 算出來插進第 3 欄。"""
    try:
        lines = PERF_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if not lines or "ms/token" in lines[0]:
        return
    out = [PERF_HEADER]
    for line in lines[1:]:
        f = line.split("\t")
        if len(f) == 12:  # 舊格式：…輸入tokens、產生tokens、原始字數
            try:
                f.insert(2, _ms_per_token(float(f[1]), int(f[10])))
            except ValueError:
                f.insert(2, "")
        out.append("\t".join(f))
    try:
        PERF_LOG.write_text("\n".join(out) + "\n", encoding="utf-8")
    except OSError:
        pass


_migrate_perf_log()


def perf_log(elapsed: float, chars: int, audio_s: float | None, gpu: dict,
             prompt_len: int = 0, gen_tokens: int = 0, raw_chars: int = 0) -> None:
    """一句一行，tab 分隔，欄位見 PERF_HEADER。ms/token 放第 3 欄，打開檔案就看得到速度。"""
    own = torch.cuda.memory_reserved() // 2**20 if torch.cuda.is_available() else 0
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), f"{elapsed:.2f}", _ms_per_token(elapsed, gen_tokens), str(chars),
           f"{audio_s:.1f}" if audio_s is not None else "",
           gpu.get("util", ""), f"{gpu.get('used', '')}/{gpu.get('total', '')}", str(own),
           gpu.get("clock", ""), gpu.get("pstate", gpu.get("error", "")),
           str(prompt_len), str(gen_tokens), str(raw_chars)]
    try:
        new = not PERF_LOG.exists()
        with PERF_LOG.open("a", encoding="utf-8") as f:
            if new:
                f.write(PERF_HEADER + "\n")
            f.write("\t".join(row) + "\n")
    except OSError:
        pass


def clean(text: str) -> str:
    """兩道確定性後處理。程式碼能答的不交給模型。

    notes: 剝標記是無條件的。自訂 prompt 實測有 7/55 筆無視指令、照樣吐時間戳，
           所以不能假設 prompt 生效。簡轉繁同理——55 筆裡 30 筆含簡體字，prompt
           寫「繁體中文」壓不住。
    """
    return cc.convert(MARKER.sub("", text)).strip()


def warmup() -> None:
    """用一秒靜音跑一次推論，結果丟掉。

    notes: 第一次推論含 CUDA kernel 載入，實測 8.4 秒音檔要 5.2 秒，穩態只要 0.5 秒。
           不暖機的話「開機後第一句」會慢一個量級，那正是最有感的一句。
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(32000))  # 一秒 16 kHz 16-bit 靜音
        generate_transcription(
            state["model"],
            state["processor"],
            build_transcription_messages(path),
            max_new_tokens=16,
            do_sample=False,
            device=state["device"],
            dtype=state["dtype"],
        )
    finally:
        Path(path).unlink(missing_ok=True)


@app.on_event("startup")
def load_model() -> None:
    t0 = time.perf_counter()
    # 給 /health 回報：ZeroType 設定頁顯示「幾點啟動、載入幾秒」
    state["started_at"] = time.time()
    log(f"啟動中，載入模型（{ENGINE}）…")
    device = resolve_device("auto")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            dtype="auto",
            attn_implementation="sdpa",
        )
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    state.update(
        model=model,
        processor=AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True),
        device=device,
        dtype=dtype,
    )
    warmup()
    state["load_seconds"] = round(time.perf_counter() - t0, 1)
    log(f"模型就緒 {device}，耗時 {state['load_seconds']:.1f} 秒（已暖機）")


@app.get("/health")
def health() -> dict:
    ready = "model" in state
    # started_at 是 Unix 秒數；load_seconds 在模型載完之前沒有值（None）。
    return {"ok": ready, "engine": ENGINE, "device": str(state.get("device")),
            "started_at": state.get("started_at"), "load_seconds": state.get("load_seconds")}


@app.post("/shutdown")
def shutdown(background_tasks: BackgroundTasks) -> dict:
    """讓呼叫端收回 GPU 記憶體，不必知道是哪一個 pid。

    notes: 沒有任何認證。端點只綁 127.0.0.1，威脅模型裡「本機上的其他程式」
           本來就能直接殺行程，多一道認證擋不到它。改綁 0.0.0.0 之前要先補認證。
    notes: 用 os._exit 而不是讓 uvicorn 優雅收工。這個行程唯一持有的資源是
           GPU 記憶體與 port，兩者都由 OS 回收；模型沒有需要 flush 的狀態。
           延遲 0.2 秒只為了讓這個回應先送出去。
    """
    log("收到停止要求，即將結束")
    background_tasks.add_task(_exit_soon)
    return {"ok": True}


def _exit_soon() -> None:
    time.sleep(0.2)
    os._exit(0)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    response_format: str = Form(default="json"),
    prompt: str = Form(default=""),
    hotwords: str = Form(default=""),
) -> dict:
    # notes: ZeroType 送來的 prompt 是給 chat 型模型的整段指令。原封不動餵給 MOSS，
    #        它會在音訊內容偏弱時把指令本身當答案抄出來（ZeroType 自己的註解也記了
    #        同一個現象）。所以指令固定用 TRANSCRIBE_PROMPT，只把熱詞接上去。
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        audio_path = tmp.name

    try:
        t0 = time.perf_counter()
        gpu: dict = {}
        snap = threading.Thread(target=lambda: gpu.update(_gpu_snapshot()), daemon=True)
        snap.start()
        instruction = TRANSCRIBE_PROMPT
        if hotwords.strip():
            instruction = f"{instruction}熱詞提示：{hotwords.strip()}"
        result = generate_transcription(
            state["model"],
            state["processor"],
            build_transcription_messages(audio_path, prompt=instruction),
            max_new_tokens=2048,
            do_sample=False,
            device=state["device"],
            dtype=state["dtype"],
        )
        text = clean(result["text"])
        elapsed = time.perf_counter() - t0
        log(f"[{ENGINE}] {elapsed:.2f}s {len(text)} 字：{text[:60]}")
        snap.join(timeout=5)
        audio_s = _audio_seconds(audio_path)
        toks = result.get("generated_tokens", 0)
        perf_log(elapsed, len(text), audio_s, gpu, result.get("prompt_len", 0), toks, len(result["text"]))
        log(f"  GPU {gpu.get('util', '?')}% {gpu.get('used', '?')}/{gpu.get('total', '?')} MiB "
            f"{gpu.get('pstate', gpu.get('error', '?'))}，音長 {audio_s if audio_s is None else round(audio_s, 1)}s，"
            f"輸入 {result.get('prompt_len', '?')} tokens，產生 {toks} tokens"
            f"（{elapsed / toks * 1000 if toks else 0:.0f} ms/token），原始輸出 {len(result['text'])} 字")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    if response_format == "text":
        return text
    # usage 照 OpenAI 轉寫回應的欄位名稱，ZeroType 的記錄分頁會顯示 in/out tokens 並算 ms/token
    return {"text": text, "usage": {"input_tokens": result.get("prompt_len"),
                                    "output_tokens": result.get("generated_tokens")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    # 只剩 moss。ZeroType v1.0.35 起選單已無 FireRed；舊的啟動設定若仍帶 --engine firered，
    # 這裡直接報錯而不是默默換成 MOSS。
    ap.add_argument("--engine", choices=["moss"], default="moss")
    # 這個旗標在檔案最上方就已經生效，這裡只是讓 argparse 不要拒收它。
    ap.add_argument("--hide-console", action="store_true",
                    help="啟動後把主控台視窗藏起來（ZeroType 預設會帶這個旗標）")
    args = ap.parse_args()
    # notes: log_level 用 info 而不是 warning——啟動、關閉與請求錯誤都要進記錄檔，
    #        不然 ZeroType 那邊看到的只有「沒反應」。access_log 另外關掉：
    #        每一次辨識我們已經自己記一行，再加一行存取記錄只是重複。
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
