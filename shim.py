"""把本機語音辨識模型包成 OpenAI 相容的轉寫端點，給 ZeroType 直接接。

ZeroType 設定：語音辨識服務商選「本機」。沒有 API Key 欄位，端點也不必填。

啟動：repo/.venv/Scripts/python.exe shim.py [--port 8123] [--host 127.0.0.1]
引擎：--engine moss（預設，MOSS-Transcribe-Diarize）或 --engine firered --firered-dir <FireRedASR2S clone>。
兩個引擎的相依套件互斥，各用各的虛擬環境啟動，見 README。
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
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


ENGINE = _arg_value("--engine", "moss")
FIRERED_DIR = Path(_arg_value("--firered-dir", str(Path(__file__).with_name("firered"))))

if ENGINE == "moss":
    from transformers import AutoModelForCausalLM, AutoProcessor

    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
        resolve_device,
    )

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

# notes: FireRedASR2-AED 一段最多 60 秒（官方 README）。短於這個長度就整段直接送、不走 VAD：
#        實測 60 筆 ZeroType 錄音，走 VAD 時每個停頓都被切成一段、各自補句號
#        （「若我想。知道這一個影片裡面的。字幕。」），整段送之後句號數中位數從 1.5 降到 1，
#        與 MOSS 去標點後的字元差異中位數從 0.036 降到 0。留 10 秒餘裕。
FIRERED_NO_VAD_MAX_S = 50

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
           FireRed 同樣成立：60 筆實測第一筆 2.9 秒，其餘中位數 0.47 秒。
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(32000))  # 一秒 16 kHz 16-bit 靜音
        if ENGINE == "firered":
            _firered_transcribe(path)
            return
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


def _load_firered() -> None:
    sys.path.insert(0, str(FIRERED_DIR))
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    # notes: fireredasr2system 在 import 時設 basicConfig(INFO)，每一句都把 VAD／ASR／Punc 的
    #        完整結果各記一行。那會把 shim.log 的每句一行洗成每句四行，ZeroType 只顯示最後 40 行。
    import logging
    logging.getLogger("fireredasr2s").setLevel(logging.WARNING)

    pm = FIRERED_DIR / "pretrained_models"
    system = FireRedAsr2System(FireRedAsr2SystemConfig(
        vad_model_dir=str(pm / "FireRedVAD" / "VAD"),
        asr_type="aed",
        asr_model_dir=str(pm / "FireRedASR2-AED"),
        punc_model_dir=str(pm / "FireRedPunc"),
        vad_config=FireRedVadConfig(use_gpu=True),
        # bf16：實測與 fp32 文字相同、快 14 倍；fp32 約 5 GB，6 GB 顯示卡會溢出到系統記憶體。
        asr_config=FireRedAsr2Config(use_gpu=True, use_half=True, return_timestamp=False),
        punc_config=FireRedPuncConfig(use_gpu=True),
        enable_lid=False,
    ))
    state.update(system=system, device=torch.device("cuda"))


def _firered_transcribe(audio_path: str) -> str:
    """轉成 16 kHz 單聲道 16-bit 再送進 FireRed。ZeroType 錄的是 44.1 kHz。"""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    duration = wav.numel() / 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path16 = tmp.name
    try:
        sf.write(path16, wav.clamp(-1, 1).numpy(), 16000, subtype="PCM_16")
        system = state["system"]
        # notes: 直接改共用的 config 旗標。端點是 async def 裡跑同步推論，請求本來就一次一個，
        #        不會有兩個請求同時讀到這個值。改成真正並行之前要換成每次請求各自的設定。
        system.config.enable_vad = duration > FIRERED_NO_VAD_MAX_S
        return system.process(path16, "zt")["text"]
    finally:
        Path(path16).unlink(missing_ok=True)


@app.on_event("startup")
def load_model() -> None:
    t0 = time.perf_counter()
    log(f"啟動中，載入模型（{ENGINE}）…")
    if ENGINE == "firered":
        _load_firered()
        warmup()
        log(f"模型就緒 firered cuda，耗時 {time.perf_counter() - t0:.1f} 秒（已暖機）")
        return
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
    log(f"模型就緒 {device}，耗時 {time.perf_counter() - t0:.1f} 秒（已暖機）")


@app.get("/health")
def health() -> dict:
    ready = "system" in state if ENGINE == "firered" else "model" in state
    return {"ok": ready, "engine": ENGINE, "device": str(state.get("device"))}


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
        if ENGINE == "firered":
            # notes: FireRedASR2-AED 沒有熱詞或 prompt 輸入，字典詞彙在這個引擎上不起作用。
            text = clean(_firered_transcribe(audio_path))
        else:
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
    finally:
        Path(audio_path).unlink(missing_ok=True)

    if response_format == "text":
        return text
    return {"text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    # 這兩個也在模組層就讀過了（決定要 import 哪個引擎），這裡只是讓 argparse 收下並列進 --help。
    ap.add_argument("--engine", choices=["moss", "firered"], default="moss")
    ap.add_argument("--firered-dir", help="FireRedASR2S 的 clone，裡面要有 pretrained_models/")
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
