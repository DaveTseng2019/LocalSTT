# LocalSTT

本機語音辨識端點。把本機語音辨識模型包成 OpenAI 相容的轉寫 API，
給 ZeroType 當語音輸入後端。所有推論都在本機 GPU 上執行。不上雲，不計費。

有兩個引擎，用 `--engine` 選。兩者的相依套件互斥，所以各用各的虛擬環境啟動：

| 引擎 | 模型 | 虛擬環境 |
|---|---|---|
| `moss`（預設） | [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) 0.9B（Apache-2.0），權重 revision `704aa4a9c304e8520be88901e0d1960158ef5b15` | `repo/.venv`（ZeroType 預設找這個） |
| `firered` | [FireRedTeam/FireRedASR2-AED](https://github.com/FireRedTeam/FireRedASR2S)（Apache-2.0）＋ FireRedVAD ＋ FireRedPunc | 另外建，見「FireRed 引擎」 |

## 需求

- NVIDIA GPU，2 GB 以上可用 VRAM（實測峰值 1808 MiB）。
- Python 3.12。虛擬環境用 uv 建立。
- torch 2.8.0+cu129、transformers 5.17.0、fastapi、uvicorn、opencc。

## 跟 ZeroType 一起用

ZeroType 會自己管理這個端點的生死。在模型設定頁選「本機」服務商之後，那一區有：

| 控制項 | 作用 |
|---|---|
| 狀態列 | 顯示端點是「執行中」或「未啟動」，旁邊有重新檢查 |
| 啟動／停止 | 手動開關。停止會收回約 1.8 GB 顯示記憶體 |
| 隨 ZeroType 一起啟動 | 預設開啟。只有服務商選「本機」時才會啟動；關閉 ZeroType 時一併關掉 |
| 端點記錄 | 展開後顯示最後 40 行，每兩秒自動更新 |

切換服務商或切換模型都會重新探測一次端點狀態。

ZeroType 不會殺掉「不是它啟動的」shim——手動跑 `start.ps1` 起來的那個，
按「停止」時改走 `/shutdown` 端點，由 shim 自己了結。

**視窗永遠隱藏，訊息永遠看得到。** shim 一律用 `pythonw.exe` 啟動，不開主控台；
`sys.stdout` 與 `sys.stderr` 在 import torch 之前就被接管，**所有**輸出都寫進 `shim.log`：
自訂訊息、uvicorn 的 `INFO`、函式庫的 DeprecationWarning、未捕捉例外的 traceback 都在內。
ZeroType 直接顯示這個檔——不必重啟、隨時看得到，而且連不是 ZeroType 啟動的 shim 也讀得到。

不提供「顯示主控台視窗」開關：Windows 的主控台是行程建立當下配置的，外部無法對執行中的
行程加掛或拆除，那種開關只能在下次啟動時生效，不如直接把訊息送進 ZeroType。

進度條（transformers 載權重）用 `\r` 原地重畫幾十次。`_Tee` 按行收斂，每行只留最後一次
重畫的結果，否則檔案裡會出現一條幾百字的長行，把真正要看的訊息擠出畫面。

**強制關閉 ZeroType（工作管理員、`Stop-Process`）時 shim 會存活。**
`quitApp()` 只在正常結束時執行。下次啟動的健康檢查會偵測到它而沿用，不會重複啟動。

## 手動啟動

```powershell
C:\Learning\LocalSTT\start.ps1
```

預設監聽 `127.0.0.1:8123`。要換 port 就加參數：`.\start.ps1 --port 8124`。
換了 port 之後 ZeroType 的自動啟動與狀態偵測就失效（它只認 8123），要自己啟動。

第一次啟動要載入模型並暖機，約 10 秒。看到 `模型就緒 ...` 才算可用。

- `GET /health` — 是否已載入完成
- `POST /shutdown` — 自行結束，收回 GPU 記憶體
- `shim.log` — 每一句的耗時與結果，每次啟動清空

## 開機自啟（不搭配 ZeroType 時）

```powershell
C:\Learning\LocalSTT\install-autostart.ps1
```

註冊工作排程項目，登入後延遲 30 秒以隱藏視窗啟動。不需要管理員權限。
移除：`install-autostart.ps1 -Uninstall`。

**跟 ZeroType 搭配使用時不需要這個。** 兩邊都開的話 ZeroType 會偵測到端點已在執行而不重複啟動，
但按「停止」時它殺不掉排程啟動的行程，只能靠 `/shutdown`。

## 實測結果

測試集是 ZeroType 自己的 55 組音檔與文字配對，其中 27 筆中英混用。
對照文字由 Gemini 產出，未經人工校對。所以 CER 只能讀成「與現行雲端方案的差異」，
不是絕對錯誤率。

| 項目 | 自訂 prompt ＋ 後處理 | 內建預設 prompt |
|---|---|---|
| CER 中位數 | 0.071 | 0.130 |
| RTF | 0.06 | 0.145 |

55 筆 × 2 組 prompt 全部跑完，0 筆執行失敗。

延遲：穩態約 500 ms。啟動後第一句原本要 5.2 秒，加入一秒靜音暖機後降到 600~675 ms。

**已知落差：專有名詞。** 公司名「XX科技」被聽成「OO科技」，
同一句 Gemini 正確。這是目前唯一明顯輸給雲端的地方。

## 檔案

| 路徑 | 用途 |
|---|---|
| `shim.py` | OpenAI 相容端點。含兩個引擎、暖機、剝時間戳標記、簡轉繁 |
| `firered_bias.py` | FireRed 引擎的熱詞加權 |
| `start.ps1` | 啟動腳本 |
| `shim.log` | 端點記錄，ZeroType 讀這個檔顯示。每次啟動清空 |
| `install-autostart.ps1` | 註冊／移除登入自啟的工作排程項目 |
| `repo/` | 上游 clone ＋ `.venv`（7.7 GB）。不進版控 |
| `eval/` | 評估用的測試集、批次腳本與原始輸出。內容是維護者本人的錄音逐字稿，不進版控 |
| `CHECKPOINT.md` | 評估過程的工作紀錄 |

## 重建虛擬環境

`repo/` 不在版控裡。要重建的話：

```powershell
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git repo
cd repo
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install -e .
uv pip install fastapi uvicorn python-multipart opencc
```

首次執行 `shim.py` 會自動從 Hugging Face 下載權重。

## FireRed 引擎

### 建立環境

FireRedASR2S 官方只在 Ubuntu 測過，但在 Windows 可以跑（2026-10-21 實測）。
官方 `requirements.txt` 釘的是 torch 2.1.0+cu118，這裡改用 cu126 的新版 torch：

```powershell
git clone --depth 1 https://github.com/FireRedTeam/FireRedASR2S.git <FireRed資料夾>\repo
uv venv --python 3.10 <FireRed資料夾>\.venv
$py = "<FireRed資料夾>\.venv\Scripts\python.exe"
uv pip install --python $py torch torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $py "transformers==4.51.3" "numpy<2" cn2an kaldiio kaldi_native_fbank sentencepiece soundfile textgrid "peft>=0.13.2" huggingface_hub fastapi uvicorn python-multipart opencc
foreach ($m in "FireRedASR2-AED","FireRedVAD","FireRedPunc") { & "<FireRed資料夾>\.venv\Scripts\hf.exe" download "FireRedTeam/$m" --local-dir "<FireRed資料夾>\repo\pretrained_models\$m" }
```

**還要修一行上游程式碼。** `fireredasr2s/fireredasr2/models/fireredasr_aed.py` 的
`forced_align` 不收 bf16。`use_half=True` 時每一句都在那裡丟例外，再被裸 `except` 吞掉，
時間戳靜默變成 `None`。把 `n_ctc_logits.unsqueeze(0)` 改成 `n_ctc_logits.float().unsqueeze(0)`。
shim 本身不用時間戳（`return_timestamp=False`），所以不修也能辨識；要拿這份 clone 做字幕時才需要。

### 跟 ZeroType 一起用

在 ZeroType 的「啟動設定（進階）」填：

| 欄位 | 值 |
|---|---|
| 程式 | `<FireRed資料夾>\.venv\Scripts\python.exe` |
| 參數 | `<LocalSTT資料夾>\shim.py --engine firered --firered-dir <FireRed資料夾>\repo` |
| 記錄檔 | 留空（預設就是 `shim.py` 旁邊的 `shim.log`） |

存檔後在 ZeroType 按「停止」再「啟動」。`shim.log` 出現 `模型就緒 firered cuda` 就是換好了。
要換回 MOSS，三個欄位清空即可。

### 跟 MOSS 的差別

2026-10-21 用 60 筆近期 ZeroType 錄音實測。**原本那 55 筆的音檔已被 ZeroType 清掉**，
所以對照文字是 MOSS 當時的輸出，不是人工校對。下面的差異只能說明「兩者不同」，不能說明誰比較準。

| 項目 | 結果 |
|---|---|
| 去標點後與 MOSS 完全相同 | 31 / 60 筆 |
| 去標點後字元差異中位數 | 0 |
| 延遲中位數 | 0.47 秒（第一句 2.9 秒，所以要暖機） |
| VRAM 峰值 | 約 2.7 GB（MOSS 約 1.8 GB） |

從差異逐筆看到的傾向（判讀，沒有音檔可對）：

- FireRed 對中文詞較準：有幾筆會計與專案用語 MOSS 寫錯，FireRed 寫對。
- FireRed 對英文詞較差：英文產品名、系統名、英文人名大多寫錯，常被音譯成中文。
- FireRed 把數字寫成國字（例如五位數的編號被逐字寫成國字）。
- FireRed 沒有文字輸入，熱詞改在解碼時加權，見下一節。

**短句不走 VAD。** 走 VAD 時每個停頓都被切成一段、各自補句號（「若我想。知道這一個影片裡面的。字幕。」）。
50 秒以下的錄音整段直接送，句號數中位數從 1.5 降到 1，與 MOSS 的字元差異中位數從 0.036 降到 0。
超過 50 秒才走 VAD，因為 AED 一段最多 60 秒。

## 熱詞怎麼送

ZeroType 的字典詞彙以 `hotwords` 表單欄位送進來，shim 再接到轉寫指令後面。

送出的是純詞彙列表（`欣興電子、臺積電`），**不是** ZeroType `buildDictionaryPrompt()`
產生的那段指令文。理由跟 `shim.py` 註解記的是同一件事：MOSS 在音訊內容偏弱時，
會把 prompt 裡的指令本身當答案抄出來。詞彙列表沒有指令語氣，抄出來的風險低得多。

同樣的理由，shim 刻意完全忽略 `prompt` 欄位。

### FireRed 的熱詞加權

FireRedASR2-AED 只吃聲音，沒有地方放熱詞。`firered_bias.py` 改在 beam search 每一步，
替「正在拼出字典詞」的 token 加分；拼到一半斷掉就扣回，拼完整個詞才保留。
它在執行時替換上游的 `batch_beam_search`，不必改 FireRedASR2S 的 clone。
英文熱詞的大小寫也照字典寫回（FireRed 的輸出全是小寫）。

權重用 `--hotword-weight` 調，預設 3.0，0 是關閉。2026-10-21 用 100 筆 ZeroType 錄音實測：
60 筆 MOSS 文字含字典詞、40 筆不含。字典 13 個詞。

**熱詞要先轉簡體。** FireRed 的字表只有簡體字，繁體的「時」「燈」「號」「組」「織」「圖」都不在裡面。
`firered_bias.py` 會先用 OpenCC `t2s` 轉換再切 token，否則含這些字的詞全部被略過。
下表權重 0～6 那幾列是修正前量的（13 詞只用上 10 詞）；修正後權重 3 重量一次：13 詞全用上，
命中仍是 35 / 63，多出的一筆「誤加」是 MOSS 寫「登號」、FireRed 寫「燈號」，判讀為 FireRed 正確。

| 權重 | 字典詞命中 | 誤加（人工判讀） | 不含字典詞的 40 筆 |
|---|---|---|---|
| 0 | 11 / 63 | 0 | 與 MOSS 差異中位數 0 |
| 3 | 35 / 63 | 0 | 0 |
| 4.5 | 44 / 63 | 約 3 | 0 |
| 6 | 51 / 63 | 大量亂插 | 0.400 |

「誤加」欄是逐筆看過的：自動統計在權重 3 時算出 2 筆，但兩筆都是 MOSS 當時聽錯、
FireRed 加權後才對（例如 MOSS 寫「潘」，實際講的是 PANE）。

**每個詞在一條候選句裡只加分一次。** 沒有這條限制時，權重 4.5 起模型會無限重複字典詞
（「Session Session Session…」）來賺加分。代價是同一句講兩次同一個詞，第二次不加分。

## 待辦

- 熱詞對專有名詞的實際改善幅度**尚未量測**。程式路徑已通，但沒有重跑 55 筆測試集
  比較「有熱詞／無熱詞」的 CER。要量的話跑 `eval/run_test.py`。
