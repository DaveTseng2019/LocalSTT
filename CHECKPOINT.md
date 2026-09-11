> 這是評估當下的工作紀錄，已過期。目前狀態以 README.md 為準。

# Checkpoint — 2026-09-11 07:0x

MOSS-Transcribe-Diarize 評估 → 接成 ZeroType 的本機語音輸入後端。

## 已完成且已驗證

- **評估**：MOSS-Transcribe-Diarize 0.9B（Apache-2.0、OpenMOSS）。權重 revision
  `704aa4a9c304e8520be88901e0d1960158ef5b15`。VRAM 峰值 1808 MiB。
- **測試集**：ZeroType 自己的 `%APPDATA%\com.zerotype\zero_type\history.json`
  ＋ `history_audio\`，55 組音檔＋文字配對（27 筆中英混用）。對照文字是 Gemini 產出、
  未人工校對，CER 只能讀成「與現行方案的差異」。
- **55 筆 × 2 組 prompt 全跑完**，0 筆執行失敗。自訂 prompt 加兩道後處理後 CER 中位數
  0.071、RTF 0.06；內建預設 prompt 0.130、RTF 0.145。
- **shim 端到端通過**：ZeroType 設定「服務商 OpenAI ＋ 自訂端點
  `http://127.0.0.1:8123/v1/audio/transcriptions` ＋ API Key 填 `local`」即可用，
  不必改 ZeroType 程式碼。`history.json` 的 `costUsd` 為空，確認沒走雲端。
- **延遲**：穩態約 500 ms。啟動後第一句原本 5.2 秒，加一秒靜音暖機後降到 600~675 ms。
- **ZeroType 已重建**（`flutter build windows --release`，43.1 秒）。`app.so` 沒變動是
  正確的——最後一次動程式碼的 commit 是 `074e428`(09-07 04:07)，之後只有文件 commit。

## 檔案

| 路徑 | 用途 |
|---|---|
| `C:\Learning\.local\mtd-eval\shim.py` | OpenAI 相容端點，含暖機、剝標記、簡轉繁 |
| `C:\Learning\.local\mtd-eval\run_test.py` | 跑 manifest 的批次測試 |
| `C:\Learning\.local\mtd-eval\report.py` | 出 MIXED-REPORT.md |
| `C:\Learning\.local\mtd-eval\MIXED-REPORT.md` | 27 筆混用並排比對 |
| `C:\Learning\.local\mtd-eval\results.json` | 110 次推論原始輸出 |
| `C:\Learning\.local\mtd-eval\repo\.venv` | Python 3.12 + torch cu129，7.6 GB |

啟動 shim：
`C:\Learning\.local\mtd-eval\repo\.venv\Scripts\python.exe C:\Learning\.local\mtd-eval\shim.py`

## 現在在跑的東西

- **shim pid 36432**（我啟動的，可自由關閉），port 8123。
- **ZeroType pid 18088**，06:51:19 啟動（我依授權關掉舊的 17828 後重開的）。
- GPU 2507 MiB / 6141 MiB。

## 未完成

1. **專有名詞是唯一明顯落後雲端的地方。** 公司名「XX科技」被聽成「OO科技」，
   同句 Gemini 正確。shim 已留 `hotwords` 表單欄位，但 ZeroType 不會送。
   要接：把 `buildDictionaryPrompt()` 的內容送進 `hotwords`，動
   `speech_recognition_service.dart` 與 `zero_type_controller.dart`，改完要重建。
2. **沒做成正式的 local provider 選項**（五處改動：`providers.json`、service 的 switch、
   兩處 apiKey gate、`isWhisper` 那一行、設定頁 UI）。陷阱是
   `zero_type_controller.dart` 的 `isWhisper = providerId == 'openai'`——決定一段式或
   兩段式，另開 `'local'` 而不改它，第二段 chat 校正會打本機端點失敗。
3. shim 沒有開機自啟，重開機後要手動啟動。
4. Claude Code 已是最新 2.1.268（今天 06:47 自己更新過），但本 session 仍用舊執行檔，
   要重開才生效。

## 未解決的疑問

休眠指令 `shutdown /h` 回傳 exit 0 但機器似乎沒睡著，沒有查證原因
（可查 Kernel-Power 事件與 `powercfg /requests` 看是否有行程阻擋）。
