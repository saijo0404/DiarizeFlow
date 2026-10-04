# [Audio/DSP]: 麥克風與系統音訊雙軌分流處理與智慧活動動態路由 (Dual-Track Decoupled Routing)

**Labels**: `audio`, `enhancement`, `architecture`

## 1. 需求背景與問題描述 (Problem Statement)
目前在 `src/diarizeflow/app/audio/capture.py` 中，當同時開啟麥克風與系統聲音（Loopback）時，處理迴圈會使用直接相加的方式進行混音：
```python
mixed = np.clip(m_chunk + l_chunk, -1.0, 1.0)
self._dispatch_chunk(mixed)
```
隨後將該單一音訊軌道傳入 `StreamingDiarizationSegmenter` 與後續的 ASR。

### 根因分析 (Root Cause Analysis)：
強行相加混音在聲學訊號處理與模型辨識上存在致命硬傷：
1. **迴音干擾與相位梳狀效應 (Acoustic Bleed & Comb Filtering)**：
   若使用者未配戴耳機（使用外放喇叭），電腦喇叭播出的聲音會在數十毫秒後被麥克風再次收錄。混音時，「原始純數位音」與「麥克風拾取的延遲迴音」相加會產生嚴重的梳狀濾波失真，導致 ASR 語音辨識率驟降（吞字、胡言亂語或幻覺）。
2. **音量能量互吃 (Volume Masking)**：
   電腦系統端（如遊戲、YouTube、會議）音量往往飽滿且為純數位訊號，而本機麥克風人聲能量較小。相加後，動態增益 (AGC) 會被大音量主導壓低增益，導致微弱的本機人聲被直接掩蓋或視為底噪丟棄。
3. **無謂增加語者分離負荷**：
   硬體層面上，麥克風（現場聲音）與系統聲音（電腦端聲音）本來就是兩個完全獨立的物理聲道。硬混在一起反而強迫 Nemotron Sortformer 去執行複雜的重疊人聲解混疊。

---

## 2. 建議解決方案 (Proposed Solution)

設計「**雙軌分流 / 智慧活動動態路由（Smart Activity Dynamic Routing）**」架構：

```
[ 麥克風 Track A ] ─── RMS/VAD 活動偵測 ───┐
                                          ├─► [ 智慧活動動態路由排程器 ] ──► [ Sortformer & ASR ]
[ 系統聲音 Track B ] ─── RMS/VAD 活動偵測 ───┘
```

### A. 智慧活動感知路由 (Smart Dynamic Activity Gating)
在大部分日常會議、遊戲與通話中，雙方通常處於「交替說話（Turn-taking）」狀態：
1. **僅 Track B（系統音訊）發言時**：
   - 僅將 Track B 的純數位音訊送入管線。
   - **完全靜音/阻斷 Track A**，杜絕環境底噪、鍵盤敲擊聲與呼吸雜音干擾。
2. **僅 Track A（本機麥克風）發言時**：
   - 僅將 Track A 的麥克風音訊送入管線。
   - 杜絕電腦背景音干擾。
3. **雙方同時搶話/重疊時（Cross-talk / Overlap）**：
   - **策略一**：利用輕量平衡混音送入 Sortformer（此時 Sortformer 專注發揮其重疊分離長處）。
   - **策略二（可選）**：雙軌各自獨立觸發發話切分與辨識，精準標記 `[現場麥克風]` 與 `[電腦端聲音]`，辨識精度達 100% 最高水準。

### B. 消除時鐘漂移與緩衝區失步
- 重構 `_processing_loop`：移除固定等待長度的阻塞邏輯，兩軌音訊使用時間戳（Timestamp）與滑動視窗對齊，避免長時間運作下的相位漂移。

---

## 3. 預期效益 (Expected Benefits)
- 徹底消除外放喇叭所產生的迴音混疊與梳狀干擾，ASR 辨識精確率顯著提升。
- 解決遊戲大音量時麥克風人聲被吞沒的問題。
- 算力開銷維持在單軌水準，兼顧效能與極致音質。
