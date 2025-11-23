---
title: Fashion Reflection Agent
emoji: 🧥
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# AI 穿搭顧問 (Fashion Reflection Agent)

本專案實作了一個基於 **Reflection Agent (反思代理)** 設計模式的衣物風格推薦系統。系統不僅能識別衣物，還能提供穿搭建議，並透過自我反思與批評機制，自動優化建議內容。

## ✨ 功能特色

1.  **衣物分類 (Classification)**：
    *   使用 `wargoninnovation/wargon-clothing-classifier` 模型。
    *   能識別 27 種不同類型的衣物 (如 T-shirt, Jeans, Dress 等)。
    *   提供詳細的分類信心分數與機率分佈圖。

2.  **風格推薦 (Recommendation)**：
    *   使用 `Qwen/Qwen2.5-3B-Instruct` 大型語言模型。
    *   根據識別結果，生成繁體中文的穿搭風格建議。

3.  **自我反思 (Reflection)**：
    *   **Draft (初稿)**：生成初步建議。
    *   **Critique (批評)**：扮演嚴格的時尚評論家，對初稿進行評分 (1-10 分) 與批評。
    *   **Refine (優化)**：根據批評意見，重新潤飾並產出最終建議。

4.  **量化指標 (Metrics)**：
    *   顯示 **初始評分** 與 **最終評分**。
    *   計算 **進步幅度 (Improvement)** 作為回歸指標，量化反思機制的效果。

5.  **互動介面**：
    *   使用 **Streamlit** 建置的網頁介面，操作直觀。
    *   支援 **圖片上傳** 與 **隨機範例** 功能。
    *   即時顯示分析步驟與進度。

6.  **紀錄 (Logging)**：
    *   所有互動過程 (User/Model) 皆會記錄於 `log.md` 檔案中。

## 🛠️ Hugging Face Spaces 部署說明

本專案已配置為可直接部署於 Hugging Face Spaces (Streamlit)。

### 部署步驟

1.  在 Hugging Face 建立一個新的 Space。
2.  選擇 **Streamlit** 作為 SDK。
3.  將本專案的所有檔案上傳至 Space 的 Repository。
4.  Space 將會自動安裝 `requirements.txt` 中的套件並啟動 `app.py`。

*注意：在 Hugging Face 的免費 CPU 環境 (Basic Hardware) 中，系統會自動切換至 CPU 模式運行。由於 Qwen-1.5B 模型較小，可以在 CPU 環境下順暢運行。*

## 💻 本地開發與執行

若您希望在本地端執行，請參考以下步驟：

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 啟動應用程式

```bash
streamlit run app.py
```

## 📂 專案結構

*   `app.py`: 主程式，包含 Streamlit 介面與 Reflection Agent 核心邏輯。
*   `model_handler.py`: 負責 AI 模型的載入、推論與錯誤處理。
*   `utils.py`: 包含 Log 紀錄與隨機圖片選取等輔助功能。
*   `simple/`: 存放範例圖片的資料夾。
*   `requirements.txt`: 專案依賴套件清單。
