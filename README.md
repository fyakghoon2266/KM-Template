# LangGraph RAG Framework

LangGraph RAG Framework 是一個基於 LangGraph 構建的知識庫框架，讓使用者能夠輕鬆創建、自訂和部署 RAG（檢索-生成）應用。本框架透過 YAML 配置文件定義各組件參數，並使用 Python API 靈活構建工作流，支援多種檢索策略以及自訂節點，並可擴展到大規模應用。

---

## 目錄結構

```bash
KM-Template/
├── configs/
│   ├── components.yaml # 組件配置檔，定義檢索器與生成器參數
│   └── hr_components.yaml # HR 專用組件配置檔（針對特定業務調整）
├── core/
│   ├── init.py # 標記為 Python package
│   ├── base.py # 定義組件/節點抽象基底類別
│   ├── component_manager.py # 根據配置初始化與管理各組件
│   ├── config.py # 讀取與解析 YAML 配置，處理環境變數替換
│   ├── types.py # 定義專案使用的數據結構、常數及枚舉
│   └── workflow_builder.py # 利用 builder 模式組裝工作流，定義節點與連線
├── doc/
│   └── requirements.txt # 列出專案運行所需的第三方 Python 套件與版本
├── examples/
│   ├── compliex_workflow.py # 複雜工作流範例（展示多節點組合）
│   ├── hr_workflow.py # HR 場景工作流範例，使用 hr_components.yaml 配置
│   └── multi_collection_workflow.py # 多知識庫集合查詢工作流範例
├── llm/
│   ├── init.py # 標記為 Python package
│   └── langchain_models.py # 封裝 LangChain 相關 LLM 模型呼叫邏輯
├── nodes/
│   ├── init.py # 標記為 Python package
│   ├── custom.py # 使用者可自訂擴充的節點基底範例
│   ├── generator.py # GeneratorNode：調用生成器組件生成回答
│   ├── reranker.py # RerankerNode：對檢索結果進行再排序
│   ├── retriever.py # RetrieverNode：從向量資料庫中搜尋相關文件
│   └── router.py # RouterNode：根據條件分流至不同工作流分支
├── utils/
│   ├── init.py # 標記為 Python package
│   ├── helpers.py # 輔助函數，例如配置讀取、資料轉換等
│   └── logging.py # 全域日誌記錄設定，統一管理 log 輸出
├── vectorstores/
│   ├── init.py # 標記為 Python package
│   └── qdrant_store.py # 與 Qdrant 向量資料庫互動，實作向量索引與搜尋
├── .gitignore # Git 忽略檔，排除不需版本控制的檔案
├── README.md # 本文件：專案說明與使用指引
├── app.py # Web 服務入口，啟動 API 服務並處理 HTTP 請求
└── main.py # CLI 入口，根據命令列參數執行工作流查詢
```


---

## 詳細檔案說明

### 根目錄

- **.gitignore**  
  定義 Git 在版本控制時忽略的檔案與目錄，如編譯產物、虛擬環境、暫存檔與 log 檔。

- **README.md**  
  此文件即為專案說明，包含框架介紹、安裝步驟、範例使用方法以及詳細的目錄結構與檔案功能說明。

- **__init__.py**  
  標記此目錄為 Python 模組，通常內容簡單。

- **app.py**  
  - **功能：**  
    作為 Web 服務入口，啟動 API 服務。  
  - **主要邏輯：**  
    - 解析環境變數與讀取 YAML 配置檔。  
    - 初始化核心元件（透過 ComponentManager 建立檢索器與生成器）。  
    - 建立 Web 路由（例如使用 Flask 或 FastAPI），處理 HTTP 請求並執行工作流。  
    - 管理例外與錯誤回報，確保服務穩定。

- **main.py**  
  - **功能：**  
    為 CLI 執行檔，根據命令列參數執行工作流查詢。  
  - **主要邏輯：**  
    - 解析命令列參數（如 `--config`、`--workflow`、`--query`）。  
    - 使用 ComponentManager 初始化所需組件。  
    - 載入工作流腳本並利用 WorkflowBuilder 組裝節點與邊。  
    - 執行查詢流程並將結果輸出。

---

### configs/ 目錄

- **components.yaml**  
  定義框架中各組件的配置：  
  - **retrievers：**  
    配置預設檢索器（例如 top_k、Qdrant 連線 URL、embedding 模型等）。  
  - **generators：**  
    定義生成器參數，如 LLM 模型名稱與生成溫度。

- **hr_components.yaml**  
  提供專門針對 HR 場景的組件配置：  
  - 調整檢索器與生成器參數以適合 HR 相關查詢（如福利、政策等）。

---

### core/ 目錄

- **__init__.py**  
  標記 core 為 Python package。

- **base.py**  
  定義組件或節點的抽象基底類別，規範各子類別必須實作的方法（如 run()）。

- **component_manager.py**  
  根據配置檔初始化與管理各組件：  
  - 讀取 YAML 配置檔並處理環境變數替換。  
  - 動態實例化檢索器、生成器等，並提供 create_retriever 與 create_generator 等介面。

- **config.py**  
  提供讀取與解析配置檔的工具：  
  - 解析 YAML 檔，處理環境變數替換，並驗證配置結構正確性。

- **types.py**  
  定義專案使用的數據結構與常數，包含資料類別、枚舉等，確保型別一致性。

- **workflow_builder.py**  
  利用 builder 模式組裝工作流：  
  - 提供 add_node、set_entry_point、add_edge 等方法來構建節點連線。  
  - 在 compile 階段驗證工作流完整性，返回可執行工作流對象。

---

### doc/ 目錄

- **requirements.txt**  
  列出專案運行所需的第三方套件與版本，如 OpenAI 套件、PyYAML、Qdrant 客戶端等。

---

### examples/ 目錄

- **compliex_workflow.py**  
  複雜工作流範例（檔名可能為 complex_workflow.py）：  
  - 展示如何組合多個節點（檢索、生成、後處理等）來處理複雜查詢。

- **hr_workflow.py**  
  HR 場景工作流範例：  
  - 使用 hr_components.yaml 配置，針對 HR 查詢做特殊處理（例如預處理與後處理）。

- **multi_collection_workflow.py**  
  多知識庫集合工作流範例：  
  - 建立檢索器節點時傳入多個集合名稱，整合跨集合檢索結果後交由生成器生成回答。

---

### llm/ 目錄

- **__init__.py**  
  標記 llm 為 Python package。

- **langchain_models.py**  
  封裝 LangChain 相關 LLM 模型呼叫邏輯：  
  - 定義生成函式（例如 generate(prompt, **kwargs)）。  
  - 讀取 API 金鑰、模型部署名稱等配置，並處理 API 請求與回應解析。

---

### nodes/ 目錄

- **__init__.py**  
  標記 nodes 為 Python package。

- **custom.py**  
  自訂節點基底範例：  
  - 提供 CustomNode 基底類別，供使用者擴充自訂業務邏輯。

- **generator.py**  
  GeneratorNode 節點：  
  - 接收生成器實例，將前序節點輸入格式化為 prompt，呼叫 LLM 生成最終回答。  
  - 處理生成結果並傳遞後續。

- **reranker.py**  
  RerankerNode 節點：  
  - 接收檢索結果後依據排序策略重新排序，確保提供給生成器的資料更精準。

- **retriever.py**  
  RetrieverNode 節點：  
  - 接收查詢，利用檢索器從向量資料庫中搜尋相關文件，並整理結果（包含相似度分數）。

- **router.py**  
  RouterNode 節點：  
  - 根據查詢特徵或條件決定資料流向，實作分流與分支合併，提升工作流彈性。

---

### utils/ 目錄

- **__init__.py**  
  標記 utils 為 Python package。

- **helpers.py**  
  通用輔助函數：  
  - 包括配置檔讀取、 YAML/JSON 解析、資料轉換與字串處理等工具函式。

- **logging.py**  
  日誌記錄設定：  
  - 配置 log 格式、等級與輸出位置（終端機或檔案），並提供初始化 log handler 的函式，方便各模組統一記錄訊息與錯誤。

---

### vectorstores/ 目錄

- **__init__.py**  
  標記 vectorstores 為 Python package。

- **qdrant_store.py**  
  與 Qdrant 向量資料庫互動的介面：  
  - 定義 QdrantStore 類別，依據配置（URL、金鑰等）初始化連線。  
  - 實作文件上傳、索引建立、向量搜尋（search(vector, top_k)）等方法，並處理 API 回應與錯誤管理。

---

## 使用說明

1. **安裝依賴**  
   在 doc/requirements.txt 定義的環境中安裝相依套件：  
   ```bash
   pip install -r doc/requirements.txt

2. **設定環境變數**
    創建 .env 文件並設定必要的環境變數，如：
    ```bash
    OPENAI_API_KEY='your_openai_api_key'
    OPENAI_ENDPOINT='your_openai_endpoint'
    OPENAI_DEPLOYMENT='your_openai_deployment'
    OPENAI_EMBEDDING_DEPLOYMENT='your_embedding_deployment'
    QDRANT_URL='your_qdrant_url'
    ```

3. **運行範例工作流執行簡單 RAG 工作流：**
    ```bash
    python main.py --config configs/components.yaml --workflow examples/simple_rag_workflow.py --query "什麼是向量數據庫?"

    ```

4. **創建自定義工作流**
    參考 examples/ 中的工作流範例，建立並運行自定義工作流。
