# 🧩 模型訓練工具開發手冊（Desktop Trainer Tool）

## 📘 專案簡介
本工具為一款桌上型應用程式，協助使用者將影像與標註資料整理為標準訓練格式，並自動生成對應的 `data.yaml` 設定檔，以供 YOLO / GroundingDINO / Ultralytics 等模型使用。

功能涵蓋：
1. 將資料自動整理成訓練結構（train / valid / test）
2. 生成標準化 YAML 設定檔
3. 支援 Dry-Run 模式（預覽將執行的操作）
4. 可後續擴充至自動啟動 Docker 訓練與 ONNX 匯出

---

## 🧱 系統架構概覽

```
trainer_desktop/
├─ app.py                      # 主程式（Tkinter）
├─ ui/
│  ├─ data_prep_page.py        # 資料整理介面
│  ├─ yaml_gen_page.py         # YAML 生成介面
│  └─ widgets.py               # 共用元件
├─ core/
│  ├─ data_prep.py             # 資料整理邏輯
│  ├─ yaml_gen.py              # YAML 產生模組
│  ├─ env_check.py             # 環境檢查
│  └─ utils.py                 # 通用工具
├─ templates/
│  └─ data_yaml.j2             # YAML Jinja2 模板
├─ runs/
│  └─ 2025-10-16_08-30-12/
│     ├─ train.txt
│     ├─ val.txt
│     ├─ test.txt
│     ├─ classes.txt
│     ├─ data.yaml
│     └─ summary.json
└─ requirements.txt
```

---

## 🧮 功能一：3.1 將資料整理符合訓練格式

### 📥 輸入資料格式

```
img/
├─ 0001.jpg
├─ 0001.txt
├─ 0002.jpg
├─ 0002.txt
└─ classes.txt
```

- `xxx.jpg`：影像檔  
- `xxx.txt`：標註檔（YOLO 格式：`class x y w h`，皆為 0~1 正規化座標）  
- `classes.txt`：每行一個類別名稱  
  ```
  DR02
  Hole
  ```

---

### 📤 輸出資料結構

```
dataset_root/
├─ train/
│  ├─ images/
│  │  ├─ xxx.jpg
│  └─ labels/
│     ├─ xxx.txt
├─ valid/
│  ├─ images/
│  └─ labels/
├─ test/
│  ├─ images/
│  └─ labels/
├─ classes.txt
├─ train.txt
├─ val.txt
├─ test.txt
└─ data.yaml
```

---

### ⚙️ 功能設計與流程

| 步驟 | 說明 |
|------|------|
| **1. 讀取 classes.txt** | 解析類別名稱、建立 name→id 對應；檢查重複與空白行 |
| **2. 掃描 img 資料夾** | 找出 `.jpg` 檔及其對應 `.txt`；若標註檔不存在，可選擇是否允許空標註 |
| **3. 標註格式驗證** | 檢查每行是否為 5 欄、座標是否在 [0,1]；類別編號是否合法 |
| **4. 資料分割** | 依照使用者輸入比例（train/valid/test）與固定 seed 進行隨機切分 |
| **5. 檔案複製與清單生成** | 將檔案複製至對應目錄，並生成 `train.txt / val.txt / test.txt`（相對路徑） |
| **6. 輸出統計報告** | 產生 `summary.json`，包含影像數量、標註數量、空標註比例、類別分布等資訊 |

---

### 📄 範例輸出檔案內容

#### `classes.txt`
```
DR02
Hole
```

#### `train.txt`
```
./train/images/000c729f-df1b-4ed5-aa84-3890ffc46864_DR02.jpg
./train/images/00172e49-c985-4d66-953f-9919fcb31227_DR02.jpg
...
```

#### `summary.json`
```json
{
  "total_images": 1000,
  "train": 700,
  "valid": 200,
  "test": 100,
  "empty_labels": 12,
  "class_distribution": {
    "DR02": 512,
    "Hole": 476
  }
}
```

---

### 🔍 Dry-Run 模式

- **用途**：模擬執行，不真的複製檔案。
- **顯示內容**：
  - 將生成的目錄結構預覽
  - 三個分割集的影像數量
  - 部分樣本路徑清單
- **用途場景**：
  - 檢查比例設定是否合理
  - 驗證路徑與檔名無誤
  - 安全預覽前處理結果

---

## 🧾 功能二：3.2 生成對應 YAML 檔

### 📘 YAML 產出規格

```yaml
path: /ultralytics/data/20241206_K2_DR02_v18.1  # dataset root dir
train: train.txt  # training list
val: val.txt      # validation list
test: test.txt    # test list

names:
  0: DR02
  1: Hole

stuff_names: [
  'DR02',
  'other',
  'unlabeled'
]

download: |
  from utils.general import download, Path
```

---

### ⚙️ 自動生成邏輯

| 項目 | 說明 |
|------|------|
| `path` | 使用者設定的資料集根目錄（絕對路徑） |
| `train/val/test` | 對應的清單檔案路徑 |
| `names` | 從 `classes.txt` 自動解析 |
| `stuff_names` | 可選，UI 勾選是否加入 |
| `download` | 保留模板占位區塊，可不填 |

---

### 🧩 Jinja2 模板（`templates/data_yaml.j2`）

```jinja2
path: "{{ path }}"
train: "train.txt"
val: "val.txt"
test: "test.txt"

names:
{% for i, name in enumerate(class_names) -%}
  {{ i }}: {{ name }}
{% endfor -%}
```

---

## 💻 介面設計（Tkinter）

### 📂 資料整理頁（Data Prep）

| 元件 | 功能 |
|------|------|
| 路徑輸入 | 選擇輸入資料夾（img/）與輸出根目錄 |
| 分割合計 | train / valid / test 比例 + seed |
| 選項 | 允許空標註、格式驗證開關 |
| 按鈕 | `[Dry-Run 預覽]`、`[開始整理]` |
| 結果顯示 | 顯示統計與 summary.json 連結 |

### ⚙️ YAML 生成頁（YAML Generator）

| 元件 | 功能 |
|------|------|
| 顯示類別表 | 讀取 classes.txt，預覽類別與索引 |
| YAML 預覽區 | 以文字區塊顯示 Jinja2 渲染結果 |
| 按鈕 | `[預覽 YAML]`、`[寫入 data.yaml]` |

---

## 🧠 例外與錯誤處理設計

| 狀況 | 處理方式 |
|------|-----------|
| `classes.txt` 不存在 | 錯誤提示「找不到類別對應檔」 |
| 有影像無標註 | 依 UI 設定：允許則生成空檔，不允許則跳過 |
| 標註格式錯誤 | 記錄於 summary.json 中；嚴格模式下中止 |
| 檔名重複 | 自動在檔名加 `_dup` 後綴，並提示 |
| 相對路徑錯誤 | 自動修正為 `./train/images/...` 形式 |
| YAML 生成失敗 | 檢查 classes 內容或權限問題並提示 |

---

## 🧩 Dry-Run 預覽範例

```
✅ [Dry-Run 模擬結果]
--------------------------------------
📂 將建立資料夾結構：
  - train/images (700 張)
  - valid/images (200 張)
  - test/images (100 張)
--------------------------------------
🧾 將生成：
  - train.txt / val.txt / test.txt
  - classes.txt (2 類)
  - data.yaml
--------------------------------------
🧩 範例影像：
  ./train/images/0001.jpg
  ./valid/images/0143.jpg
  ./test/images/0866.jpg
--------------------------------------
🚫 尚未複製檔案（Dry-Run 模式）
```

---

## 🧰 開發重點與擴充性

| 模組 | 可擴充方向 |
|------|-------------|
| `core/data_prep.py` | 支援 COCO → YOLO / VOC → YOLO 轉換 |
| `core/yaml_gen.py` | 多框架模板（GroundingDINO、YOLOv5、YOLOv8） |
| `core/env_check.py` | GPU / CUDA / Docker 檢查 |
| `ui/` | 改為 PyQt 或 Electron 前端 |
| `runs/` | 可與 MLflow 或 SQLite 整合做紀錄追蹤 |

---

## 🧾 推薦開發順序
1. 完成 `data_prep.py`（含 Dry-Run 與 summary.json 輸出）  
2. 完成 `yaml_gen.py`（用 Jinja2 模板生成 data.yaml）  
3. 串接 Tkinter 介面（兩頁）  
4. 增加「歷史紀錄」功能，將每次運行參數寫入 `runs/`  
5. 後續整合 Docker 訓練與 ONNX 匯出  

---

## ✅ 總結

此工具將：
- **統一資料輸入格式**
- **自動生成訓練 YAML**
- **支援 Dry-Run 預覽、安全測試**
- **完整記錄每次處理結果**
  
能顯著減少人工整理錯誤與訓練前準備時間，適合企業內部資料標準化與模型快速實驗。
