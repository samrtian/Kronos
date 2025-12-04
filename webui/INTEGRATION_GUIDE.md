# Kronos WebUI 整合说明文档

## 📋 整合概述

本次整合将 `examples/stock_prediction_interactive.py` (交互式CLI) 和 `webui/app.py` (Web界面) 的优势功能合并，创建了一个功能更强大的增强版WebUI。

## ✨ 新增功能

### 1. **股票代码直接查询** (akshare集成)
- ✅ 支持通过股票代码直接获取数据
- ✅ 无需预先准备CSV文件
- ✅ 自动从akshare下载最新A股数据
- ✅ 支持的股票代码格式：000001, 600000, 002594等

### 2. **涨跌幅限制**
- ✅ 符合中国A股±10%涨跌幅规则
- ✅ 可自定义限制比例(1%-20%)
- ✅ 可选择性启用/禁用
- ✅ 适用于不同市场的监管要求

### 3. **双数据源支持**
- ✅ **CSV文件模式**: 上传本地K线数据文件
- ✅ **股票代码模式**: 输入股票代码实时获取
- ✅ 一键切换，灵活选择

### 4. **增强的预测参数**
- ✅ Temperature控制 (预测随机性)
- ✅ Top-p核采样 (预测多样性)
- ✅ Sample count (样本数量)
- ✅ Price limits (涨跌幅限制)
- ✅ 自定义历史窗口和预测长度

## 🚀 使用方法

### 安装依赖

```bash
cd webui
pip install -r requirements.txt
```

新增依赖：
- `akshare` - 股票数据获取
- `matplotlib==3.9.3` - 图表生成

### 启动Web界面

```bash
cd webui
python run.py
```

或直接：
```bash
python app.py
```

访问: `http://localhost:7070`

## 📝 操作流程

### 方式1: 使用CSV文件

1. **选择数据源**: 保持默认"CSV File"
2. **选择文件**: 从下拉列表选择数据文件
3. **加载数据**: 点击"📁 Load Data"
4. **加载模型**: 选择Kronos模型，点击"🔄 Load Model"
5. **配置参数**:
   - 设置预测参数 (Temperature, Top-p, Sample count)
   - 可选：勾选"Apply Price Limits"并设置比例
6. **开始预测**: 点击"🔮 Start Prediction"

### 方式2: 使用股票代码

1. **选择数据源**: 切换到"Stock Code (akshare)"
2. **输入股票代码**: 例如 `000001`, `600000`, `002594`
3. **加载数据**: 点击"📁 Load Data"
   - 系统会自动从akshare下载该股票的历史数据
4. **加载模型**: 选择Kronos模型，点击"🔄 Load Model"
5. **配置参数**:
   - 设置预测参数
   - **建议勾选"Apply Price Limits"** (A股有涨跌幅限制)
   - 默认10%，符合A股规则
6. **开始预测**: 点击"🔮 Start Prediction"

## 🔧 API端点说明

### 新增端点

#### 1. `/api/akshare-status` (GET)
检查akshare库是否可用

**响应示例:**
```json
{
  "available": true,
  "message": "akshare available, stock code query enabled"
}
```

#### 2. `/api/load-stock` (POST)
通过股票代码加载数据

**请求参数:**
```json
{
  "stock_code": "000001"
}
```

**响应示例:**
```json
{
  "success": true,
  "data_info": {
    "rows": 450,
    "columns": ["timestamps", "open", "high", "low", "close", "volume", "amount"],
    "start_date": "2023-01-01T00:00:00",
    "end_date": "2024-12-04T00:00:00",
    "price_range": {
      "min": 10.23,
      "max": 15.67
    },
    "timeframe": "1 days",
    "stock_code": "000001",
    "data_source": "akshare"
  },
  "message": "Successfully loaded stock 000001, total 450 records"
}
```

### 更新端点

#### `/api/predict` (POST)
预测功能增强

**新增参数:**
```json
{
  "stock_code": "000001",           // 可选：股票代码 (与file_path二选一)
  "file_path": "/path/to/data.csv", // 可选：文件路径 (与stock_code二选一)
  "apply_price_limits": true,       // 新增：是否应用涨跌幅限制
  "limit_rate": 0.1,                // 新增：涨跌幅比例 (0.1 = 10%)
  "lookback": 400,
  "pred_len": 120,
  "temperature": 1.0,
  "top_p": 0.9,
  "sample_count": 1
}
```

## 📊 功能对比

| 功能 | 原CLI版本 | 原WebUI版本 | 整合后WebUI |
|------|-----------|-------------|-------------|
| akshare股票查询 | ✅ | ❌ | ✅ |
| CSV文件上传 | ❌ | ✅ | ✅ |
| 涨跌幅限制 | ✅ | ❌ | ✅ |
| Web界面 | ❌ | ✅ | ✅ |
| 批量预测 | ✅ | ❌ | 🔄 (计划中) |
| 可交互图表 | ❌ | ✅ | ✅ |
| 实时对比分析 | ❌ | ✅ | ✅ |
| 多模型支持 | ❌ | ✅ | ✅ |
| 自定义时间窗口 | ❌ | ✅ | ✅ |

## ⚙️ 配置说明

### 涨跌幅限制配置

不同市场的涨跌幅规则：

| 市场 | 限制比例 | 配置值 |
|------|----------|--------|
| 中国A股 (主板) | ±10% | 10 |
| 中国A股 (创业板/科创板) | ±20% | 20 |
| 香港股市 | 无限制 | 不勾选 |
| 美国股市 | 无限制 | 不勾选 |

### akshare数据说明

- **数据来源**: 中国A股市场
- **数据类型**: 日线K线数据 (OHLCV)
- **数据范围**: 默认获取最近500个交易日
- **更新频率**: 实时获取最新数据
- **支持股票**: 沪深A股所有股票

## 🐛 故障排除

### 1. akshare不可用
**问题**: 页面显示"akshare not available"

**解决方案**:
```bash
pip install akshare
```

### 2. 股票代码无效
**问题**: 提示"Unable to fetch data for stock"

**解决方案**:
- 确认股票代码格式正确 (6位数字)
- 检查股票是否存在
- 尝试添加市场前缀 (如sh600000, sz000001)

### 3. 涨跌幅限制异常
**问题**: 预测结果不符合涨跌幅规则

**解决方案**:
- 确认已勾选"Apply Price Limits"
- 检查limit_rate设置是否正确
- 查看后端日志确认限制是否应用

## 🔄 未来计划

- [ ] 批量股票预测功能
- [ ] 股票列表TXT上传支持
- [ ] 预测结果CSV/Excel导出
- [ ] 实时股价推送
- [ ] 更多技术指标支持
- [ ] 多时间周期支持 (分钟、小时、日线)

## 📚 参考资料

- **akshare文档**: https://akshare.akfamily.xyz/
- **Kronos模型**: https://huggingface.co/NeoQuasar
- **原CLI代码**: `examples/stock_prediction_interactive.py`
- **原WebUI代码**: `webui/app.py`

## 👨‍💻 技术栈

- **后端**: Flask, pandas, numpy, akshare, matplotlib
- **前端**: HTML5, JavaScript, Plotly.js, Axios
- **AI模型**: Kronos (PyTorch)
- **数据源**: akshare, CSV文件

## 📄 许可证

与Kronos项目相同的MIT许可证

---

**更新日期**: 2025-12-04
**版本**: v1.0 (Enhanced Integration)
