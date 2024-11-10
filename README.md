# NewsCatch

NewsCatch是一个基于Python的新闻爬取和分析系统，支持实时新闻抓取、话题聚类和趋势分析。

## 系统要求

### 基础环境
- Python 3.8+ (推荐 3.8.10)
- pip 20.0+
- virtualenv 或 conda (推荐用于环境管理)
- Git (用于版本控制)

### 操作系统支持
- Windows 10/11 (64位)
- Ubuntu 20.04+ / CentOS 7+
- macOS 10.15+ (Intel/M1)

### 硬件要求
- CPU: 双核及以上
- RAM: 8GB+ (推荐16GB)
- 存储空间: 10GB+ 可用空间
- 网络: 稳定的互联网连接

### 浏览器驱动
- Microsoft Edge WebDriver (版本需与本地Edge浏览器匹配)
- Chrome WebDriver (可选，作为备选驱动)
- Firefox GeckoDriver (可选，作为备选驱动)

### Python包依赖

## 安装说明

1. 克隆仓库
```bash
git clone https://github.com/yourusername/newscatch.git
cd newscatch
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 下载Edge驱动
```bash
python scripts/download_edge_driver.py
```

4. 安装NLTK数据
```bash
python scripts/download_nltk_data.py
```

## 配置说明

1. 编辑配置文件
```bash
cp config.example.yml config.yml
# 修改config.yml中的配置
```

2. 设置新闻源
```python
# 在src/event_detection/crawler.py中配置新闻源
self.start_urls = [
    'https://your-news-source.com',
    ...
]
```

## 使用方法

1. 启动系统
```bash
python src/main.py
```

2. 查看结果
```bash
# 生成的新闻保存在
generated_news/

# 日志文件位于
logs/
```

## 项目结构

```
newscatch/
├── src/
│   ├── event_detection/    # 事件检测模块
│   ├── clustering/         # 话题聚类模块
│   ├── lifecycle/         # 生命周期分析
│   ├── quality/           # 质量控制
│   └── utils/             # 工具函数
├── data/                  # 数据文件
├── logs/                  # 日志文件
├── tests/                 # 测试代码
└── generated_news/        # 生成的新闻
```

## API文档

详细的API文档请参考 `docs/api.md`

## 测试

运行测试套件：
```bash
python -m pytest tests/
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证

## 作者

Your Name <your.email@example.com>

## 致谢

- 感谢所有贡献者
- 特别感谢使用的开源项目 