# 智学 AI 学习助手 - 生产部署镜像
# 构建：docker build -t zhixue-app .
# 运行：docker run -p 8501:8501 --env-file .env -v ./data:/app/data -v ./logs:/app/logs zhixue-app

FROM python:3.11-slim

# 容器内工作目录（与下方挂载路径 /app/data、/app/logs 对应）
WORKDIR /app

# 先单独复制依赖清单：requirements.txt 未变化时跳过安装层，加速重复构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码与静态资源（assets/logo.png 运行时读取；.streamlit 配置随镜像）
COPY . .

EXPOSE 8501

# 健康检查：每 30s 用 Python 标准库探活一次首页（slim 镜像无 curl）
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4); sys.exit(0)" || exit 1

# 启动命令：0.0.0.0 监听保证容器外可访问；headless 关闭浏览器自动弹出
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
