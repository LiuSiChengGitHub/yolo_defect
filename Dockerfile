# 选择与项目一致的 Python 版本，同时尽量保持镜像轻量。
FROM python:3.9-slim

# 让容器里的命令都在 /app 下执行，方便统一相对路径。
WORKDIR /app

# 关闭 .pyc 并让日志直接输出到终端，便于排查容器问题。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# slim 镜像很精简，OpenCV 在 Linux 运行时需要补这些系统库。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件再安装，可以最大化利用 Docker 缓存。
COPY requirements-api.txt .

# 安装 API 推理所需的最小 Python 依赖，不把训练环境一起装进镜像。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# 只复制运行 API 必需的代码和模型，避免把整个仓库都打进镜像。
COPY src/ src/
COPY api/ api/
COPY models/ models/

# 声明服务监听端口，便于运行时做端口映射。
EXPOSE 8000

# 容器内必须监听 0.0.0.0，宿主机映射端口后才能访问。
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
