FROM python:3.12-slim

LABEL maintainer="Quant Team"
LABEL description="A股智能投资系统"

# 设置工作目录
WORKDIR /workspace

# 配置apt使用国内镜像源（清华大学镜像）
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装TA-Lib (可选，如果需要技术指标库)
# RUN apt-get install -y ta-lib && pip install TA-Lib

# 暴露Jupyter端口
EXPOSE 8888

# 启动Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
