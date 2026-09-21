FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CATENARY_DATA_DIR=/data \
    PORT=8000

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷应用代码（测试不进生产镜像）
COPY app ./app

# 几何档文件目录：交给非 root 用户写
RUN mkdir -p /data && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

# 标定是 CPU 计算 + 本地文件，单进程多线程即可；线程数可用环境变量调
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads ${GUNICORN_THREADS:-4} --timeout 30 app.app:app"]
