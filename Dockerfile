FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PORT=8000
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mplsoccer_mcp ./mplsoccer_mcp
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
# Build the matplotlib font cache at build time so the first request is fast
RUN python -c "import matplotlib.pyplot as plt; import mplsoccer"
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"
CMD ["python", "-m", "mplsoccer_mcp"]
