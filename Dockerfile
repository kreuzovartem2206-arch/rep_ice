FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
RUN useradd --create-home monitor && mkdir /data && chown monitor:monitor /data
USER monitor
ENTRYPOINT ["ice-monitor", "--output", "/data/kara"]
CMD ["watch", "--interval-minutes", "30"]
