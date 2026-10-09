# ---------- Tahap build: kompilasi backend (butuh cgo untuk onnxruntime_go) ----------
FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY *.go ./
RUN CGO_ENABLED=1 go build -o /out/spinecheck .

# ---------- Tahap jalan: Linux x64 + ONNX Runtime versi yang sama dengan onnxruntime_go ----------
FROM debian:bookworm-slim
ARG ORT_VERSION=1.29.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | tar -xz -C /opt \
 && ln -s "/opt/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so" /usr/local/lib/libonnxruntime.so \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /out/spinecheck ./
COPY models/ ./models/

# GOMEMLIMIT membuat Go lebih rajin membebaskan memori; aman untuk instance 512 MB.
ENV ONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so \
    MODELS_DIR=/app/models \
    GOMEMLIMIT=300MiB

# Render mengisi variabel PORT sendiri; backend membacanya saat start.
CMD ["./spinecheck"]