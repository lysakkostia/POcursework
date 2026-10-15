
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/ /build/src/

RUN g++ -std=c++20 -O3 -pthread src/main.cpp -o search_server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/search_server /app/search_server
COPY static/ /app/static/

EXPOSE 8080

CMD ["/app/search_server"]
