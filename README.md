# 使用方式


# 构建 embedding 镜像
docker build -t embedding-api -f Dockerfile .


# 构建 rerank 镜像
docker build -t rerank-api -f Dockerfile.rerank .