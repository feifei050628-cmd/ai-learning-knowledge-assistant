# Docker部署

Docker 可以把应用程序、运行环境和依赖打包为镜像，从而减少不同计算机之间的环境差异。

Dockerfile 描述镜像如何构建，docker build 用于构建镜像，docker run 用于根据镜像创建并运行容器。端口映射可以把容器内部的服务端口暴露给宿主机。

部署 FastAPI 服务时，需要在镜像中安装项目依赖并启动 Uvicorn。如果希望容器外部访问服务，Uvicorn 通常需要监听 0.0.0.0，同时通过 Docker 的端口映射开放对应端口。