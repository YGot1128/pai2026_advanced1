FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# uv（Python本体・ライブラリはすべて uv で管理する）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# MuJoCo のオフスクリーン描画(EGL / OSMesa)と動画書き出し用のシステムライブラリ
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libglvnd0 libglx0 \
        libosmesa6 \
        libglfw3 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA の EGL ベンダ定義（コンテナランタイムが注入しない環境向けの保険）
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '{\n  "file_format_version" : "1.0.0",\n  "ICD" : { "library_path" : "libEGL_nvidia.so.0" }\n}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN mkdir -p /opt/uv-python && chown vscode:vscode /opt/uv-python \
    && mkdir -p /home/vscode/.cache && chown vscode:vscode /home/vscode/.cache

USER vscode

# Python 本体をイメージに焼き込む（コンテナ再作成のたびにダウンロードしない）
RUN uv python install 3.12
