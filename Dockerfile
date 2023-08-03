# FROM python:3.11 as requirements-stage

# RUN groupadd -r base && useradd --no-log-init -r -g base base
# USER base
# WORKDIR /home/base

# RUN python3 -m pip install --user pipx
# RUN python3 -m pipx ensurepath
# ENV PATH=/home/base/.local/bin:$PATH

# RUN pipx install poetry==1.5.1
# COPY --chown=base:base ./pyproject.toml ./poetry.lock* /home/base/
# RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# RUN pip install --no-cache-dir --upgrade -r /home/base/requirements.txt
# COPY ./app /home/base/app

# # CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

# FROM osrf/ros:humble-simulation-jammy
# FROM gazebo:gzserver8
# FROM ros:iron-ros-core-jammy
# FROM ubuntu:22.04

# RUN apt-get update && apt-get install -y \
#     locales \
#     && rm -rf /var/lib/apt/lists/* \
# 	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# ENV LANG en_US.utf8
# ENV TZ=America/Los_Angeles

# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     direnv \
#     git \
#     libbz2-dev \
#     libffi-dev \
#     liblzma-dev \
#     libncursesw5-dev \
#     libnss3-tools \
#     libreadline-dev \
#     libsqlite3-dev \
#     libssl-dev \
#     libxml2-dev \
#     libxmlsec1-dev \
#     swig \
#     tk-dev \
#     xz-utils \
#     zlib1g-dev \
#     && rm -rf /var/lib/apt/lists/*

# RUN curl -sLS https://get.arkade.dev | sh

# RUN groupadd -r base && useradd --no-log-init -r -g base -s /bin/bash base
# USER base
# WORKDIR /home/base

# RUN git clone https://github.com/anyenv/anyenv /home/base/.anyenv
# ENV PATH=/home/base/.anyenv/bin:/home/base/.arkade/bin:/home/base/.local/bin:$PATH

# RUN echo 'eval "$(anyenv init -)"' >> /home/base/.bashrc

# RUN anyenv install --force-init
# RUN anyenv install nodenv
# RUN anyenv install pyenv
# RUN ark get caddy

# ENV NODENV_VERSION=18.15.0
# RUN ["/bin/bash", "-c", "eval \"$(anyenv init -)\" && nodenv install $NODENV_VERSION"]

# ENV PYENV_VERSION=3.11.3
# RUN ["/bin/bash", "-c", "eval \"$(anyenv init -)\" && pyenv install $PYENV_VERSION"]
# RUN ["/bin/bash", "-c", "eval \"$(anyenv init -)\" && python3 -m pip install --user pipx && python3 -m pipx ensurepath"]
# RUN pipx install poetry

# ENV PROJECT_NAME=www

# COPY --chown=base:base $PROJECT_NAME /home/base/$PROJECT_NAME
# WORKDIR /home/base/$PROJECT_NAME
# RUN ["/bin/bash", "-c", "eval \"$(anyenv init -)\" && npm install && npm run build"]

# # COPY --chown=base:base pyproject.toml poetry.lock* /home/base/
# # RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
# # RUN poetry install --no-root

# # SHELL [ "/bin/bash", "-c" ]

# # RUN python3 -m pip install --no-cache-dir --upgrade -r /home/base/requirements.txt

# WORKDIR /home/base
# COPY --chown=base:base Caddyfile /home/base/Caddyfile

# CMD [ "tail", "-f", "/dev/null" ]
# # CMD [ "caddy", "file-server", "--browse" ]

# # # EXPOSE 8080

# FROM python:3.11.3 as requirements-stage

# WORKDIR /tmp

# RUN pip install poetry==1.5.1

# COPY ./pyproject.toml ./poetry.lock* /tmp/

# RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# FROM python:3.11.3

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libnss3-tools \
#     && rm -rf /var/lib/apt/lists/*

# RUN curl -sLS https://get.arkade.dev | sh
# RUN ark system install node --version v18.15.0

# RUN groupadd -r base && useradd --no-log-init -r -g base -s /bin/bash base
# USER base
# WORKDIR /home/base

# ENV PATH=/home/base/.arkade/bin:$PATH
# RUN ark get caddy

# COPY --chown=base:base www /home/base/www
# WORKDIR /home/base/www
# RUN npm install && npm run build

# WORKDIR /home/base
# COPY --from=requirements-stage /tmp/requirements.txt /home/base/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /home/base/requirements.txt

# COPY ./app /home/base/app

# COPY --chown=base:base Caddyfile /home/base/Caddyfile

# CMD [ "tail", "-f", "/dev/null" ]

# FROM caddy:2.6.4-builder as builder

# RUN xcaddy build \
#     --with github.com/greenpau/caddy-security

# # RUN curl -sLS https://get.arkade.dev | sh
# # ENV PATH=/home/base/.arkade/bin:$PATH
# # RUN ark system install node --version v18.15.0

# # COPY www /tmp/www
# # WORKDIR /tmp/www
# # RUN npm install && npm run build

# FROM caddy:2.6.4

# COPY --from=builder /usr/bin/caddy /usr/bin/caddy
# # COPY --from=builder /tmp/www/dist /srv
# COPY Caddyfile /etc/caddy/Caddyfile

# RUN groupadd -r base && useradd --no-log-init -r -g base -s /bin/bash base
# USER base
# WORKDIR /home/base

# RUN curl -sLS https://get.arkade.dev | sh
# ENV PATH=/home/base/.arkade/bin:$PATH
# RUN ark system install node --version v18.15.0

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    locales \
    && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8
ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    direnv \
    git \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libnss3-tools \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    sudo \
    swig \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sLS https://get.arkade.dev | sudo sh
RUN ark system install node --version v18.15.0

RUN groupadd -r base && useradd --no-log-init -r -g base -s /bin/bash base
USER base
WORKDIR /home/base

RUN ark get caddy
# RUN ark system install node --version v18.15.0

COPY --chown=base:base www /home/base/www
WORKDIR /home/base/www
RUN npm install && npm run build

WORKDIR /home/base
COPY --chown=base:base Caddyfile /home/base/Caddyfile

# SHELL [ "/bin/bash", "-c" ]

# RUN caddy version
ENV PATH=/home/base/.arkade/bin:$PATH
RUN caddy version

# # CMD [ "tail", "-f", "/dev/null" ]
# CMD ["/home/base/.arkade/bin/caddy" "run" "--config" "/home/base/Caddyfile" "--adapter" "caddyfile"]
CMD ["caddy", "run", "--config", "/home/base/Caddyfile", "--adapter", "caddyfile"]
