# Timestep AI

Timestep AI CLI - free, local-first, open-source AI

**Setup**:

```console
$ cp .env.example .env # Customize as desired
$ direnv allow # See https://direnv.net/#getting-started to install direnv on your platform
$ poetry install # See https://python-poetry.org/docs/#installation to install Poetry on your platform
$ prefect server start
$ prefect worker start --pool "default"
$ timestep serve
```

**Usage**:

```console
$ timestep [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `evals`: Run evaluations.
* `serve`: Run serving.
* `train`: Run training.

## `timestep evals`

Run evaluations.

**Usage**:

```console
$ timestep evals [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `timestep serve`

Run serving.

**Usage**:

```console
$ timestep serve [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `timestep train`

Run training.

**Usage**:

```console
$ timestep train [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.
