# evolutionGPT
training LLMs with evolution and that biology stuff from high school

base LLM code ~~stolen~~ referenced from [nanochat](https://github.com/karpathy/nanochat)
### Setup

evolutionGPT uses [uv](https://docs.astral.sh/uv/) for dependency management. To install:

```bash
uv sync --extra gpu    # Use for CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) Use for CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```


