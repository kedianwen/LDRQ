# Third-party components in the release archives

The project's own code is under the Apache License 2.0 (`LICENSE`, `NOTICE`). The release
archives also redistribute the following, each under its own licence.

| component | where in the release | licence | licence text |
|---|---|---|---|
| **Ollama** v0.34.4, linux-arm64 + jetpack5 builds, unmodified, from <https://github.com/ollama/ollama/releases/tag/v0.34.4> | `llm/ollama/` | MIT | `llm/ollama/LICENSE` |
| llama.cpp and its vendored code (inside Ollama) | `llm/ollama/lib/ollama/` | MIT and others | `llm/ollama/lib/ollama/LLAMA_CPP_LICENSE`, `LLAMA_CPP_VENDORS_LICENSE` |
| cpp-httplib, Go runtime (inside Ollama) | `llm/ollama/` | MIT; BSD-3-Clause | `llm/ollama/lib/ollama/CPP_HTTPLIB_LICENSE`, `GO_LICENSE` |
| NVIDIA CUDA runtime and cuBLAS 11 (shipped by Ollama's jetpack5 build) | `llm/ollama/lib/ollama/cuda_jetpack5/` | NVIDIA CUDA Toolkit EULA, as redistributable runtime components | <https://docs.nvidia.com/cuda/eula/> |
| **Qwen3-1.7B** (Q4_K_M, as published by Ollama as `qwen3:1.7b`), by the Qwen team, Alibaba Cloud | `llm/models/` (second archive) | Apache-2.0 | the model's `license` blob in `llm/models/blobs/` (`sha256-d18a5cc7…`) |

Only the Ollama files needed on a Jetson are included: the `cuda_v12` and `cuda_v13`
directories of the release, which are for data-centre GPUs, are left out. Nothing else
was changed.

The Unitree SDK (unitree_sdk2), ROS 2 foxy, JetPack, CUDA and TensorRT are **not**
included. They must already be installed on the robot.
