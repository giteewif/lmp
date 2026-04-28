# ServerlessLLM（`serverless_llm`）对 vLLM 的修改说明

本文档概括 [ServerlessLLM Store](https://github.com/ServerlessLLM/ServerlessLLM) 为支持 **从 sllm-store 快速加载 / 导出权重** 而对 vLLM 做的补丁内容。官方补丁文件路径：

`sllm_store/vllm_patch/sllm_load.patch`

应用方式一般为对 vLLM 源码执行 `git apply`（版本需与补丁匹配；不同 vLLM 大版本行号可能需手工调整）。

---

## 1. 总览

| 类别 | 作用 |
|------|------|
| **加载格式** | 新增 `load_format="serverless_llm"`，由 `ServerlessLLMLoader` 从 `sllm_store` 拉权重填 vLLM 模型。 |
| **导出 API** | 在 V1 引擎链路上增加 `save_serverless_llm_state`，把当前 GPU 上的模型权重写成 sllm-store 二进制格式（按 TP rank 分目录）。 |
| **外部依赖** | Python 包 `sllm_store`（含 gRPC 客户端与 C++ 扩展）；checkpoint store 服务需按项目文档单独启动。 |

---

## 2. 模型加载：`model_loader`

### 2.1 `vllm/model_executor/model_loader/__init__.py`

- 增加导入：`ServerlessLLMLoader`。
- 在 `LoadFormats` 字面量中增加 `"serverless_llm"`。
- 在 `_LOAD_FORMAT_TO_MODEL_LOADER` 中注册：`"serverless_llm": ServerlessLLMLoader`。

### 2.2 新文件 `vllm/model_executor/model_loader/sllm_loader.py`

实现类 **`ServerlessLLMLoader`**（继承 `BaseModelLoader`）。

**`__init__`**  
- 拒绝 `model_loader_extra_config` 中非空多余键（与默认 loader 行为对齐）。

**`_filter_subtensors(state_dict)`**（静态方法）  
- 对共享同一块 storage 的张量做去重，只保留「不被其它键更大范围覆盖」的条目，避免 `save_dict` / 加载时重复序列化或歧义。

**`load_model(self, *, vllm_config, **kwargs)`**  
1. 断言 `model_config.model` 为本地目录（HF 式目录，用于读 `config.json` 等）。  
2. 取张量并行 rank：`local_model_path = join(model, "rank_{rank}")`。  
3. 用环境变量 **`STORAGE_PATH`**（默认 `~/models`）对 `local_model_path` 做前缀剥离，得到传给 store 的 **模型 ID**（相对路径 `model_path`）。  
4. 在目标 dtype 下：`initialize_model` 于 CPU 上建图 → `eval()`。  
5. 对 `named_parameters` 中需加载的键，先把 `param.data` 置为 `cuda` 上的占位小 tensor，再 `gc.collect()`。  
6. `device_map = {"": torch.cuda.current_device()}`，调用 **`sllm_store.torch.load_dict(model_path, device_map)`** 得到 `sllm_state_dict`。  
7. 将各 `param.data` 替换为 `sllm_state_dict` 中对应张量；未覆盖的键报错。  
8. 将仍在 CPU 的 **buffer** 迁到当前 GPU。  
9. 返回 `nn.Module`。

**`download_model` / `load_weights`**  
- 空实现（权重由 `load_model` 内一次性完成）。

**`save_model(model, path, pattern=None, max_size=None)`**（静态）  
- 按 TP rank 将 `state_dict` 中张量 **`.cpu().contiguous()`** 后，调用 **`sllm_store.torch.save_dict`** 写入 `join(path, f"rank_{rank}")`。

**与 store 的衔接（不在 vLLM 内硬编码）**  
- 实际 gRPC 地址由 **`sllm_store`** 实现决定。上游示例常为固定 `host:port`；在 **lmp** 维护的 `sllm_store` 中可通过 **`SLLM_STORE_ADDRESS`** 或 **`SLLM_STORE_HOST` / `SLLM_STORE_PORT`** 配置，`load_dict` 还支持显式 `store_server_address` 参数（若你在本地 `sllm_loader.py` 中传入）。

---

## 3. 权重导出：`save_serverless_llm_state`（V1 引擎）

补丁在 **Engine → Executor → Worker → ModelRunner** 上逐级转发同名方法，最终调用 `ServerlessLLMLoader.save_model`。

| 文件 | 变更要点 |
|------|----------|
| `vllm/v1/engine/core.py` | `EngineCore.save_serverless_llm_state` → `model_executor.save_serverless_llm_state(...)` |
| `vllm/v1/engine/core_client.py` | 抽象客户端默认转发；`SyncMPClient` / `AsyncMPClient` 通过 `call_utility` / `call_utility_async` 调用 `"save_serverless_llm_state"` |
| `vllm/v1/executor/abstract.py` | `Executor.save_serverless_llm_state` → `collective_rpc("save_serverless_llm_state", kwargs=...)` |
| `vllm/v1/worker/gpu_worker.py` | `Worker.save_serverless_llm_state` → `model_runner.save_serverless_llm_state(...)` |
| `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.save_serverless_llm_state` → `ServerlessLLMLoader.save_model(self.model, path, ...)` |

典型用法（概念上）：在已加载模型的 vLLM 实例上调用保存接口，将各 rank 权重写入给定 `path` 下的 `rank_*` 目录，供后续 `load_format="serverless_llm"` 加载。具体 API 名称以你所安装的 vLLM 版本为准（例如经 `LLM` / `engine` 暴露的 `save_serverless_llm_state` 或异步变体）。

---

## 4. 使用侧约定（与示例脚本一致）

- **HF 配置与 tokenizer**：仍使用本地目录 `model=` 指向的 HuggingFace 式树。  
- **sllm-store 权重文件**：位于 `STORAGE_PATH` 下的相对路径与 `model/rank_*` 对齐；`STORAGE_PATH` 与导出/注册模型时使用的根目录应一致。  
- **启动 vLLM**：`LLM(model=..., load_format="serverless_llm", ...)`（见 `sllm_store/examples/load_vllm_model.py`）。

---

## 5. 未修改的部分

补丁 **不** 替换 vLLM 的算子、调度或 KV 实现；仅在 **模型权重 I/O** 上增加一条路径。推理行为应与同架构、同权重的默认加载方式一致（在 store 与 TP 切分正确的前提下）。

---

## 6. 维护注意

- vLLM 升级后 `sllm_load.patch` 可能无法干净应用，需对照当前分支的 `EngineCore` / `gpu_model_runner` 等类名与行上下文做三向合并。  
- `sllm_store` 与 vLLM 的 **CUDA / PyTorch 版本** 需与编译扩展时一致，否则可能出现加载或 IPC 相关错误。
