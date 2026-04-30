# CPU -> GPU 拷贝调度改造方案

## 目标

围绕 `sllm_store` 的内存加载链路（`client.py -> server.py -> checkpoint_store -> model`），实现以下能力：

1. CPU 到 GPU 拷贝支持高/低优先级双队列，高优先级优先。
2. 使用无锁位图记录“是否已拷贝完成”，避免重复拷贝。
3. 支持基于位图的重排序：命中重排序位图的任务，放入低优先级队列尾部（仅对低优先级生效）。
4. 兼容现有实现（不影响旧调用方）。
5. 在 `client.py` 提供相应接口。
6. 支持动态插入高优先级任务；当该任务已在低优先级队列中时，通过无锁位图去重，避免重复执行。
7. 支持批量等待“指定拷贝任务集合”完成，等待判定基于位图，`client.py` 提供对应接口。

---

## 当前实现现状（基于 `lmp/src/sllm_store`）

- 当前 GPU 拷贝入口是 `Model::ToGpu` / `Model::DispatchToGpu`（`model.cpp`）。
- 每个 device 仅有一个 `BatchQueue`（`ConcurrentQueue<GpuBatch>`），没有优先级区分。
- 去重能力仅在 host 读取阶段通过 `ConcurrentVector::keys_` 做了互斥保护去重，不是无锁位图，也不覆盖 GPU 拷贝阶段。
- `proto/storage.proto` 的 `MemCopyChunk` 只有 `src_offset/size/dst_offset/handle_idx`，没有优先级、重排序、任务 key 等字段。
- `client.py` 的 `load_into_gpu()` 只接受 `tensor_copy_chunks + cuda_memory_handles`，没有优先级/重排序参数。

结论：当前代码可以完成异步拷贝，但尚不满足本文件定义的 6 项目标，需要新增调度层。

---

## 设计总览

建议新增独立 GPU 拷贝调度器（可命名 `GpuLoader`），由 `Model::DispatchToGpu` 调用：

- **输入**：按 device 的拷贝任务列表（可混合高/低优先级）
- **调度**：每个 device 两个队列（high/low），worker 线程永远优先消费 high
- **去重**：全局无锁位图（完成位图）+ 可选的排队位图（防重复入队）
- **重排序**：仅对低优先级任务生效，命中重排序位图的任务追加到 low queue 尾部
- **兼容**：旧请求默认全部作为 low queue，行为与当前一致

---

## 核心数据结构

### 1) 任务描述

```cpp
enum class CopyPriority : uint8_t { LOW = 0, HIGH = 1 };

struct CopyTask {
  uint64_t task_id;       // 全局唯一，建议由 chunk 粒度映射
  int device_id;
  size_t chunk_id;
  size_t chunk_offset;
  size_t size;
  size_t gpu_offset;
  size_t handle_idx;
  CopyPriority priority;
};
```

`task_id` 必须稳定可重算（建议 `chunk_id + chunk_offset + size + device_id + handle_idx` 哈希），用于位图索引。

### 2) 无锁位图

```cpp
class LockFreeBitmap {
 public:
  explicit LockFreeBitmap(size_t num_bits);
  bool test(size_t bit) const;
  bool test_and_set(size_t bit);   // 原子，返回 old value
  void clear(size_t bit);
 private:
  std::vector<std::atomic<uint64_t>> words_;
};
```

- `completed_bitmap`: 拷贝完成后置位，任何重复任务直接跳过。
- `enqueued_bitmap`（可选但推荐）: 入队前 `test_and_set`，避免同一任务被重复插入队列。

位图补充语义（用于“批量等待”）：

- `completed_bitmap` 是等待接口的唯一真值来源：`task_id` 对应 bit=1 即该任务已完成。
- 支持客户端传入 `task_id` 集合，服务端循环检查对应 bit 是否全部为 1。
- 推荐增加 `epoch_id`（每次新批任务递增）避免任务 ID 复用带来的误判；等待条件可升级为 `(epoch_id, task_id)` 二元组。

### 3) 双队列

每个 device：

- `high_queue: ConcurrentQueue<CopyTask>`
- `low_queue: ConcurrentQueue<CopyTask>`

worker 伪逻辑：

1. 先尝试取 `high_queue`
2. high 为空再取 `low_queue`
3. 成功执行后设置 `completed_bitmap`

---

## 调度与执行流程

### A. 构建任务

在 `Model::DispatchToGpu` 从 `mem_copy_chunks` 映射出 chunk 后，生成 `CopyTask`，并带上 priority 信息。

### B. 入队策略

1. 若 `completed_bitmap.test(task_id)` 为 true：直接跳过（已完成）。
2. 否则尝试 `enqueued_bitmap.test_and_set(task_id)`：
   - 已置位：说明已在队列中，跳过重复入队。
   - 未置位：继续决定入哪条队列。
3. 根据 priority：
   - HIGH：入 `high_queue`。
   - LOW：若命中 `reorder_bitmap`，入 `low_queue` 尾部（默认尾插，本身即满足）；否则按常规 low 入队。

### C. Worker 执行

1. worker 优先消费 `high_queue`。
2. 拷贝前再次检查 `completed_bitmap`（并发兜底）。
3. 执行 `cudaMemcpy`。
4. 成功后置位 `completed_bitmap`，并清理 `enqueued_bitmap` 对应位（可选，便于后续复用）。

### D. 批量等待指定任务集合

服务端提供 `WaitCopyTasks`（或等价能力），输入为 `task_ids`：

1. 逐个检查 `completed_bitmap.test(task_id)`。
2. 全部为 true 则立即返回成功。
3. 若存在未完成任务，不做固定周期轮询，而是进入“事件驱动等待”，直到：
   - 全部完成，返回成功；
   - 超时，返回 `DEADLINE_EXCEEDED`；
   - 任务不存在/epoch 不匹配，返回 `INVALID_ARGUMENT`。

实现建议：

- 优先采用“位图 + 条件变量通知”混合模式：worker 完成任务后立即通知等待方，避免固定 sleep 造成的通知延迟。
- 为了避免 `notify_all` 风暴，建议引入 **按 task 分片的 waiter 索引**：`task_id -> waiter_list`（分片锁保护）。
- worker 路径应先原子置位 `completed_bitmap`，再触发对应 waiter 通知，确保等待线程醒来后可直接观察到完成位。
- 批量等待是“集合语义”而非“单任务语义”，接口必须支持一次传入多个 task。

高性能等待实现要点：

- `WaitCopyTasks` 首次检查失败后，注册 `WaitContext`（包含 `remaining_count` 与 `pending_bitset`），然后阻塞在条件变量/事件对象上。
- 每个任务完成时，仅更新关联 `WaitContext` 的计数；当 `remaining_count==0` 立即唤醒对应请求，不等待下一轮扫描。
- 超时路径返回 `pending_task_ids`，避免客户端二次全量查询。
- 建议服务端增加配置：
  - `wait_max_tasks_per_request`（防止单次等待集合过大）
  - `wait_queue_max_requests`（限制并发等待请求）
  - `wait_fast_path_only`（仅位图快路径，不注册等待，给极低时延场景使用）

延迟目标（建议）：

- 快路径（任务已完成）：`p99 < 100us`
- 通知路径（任务刚完成触发唤醒）：`p99 < 1ms`
- 不允许默认使用 1ms 级固定轮询作为主路径

---

## 重排序位图语义

- 重排序位图仅影响 **低优先级任务**。
- 命中位图的 low 任务必须延后，保证“热点任务/关键路径任务”先走。
- 高优先级任务不受重排序位图影响，始终抢占 low 队列。

---

## 与现有代码的兼容策略

### 协议兼容（`proto/storage.proto`）

为 `MemCopyChunk` 增加可选字段（旧客户端不传时走默认值）：

- `priority`（默认 LOW）
- `task_id`（可选；不传则服务端按规则生成）
- `reorder_hint`（可选；或用独立 bitmap 字段）

新增等待 RPC（推荐）：

- `WaitCopyTasksRequest { model_path, replica_uuid, repeated uint64 task_ids, uint64 timeout_ms, uint64 epoch_id(optional) }`
- `WaitCopyTasksResponse { int32 code, repeated uint64 pending_task_ids }`

注意：protobuf 新增字段不影响老字段解析，保持向后兼容。

### 服务端兼容（`server.py`）

- 旧请求：没有 priority/reorder 字段时，全部转换为 LOW。
- 新请求：按字段填充 `CopyTask`，驱动高低队列逻辑。
- 对 `task_id` 提优请求，先检查 `completed_bitmap`：已完成任务直接忽略，不重复拷贝。
- 对未实现 `WaitCopyTasks` 的旧服务端，客户端可降级为 `confirm_model_loaded`（语义更粗，不建议长期使用）。

### C++ 兼容（`checkpoint_store.cpp` / `model.cpp`）

- `Model::ToGpu` 接口可先保持原签名；在内部把旧 `MemCopyChunk` 映射为默认 LOW 任务。
- 若需要更清晰扩展，可新增重载接口，但保留旧接口不删。

---

## 服务端实现要求（保证 Python 调用 C++ 正确处理）

### 1) 分层职责与调用链

- `client.py` 只负责构造请求与重试策略，不做任务状态真值判断。
- `server.py` 负责参数校验、协议转换、错误码映射、调用 C++ 绑定层。
- `checkpoint_store_py.cpp`（pybind）负责 Python 类型到 C++ 类型的确定性转换。
- `checkpoint_store.cpp` / `model.cpp` 负责调度、位图、队列、拷贝执行与等待通知。

必须保证同一请求在这四层的语义一致，尤其是 `task_id`、`epoch_id`、`reorder_bitmap`。

### 2) Python -> C++ 参数契约（强约束）

- `task_ids`：
  - Python 侧允许 `list[int]` / `set[int]`；
  - 进入 C++ 前统一去重并按 `uint64_t` 校验范围；
  - 发现负数、溢出、空字符串等非法值立即报 `INVALID_ARGUMENT`。
- `reorder_bitmap`：
  - 若传 bitset bytes，必须校验长度与 `bitmap_capacity` 一致；
  - 若传 `task_id` 列表，服务端先转换为位图再落 C++。
- `model_path`、`replica_uuid`：
  - 为空直接拒绝；
  - `replica_uuid` 不存在返回明确错误（不可静默创建）。
- `epoch_id`：
  - 若启用 epoch，等待/提优/重排序三类请求必须使用同一 epoch 语义；
  - epoch 不匹配返回 `INVALID_ARGUMENT`，并附带当前 epoch 供诊断。

### 3) 错误处理与 gRPC 状态码映射

建议统一错误码策略，避免 Python 侧误判：

- 参数错误 -> `grpc.StatusCode.INVALID_ARGUMENT`
- replica/model 不存在 -> `grpc.StatusCode.NOT_FOUND`
- 请求超时（wait）-> `grpc.StatusCode.DEADLINE_EXCEEDED`
- 资源不足（队列满/wait 队列满）-> `grpc.StatusCode.RESOURCE_EXHAUSTED`
- 内部异常（C++ 抛错/未知错误）-> `grpc.StatusCode.INTERNAL`

要求：`server.py` 捕获 pybind 抛出的异常后，必须映射成稳定的 gRPC 状态码，并写入可读 error message（含 model/replica/epoch/task_count）。

### 4) 并发与一致性要求

- 所有“是否处理任务”的判定以 C++ 位图为准，Python 不缓存完成态。
- 高优提交流程必须原子化：
  1. 先查 `completed_bitmap`
  2. 再查/置 `enqueued_bitmap`
  3. 再执行入队或跳过
- `WaitCopyTasks` 注册等待上下文与任务完成通知必须满足 happens-before（先置 completed，再通知 waiter）。
- 禁止在 `server.py` 层做 sleep 轮询等待，等待逻辑全部下沉到 C++。

### 5) pybind 绑定要求（`checkpoint_store_py.cpp`）

- 新增绑定方法（与方案 B 对齐）：
  - `submit_high_priority_tasks(model_path, replica_uuid, task_ids, epoch_id)`
  - `set_reorder_bitmap(model_path, replica_uuid, reorder_bitmap, epoch_id)`
  - `wait_copy_tasks(model_path, replica_uuid, task_ids, timeout_ms, epoch_id)`
- 对 Python 容器的遍历顺序不做语义依赖；必要时在 C++ 侧排序/去重。
- 对返回值采用稳定结构（建议 tuple/dict 固定字段），避免 client 解析歧义。

### 6) 可观测性与排障

- 每次 RPC 打点至少包含：
  - `model_path`、`replica_uuid`、`epoch_id`
  - `task_count`、`pending_count`
  - `queue_high_len`、`queue_low_len`
  - `latency_us`、`result_code`
- 对“已完成而跳过”的任务数量单独统计（例如 `skipped_completed_count`）。
- 对 wait 超时返回 `pending_task_ids`（可截断+计数）便于 client 精确补偿。

---

## `client.py` 接口改造建议

当前接口：

```python
load_into_gpu(model_path, replica_uuid, tensor_copy_chunks, cuda_memory_handles)
```

建议保留该接口用于“首次批量提交（默认 low）”，并通过独立接口完成“任务提优”和“重排序位图设置”。

### 1) `load_into_gpu` 保持精简（不加新参数）

```python
load_into_gpu(
    model_path,
    replica_uuid,
    tensor_copy_chunks,
    cuda_memory_handles,
)
```

规则：

- `load_into_gpu` 只负责初始提交 low 任务，避免接口语义膨胀。
- 高优先级插队、重排序位图更新全部走独立接口。

### 2) 新增增量接口：按 `task_id` 提交高优先级任务

```python
submit_high_priority_copy_tasks(
    model_path,
    replica_uuid,
    task_ids,             # List[int] / Set[int]
)
```

接口语义：

- 该接口只做一件事：将 `task_ids` 对应任务加入 `high_queue`（或标记提优）。
- 服务端处理时查询位图并去重：
  - `completed_bitmap=1`：任务已完成，直接忽略；
  - `enqueued_bitmap=1` 且已在 high：忽略重复提交；
  - 任务仅在 low：提优到 high（最终仍仅执行一次）。
- 客户端无需重复上传 chunk 元数据，避免冗余与不一致。

### 3) 新增接口：设置重排序位图

```python
set_reorder_bitmap(
    model_path,
    replica_uuid,
    reorder_bitmap,       # bitset bytes 或 task_id 列表
    epoch_id=None,
)
```

接口语义：

- 仅影响 low 队列任务顺序，不改变 high-first 调度规则。
- 支持运行期更新（覆盖式或增量式，由服务端定义）。
- 建议携带 `epoch_id`，避免旧位图污染新批次任务。
- 服务端处理重排序命中任务时按位图判定：
  - 若 `completed_bitmap=1`（已拷贝完成），则不再处理该任务；
  - 若 `completed_bitmap=0`（未完成），则将该任务加入 low 队列末尾等待执行。

### 4) 新增批量等待接口：等待指定任务集合完成

```python
wait_copy_tasks(
    model_path,
    replica_uuid,
    task_ids,             # List[int] / Set[int]
    timeout_ms=30000,
    epoch_id=None,        # 可选，建议传入避免 task_id 复用歧义
) -> tuple[bool, list[int]]
```

返回语义：

- `(True, [])`：指定集合全部完成。
- `(False, pending_task_ids)`：超时或部分失败，返回尚未完成任务列表。

接口约束：

- `task_ids` 支持批量（建议上限由服务端配置，例如 4K/次）。
- 空集合直接返回成功（幂等）。
- 与 `submit_high_priority_copy_tasks` 协同：可先按 `task_id` 增量提优，再等待“关键任务子集”完成。
- 默认使用服务端阻塞等待（事件驱动通知），避免客户端轮询。

可选补充接口（性能场景推荐）：

```python
wait_copy_tasks_async(
    model_path,
    replica_uuid,
    task_ids,
    timeout_ms=30000,
    epoch_id=None,
    on_update=None,       # 可选回调：接收 pending_task_ids 变化
) -> Future
```

说明：

- `wait_copy_tasks_async` 适合批量并发等待多个任务组，避免主线程阻塞。
- 仍由服务端通知驱动，不建议在 client 侧自行 sleep+轮询。

### 5) `client.py` 推荐实现形态

```python
class SllmStoreClient:
    def load_into_gpu(...): ...

    def submit_high_priority_copy_tasks(...):
        # 仅提交 task_id 集合用于提优
        # 服务端依据位图状态去重（已完成则跳过）
        ...

    def set_reorder_bitmap(...):
        # 设置/更新低优先级重排序位图
        ...

    def wait_copy_tasks(...):
        # 调用 WaitCopyTasks RPC
        # 根据 returned pending_task_ids 判断是否全量完成
        ...
```

协议采用 **方案 B**（固定）：

- 新增 `SubmitHighPriorityTasks` RPC：按 `task_id` 提交高优先级任务。
- 新增 `SetReorderBitmap` RPC：设置/更新低优先级重排序位图。
- 新增 `WaitCopyTasks` RPC：批量等待指定任务集合完成（事件驱动通知）。

### 6) 调用示例（运行中抢占 + 重排序设置 + 批量等待）

```python
# 初次提交：默认 low 任务
client.load_into_gpu(
    model_path=model_path,
    replica_uuid=replica_uuid,
    tensor_copy_chunks=initial_chunks,
    cuda_memory_handles=handles,
)

# 运行中发现关键层需要优先恢复：增量提交 high 任务
client.submit_high_priority_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
)

# 动态设置低优先级重排序位图
client.set_reorder_bitmap(
    model_path=model_path,
    replica_uuid=replica_uuid,
    reorder_bitmap=low_priority_reorder_bitmap,
)

# 等待关键任务集合完成（不必等待整模型）
ok, pending = client.wait_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
    timeout_ms=5000,
)
if not ok:
    raise RuntimeError(f"critical copy tasks timeout, pending={pending}")
```

---

## 建议改动点（文件级）

- `csrc/sllm_store/model.cpp`
  - `DispatchToGpu`：引入任务构建、双队列入队、位图去重逻辑
  - device worker：实现 high-first 取任务策略
- `csrc/sllm_store/types_and_defs.h`
  - 扩展 `MemCopyChunk` 或新增 `CopyTask`/priority 定义
- `proto/storage.proto`
  - 扩展 `MemCopyChunk`（新增可选字段）
  - 新增提优与重排序请求字段（`task_ids` / `reorder_bitmap` / `request_mode`）或新增独立 RPC
  - 新增 `WaitCopyTasksRequest/Response`（推荐）
- `sllm_store/server.py`
  - 解析新增字段并向 C++ 层传递
  - 高优提交流程先查询 `completed_bitmap`，已完成任务直接跳过
  - 增加 `WaitCopyTasks` RPC 路由
  - 增加 waiter 注册/通知逻辑（按 task 分片，降低锁竞争）
- `csrc/sllm_store/checkpoint_store_py.cpp`
  - 新增方案 B 三个接口的 pybind 绑定与类型转换
  - 统一异常到 Python RuntimeError/ValueError，再由 `server.py` 映射 gRPC 状态码
- `csrc/sllm_store/checkpoint_store.h/.cpp`
  - 新增 `SubmitHighPriorityTasks` / `SetReorderBitmap` / `WaitCopyTasks` C++ 方法
  - 提供稳定返回结构（code + pending_task_ids）
- `sllm_store/client.py`
  - 保持 `load_into_gpu()` 精简签名
  - 新增 `submit_high_priority_copy_tasks(task_ids)` 接口
  - 新增 `set_reorder_bitmap()` 接口
  - 新增 `wait_copy_tasks()` 批量等待接口
  - 可选新增 `wait_copy_tasks_async()`，用于并发等待场景

---

## 验收标准

1. **优先级正确性**：同时有 high 和 low 任务时，high 的平均等待时延显著低于 low。
2. **去重正确性**：同一 task 重复提交（low + high）仅执行一次 `cudaMemcpy`。
3. **重排序生效**：命中 reorder 位图的 low 任务被延后，不影响 high 抢占。
4. **批量等待正确性**：`wait_copy_tasks(task_ids)` 仅在对应任务集合全部完成后返回成功。
5. **兼容性**：旧客户端（不带新字段）可无改动继续加载成功。
6. **稳定性**：高并发下无死锁、无重复完成、无任务丢失。
7. **通知时效性**：任务完成后等待方应被及时唤醒（通知路径 `p99 < 1ms`，不得依赖固定轮询）。

---

## 测试建议

- 单元测试（C++）：
  - `LockFreeBitmap` 并发 `test_and_set` 正确性
  - 双队列 high-first 调度顺序
  - low + high 重复任务去重
- 集成测试（Python + gRPC）：
  - 旧接口回归（仅 `tensor_copy_chunks`）
  - 新接口（`submit_high_priority_copy_tasks(task_ids)` + `set_reorder_bitmap()`）
  - `wait_copy_tasks`：全完成、部分完成超时、空集合、重复 task_id 场景
  - `wait_copy_tasks` 通知时效：验证完成后立即返回，避免 sleep 周期抖动
  - 多 GPU、多 handle 场景
- 性能测试：
  - 高低混合负载下的 tail latency 对比改造前后
  - 去重命中率及拷贝总字节下降比例
  - 等待路径延迟分布（快路径/通知路径），确认 `p99` 指标

---

## 实施顺序（推荐）

1. 先落 `LockFreeBitmap + high/low queue`（不改 proto，内部默认 low）。
2. 再扩 `proto/server/client`，新增“按 `task_id` 提优”和“设置重排序位图”接口，并增加 `WaitCopyTasks` 批量等待能力。
3. 最后补测试与性能回归，确保兼容旧链路。

---

## 最新代码更新概括（当前实现）

基于上述方案，`lmp/src/sllm_store` 已完成以下改造：

1. **双队列调度已接入 `Model::ToGpu`**
   - 每个 device 使用 `PriorityGpuQueue`（`high_queue_` + `low_queue_`）。
   - worker 在 `Pop()` 时始终先取 `high_queue_`，再取 `low_queue_`，实现 high-first。

2. **无锁位图去重已接入**
   - `completed_bitmap_`：任务完成后置位，重复任务直接跳过。
   - `enqueued_bitmap_`：避免重复入队。
   - `enqueued_high_bitmap_`：避免高优任务重复提交流程。

3. **动态高优提优已接入**
   - 新增 `SubmitHighPriorityTasks`（client/server/C++/pybind 全链路）。
   - 运行中可按 `task_id` 将任务增量插入 high 队列。
   - 对已完成/已提优任务通过位图去重，避免重复执行。

4. **`reorder_bitmap` 延后 + 一次性清位已接入**
   - worker 从 low 队列取任务后，若未完成且命中 `reorder_task_ids_`：
     - 该任务重新尾插回 low 队列（延后执行）；
     - 同时立即清除对应重排序位（`erase(task_id)`），保证“生效一次后重置”。
   - 后续该任务再次出队时不再因同一位反复延后。

5. **批量等待接口已接入**
   - 新增 `WaitCopyTasks`（按 `task_id` 集合等待）。
   - 使用位图 + 条件变量通知，超时返回 `pending_task_ids`。

---

## Client 端使用示例（参照 `cuda_memory_view.py` 的分配流程）

下面示例按 `@lmp/src/lmp/cuda_memory_view.py:1253-1260` 的模式：先计算 `tensor_copy_chunks` 与 `tensor_device_size`，再 `allocate_cuda_memory`，然后提交拷贝任务，并在运行中动态提优/重排序/等待关键子集。

```python
from sllm_store.client import SllmStoreClient
from sllm_store.proto import storage_pb2

# 1) 和 cuda_memory_view 一样，先得到 copy chunks + device size
tensor_meta_index, tensor_data_index, tensor_device_offsets, tensor_copy_chunks, tensor_device_size = \
    self.get_meta_data_offsets_and_copy_chunks(tensor_index_names, device_index_int)

device_memory = {device_index_int: tensor_device_size}
cuda_memory_ptrs = allocate_cuda_memory(device_memory)
cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)

client = SllmStoreClient(server_address="127.0.0.1:8073")
device_uuid = self.device_uuid_map[device_index_int]
model_path = self.mlpm.model_path
replica_uuid = "your-replica-uuid"

# 2) 可选：为关键 chunks 显式构造 task_id，并设置优先级/重排序 hint
def build_task_id(src_offset: int, size: int, dst_offset: int, device_id: int, handle_idx: int) -> int:
    seed = 0
    for v in (src_offset, size, dst_offset, device_id, handle_idx):
        seed ^= v + 0x9E3779B97F4A7C15 + ((seed << 6) & 0xFFFFFFFFFFFFFFFF) + (seed >> 2)
    return seed & 0xFFFFFFFFFFFFFFFF

patched_chunks = []
critical_task_ids = []
for src_offset, size, dst_offset, handle_idx in tensor_copy_chunks[device_index_int]:
    task_id = build_task_id(src_offset, size, dst_offset, device_index_int, handle_idx)
    is_critical = size >= (4 << 20)  # 例子：大块先提优
    if is_critical:
        critical_task_ids.append(task_id)
    patched_chunks.append(
        (
            src_offset,
            size,
            dst_offset,
            handle_idx,
            task_id,
            storage_pb2.COPY_PRIORITY_LOW,  # 初始仍走 low
            False,                          # reorder_hint (可按需设 True)
        )
    )

# 3) 首次提交（兼容旧接口；这里传扩展字段）
client.load_into_gpu(
    model_path=model_path,
    replica_uuid=replica_uuid,
    tensor_copy_chunks={device_uuid: patched_chunks},
    cuda_memory_handles={device_uuid: [cuda_memory_handles[device_index_int]]},
)

# 4) 动态设置低优先级重排序位图（命中后会“延后一次并清位”）
client.set_reorder_bitmap(
    model_path=model_path,
    replica_uuid=replica_uuid,
    reorder_bitmap=critical_task_ids[:8],   # 示例：先延后一小批 low 任务
)

# 5) 运行中把真正关键任务提到 high 队列
client.submit_high_priority_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
)

# 6) 仅等待关键任务集合，不必等整模型
ok, pending = client.wait_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
    timeout_ms=5000,
)
if not ok:
    raise RuntimeError(f"critical task timeout, pending={pending}")
```

说明：

- 如果你继续使用旧格式 chunk（4 元组：`src_offset,size,dst_offset,handle_idx`），链路仍兼容；
- 要使用提优/重排序/按任务等待能力，建议显式传 `task_id`（并在客户端与服务端保持同一计算规则）；
- `reorder_bitmap` 当前语义是 **命中一次后自动清位**，避免同一 low 任务反复延后。


import uuid
from sllm_store.client import SllmStoreClient
from sllm_store.proto import storage_pb2

# 你项目里已有这些工具函数（示例按常见签名写）
# from lmp.cuda_utils import allocate_cuda_memory, get_cuda_memory_handles, get_device_uuid_map

def build_task_id(src_offset: int, size: int, dst_offset: int, device_id: int, handle_idx: int) -> int:
    """与服务端保持稳定映射即可；不要求和服务端内部实现完全一致。"""
    seed = 0
    for v in (src_offset, size, dst_offset, device_id, handle_idx):
        seed ^= v + 0x9E3779B97F4A7C15 + ((seed << 6) & 0xFFFFFFFFFFFFFFFF) + (seed >> 2)
    return seed & 0xFFFFFFFFFFFFFFFF


# ----------------------------
# 1) 申请 GPU 内存 + 获取 handle（对应你标注的 553-555）
# ----------------------------
device_id = 0
device_memory = {device_id: 8 * 1024 * 1024 * 1024}  # 例：8GB

cuda_memory_ptrs = allocate_cuda_memory(device_memory)
cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)

# 假设拿到类似:
# cuda_memory_ptrs   = {0: <cuda_ptr>}
# cuda_memory_handles= {0: b"...cudaIpcHandle bytes..."}

device_uuid_map = get_device_uuid_map()  # 例如 {0: "GPU-xxxxxxxx-..."}
device_uuid = device_uuid_map[device_id]


# ----------------------------
# 2) 构造 copy chunks（gpuloader 任务模型）
# ----------------------------
# 原始4元组: (src_offset, size, dst_offset, handle_idx)
raw_chunks = [
    (0,        4 << 20, 0,        0),
    (4 << 20,  2 << 20, 4 << 20,  0),
    (6 << 20,  8 << 20, 6 << 20,  0),
    (14 << 20, 1 << 20, 14 << 20, 0),
]

patched_chunks = []
critical_task_ids = []     # 想优先恢复的关键任务
reorder_task_ids = []      # 想延后的一些低优任务

for src_offset, size, dst_offset, handle_idx in raw_chunks:
    task_id = build_task_id(src_offset, size, dst_offset, device_id, handle_idx)

    # 示例策略：>=4MB 作为关键任务
    is_critical = size >= (4 << 20)
    if is_critical:
        critical_task_ids.append(task_id)
    else:
        reorder_task_ids.append(task_id)

    # 7元组:
    # (src_offset, size, dst_offset, handle_idx, task_id, priority, reorder_hint)
    patched_chunks.append(
        (
            src_offset,
            size,
            dst_offset,
            handle_idx,
            task_id,
            storage_pb2.COPY_PRIORITY_LOW,  # 初始都先LOW，后续再动态提优
            False,                          # 可选 hint；也可统一走 set_reorder_bitmap
        )
    )


# ----------------------------
# 3) client 提交 + 提优 + 重排序 + 按任务等待
# ----------------------------
client = SllmStoreClient("127.0.0.1:8073")
model_path = "your_model_path"
replica_uuid = str(uuid.uuid4())

# 首次提交到GPU
ret = client.load_into_gpu(
    model_path=model_path,
    replica_uuid=replica_uuid,
    tensor_copy_chunks={
        device_uuid: patched_chunks,               # key 必须是 device_uuid
    },
    cuda_memory_handles={
        device_uuid: [cuda_memory_handles[device_id]],  # list[bytes]
    },
)
if not ret:
    raise RuntimeError("load_into_gpu failed")

# 低优重排序：命中的 low 任务会被延后（当前实现是命中一次后清位）
ok_reorder = client.set_reorder_bitmap(
    model_path=model_path,
    replica_uuid=replica_uuid,
    reorder_bitmap=reorder_task_ids[:2],  # 示例：延后2个低优任务
)
if not ok_reorder:
    raise RuntimeError("set_reorder_bitmap failed")

# 动态提优：把关键任务推到 high 队列
ok_high, pending_high = client.submit_high_priority_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
)
if not ok_high:
    # pending_high 常见含义：尚未可提优/未找到的task
    print("submit_high_priority_copy_tasks partial:", pending_high)

# 按任务等待：只等关键子集，不必等整模型
ok_wait, pending = client.wait_copy_tasks(
    model_path=model_path,
    replica_uuid=replica_uuid,
    task_ids=critical_task_ids,
    timeout_ms=5000,
)
if not ok_wait:
    raise RuntimeError(f"critical tasks timeout, pending={pending}")

print("critical tasks done")