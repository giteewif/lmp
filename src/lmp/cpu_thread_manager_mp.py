import os
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
import torch
import queue
from typing import Optional, Dict, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from utils.cuda_h import *
from models.mlpmodule import ExpertEinsumResult

# 在模块级别设置启动方法为 'spawn'
# 这对于 CUDA 多进程是必需的
# 注意：这必须在导入模块时设置，且只能在主进程中设置一次
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    # 已经设置过了，检查是否是 'spawn'
    current_method = mp.get_start_method(allow_none=True)
    if current_method != 'spawn':
        # 如果不是 'spawn'，尝试强制设置
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果强制设置也失败，说明已经有进程在运行
            # 这种情况下，用户需要在主脚本中手动设置
            pass


def _resolve_mp_worker_intraop_threads() -> int:
    """
    CPU experts 子进程内 ``torch.set_num_threads`` 取值（避免硬编码 64 导致 BF16 BMM 过度抢占）。

    优先级（取第一个合法正整数）：
    1. ``LMP_MP_CPU_INTRAOP_THREADS``
    2. ``OMP_NUM_THREADS``
    3. 默认 ``min(32, max(1, os.cpu_count() or 8))``
    """
    for key in ("LMP_MP_CPU_INTRAOP_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            n = int(raw, 10)
            return max(1, min(n, 512))
        except ValueError:
            continue
    try:
        c = int(os.cpu_count() or 8)
    except (TypeError, ValueError):
        c = 8
    return max(1, min(c, 32))


def _parse_cpu_id_list(raw: str) -> set[int]:
    """Parse ``0,1,2-5,7`` into a set of CPU ids."""
    out: set[int] = set()
    s = (raw or "").strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            lo = int(a.strip(), 10)
            hi = int(b.strip(), 10)
            if hi < lo:
                lo, hi = hi, lo
            for x in range(lo, hi + 1):
                out.add(int(x))
        else:
            out.add(int(p, 10))
    return out


def _default_online_cpus() -> list[int]:
    """Linux ``/sys/.../cpu/online`` if present, else ``range(os.cpu_count())``."""
    path = "/sys/devices/system/cpu/online"
    try:
        txt = open(path, "r", encoding="utf-8").read().strip()
    except OSError:
        n = int(os.cpu_count() or 1)
        return list(range(max(1, n)))

    cpus: list[int] = []
    for seg in txt.split(","):
        seg = seg.strip()
        if not seg:
            continue
        if "-" in seg:
            a, b = seg.split("-", 1)
            lo = int(a.strip(), 10)
            hi = int(b.strip(), 10)
            if hi < lo:
                lo, hi = hi, lo
            cpus.extend(range(lo, hi + 1))
        else:
            cpus.append(int(seg, 10))
    cpus = sorted(set(cpus))
    return cpus if cpus else list(range(max(1, int(os.cpu_count() or 1))))


def _discover_numa_node_cpus() -> dict[int, list[int]]:
    """
    NUMA node -> cpus from sysfs (``node*/cpulist`` or ``cpumask``), intersected with online CPUs.
    """
    base = "/sys/devices/system/node"
    online = set(_default_online_cpus())
    out: dict[int, list[int]] = {}
    try:
        entries = os.listdir(base)
    except OSError:
        return {}

    for name in entries:
        if not name.startswith("node") or name == "node":
            continue
        suffix = name[len("node") :]
        if not suffix.isdigit():
            continue
        nid = int(suffix, 10)
        cpulist_path = os.path.join(base, name, "cpulist")
        cpus: set[int] = set()
        try:
            raw = open(cpulist_path, "r", encoding="utf-8").read().strip()
            cpus |= _parse_cpu_id_list(raw)
        except OSError:
            try:
                mask_path = os.path.join(base, name, "cpumask")
                raw = open(mask_path, "r", encoding="utf-8").read().strip()
                if "," in raw or "-" in raw:
                    cpus |= _parse_cpu_id_list(raw)
            except OSError:
                cpus = set()

        if not cpus:
            continue
        if online:
            cpus &= online
        if not cpus:
            continue
        out[nid] = sorted(cpus)

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _parse_numa_node_id_list(raw: str) -> list[int]:
    out: list[int] = []
    s = (raw or "").strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p, 10))
    return out


def _affinity_mode() -> str:
    """
    ``LMP_MP_CPU_AFFINITY_MODE``: ``auto`` | ``numa`` | ``spread`` (default when unset: ``auto``).

    - ``numa``: if ``num_workers <= num_nodes``, worker ``i`` uses all CPUs of the ``i``-th node
      (after ``LMP_MP_NUMA_NODES`` filter, sorted by node id); else workers share nodes and slice CPUs.
    - ``spread``: split the global online CPU pool (or ``LMP_MP_CPU_AFFINITY_LIST``) across workers.
    - ``auto``: resolved in ``_resolve_worker_cpu_affinity`` to ``numa`` when sysfs NUMA layout exists,
      else ``spread``.
    """
    raw = os.environ.get("LMP_MP_CPU_AFFINITY_MODE", "numa").strip().lower()
    if raw in ("numa", "spread", "auto"):
        return raw
    return "auto"


def _even_slice_from_sorted_list(pool: list[int], wid: int, nw: int) -> list[int]:
    if not pool:
        return []
    n = len(pool)
    base = n // nw
    rem = n % nw
    start = wid * base + min(wid, rem)
    extra = 1 if wid < rem else 0
    end = start + base + extra
    return pool[start:end]


def _resolve_worker_cpu_affinity(worker_id: int, num_workers: int) -> Optional[set[int]]:
    """
    Per-worker CPU set for ``sched_setaffinity`` (Linux).

    - ``LMP_MP_CPU_AFFINITY=0`` / ``off`` / …: disable.
    - Unset: default **on** (same as ``1``).
    - ``LMP_MP_CPU_AFFINITY_LIST``: restrict pool for auto / numa filtering.
    - ``LMP_MP_CPU_AFFINITY_W<i>=0,1,2``: override for worker ``i`` (highest priority).
    """
    raw = os.environ.get("LMP_MP_CPU_AFFINITY", "1").strip()
    if raw in ("0", "false", "False", "no", "No", "off", "OFF"):
        return None
    if raw == "":
        enabled = True
    else:
        enabled = raw not in ("0", "false", "False", "no", "No", "off", "OFF")

    if not enabled:
        return None

    per_key = f"LMP_MP_CPU_AFFINITY_W{int(worker_id)}"
    per_raw = os.environ.get(per_key, "").strip()
    if per_raw:
        cpus = _parse_cpu_id_list(per_raw)
        return cpus or None

    pool_raw = os.environ.get("LMP_MP_CPU_AFFINITY_LIST", "").strip()
    user_pool = sorted(_parse_cpu_id_list(pool_raw)) if pool_raw else []
    online_pool = _default_online_cpus()

    nw = max(1, int(num_workers))
    wid = int(worker_id)
    if wid < 0 or wid >= nw:
        wid = wid % nw

    mode = _affinity_mode()
    numa_layout = _discover_numa_node_cpus()
    if mode == "auto":
        mode = "numa" if numa_layout else "spread"

    if mode == "numa" and numa_layout:
        nodes_filter_raw = os.environ.get("LMP_MP_NUMA_NODES", "").strip()
        if nodes_filter_raw:
            allow = set(_parse_numa_node_id_list(nodes_filter_raw))
            numa_layout = {nid: cpus for nid, cpus in numa_layout.items() if nid in allow}
        if not numa_layout:
            mode = "spread"

    if mode == "numa" and numa_layout:
        node_ids = sorted(numa_layout.keys())
        nn = len(node_ids)

        if nw <= nn:
            target_node = node_ids[wid]
            node_cpus = list(numa_layout[target_node])
            if user_pool:
                allow = set(user_pool)
                node_cpus = [c for c in node_cpus if c in allow]
            if not node_cpus:
                mode = "spread"
            else:
                return set(node_cpus)

        target_node = node_ids[wid % nn]
        node_cpus = list(numa_layout[target_node])
        if user_pool:
            allow = set(user_pool)
            node_cpus = [c for c in node_cpus if c in allow]
        if not node_cpus:
            mode = "spread"
        else:
            group_wids = [w for w in range(nw) if node_ids[w % nn] == target_node]
            if not group_wids:
                return set(node_cpus)
            pos = group_wids.index(wid)
            gw = len(group_wids)
            sub = _even_slice_from_sorted_list(node_cpus, pos, gw)
            if not sub and node_cpus:
                sub = [node_cpus[pos % len(node_cpus)]]
            return set(sub) or None

    pool = user_pool if user_pool else online_pool
    if not pool:
        return None
    return set(_even_slice_from_sorted_list(pool, wid, nw)) or None


def _apply_worker_cpu_affinity(worker_id: int, num_workers: int) -> None:
    if int(num_workers) <= 1:
        return
    cpus = _resolve_worker_cpu_affinity(worker_id=worker_id, num_workers=num_workers)
    if not cpus:
        return
    if not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, cpus)
    except Exception as e:
        try:
            from utils.logger import init_logger

            init_logger(__name__).warning(
                "CPUExperts MP worker %d/%d: sched_setaffinity failed (%s); cpus=%s",
                int(worker_id),
                int(num_workers),
                e,
                sorted(cpus),
            )
        except Exception:
            pass


def _mp_worker_diag_line(msg: str) -> None:
    """子进程诊断行：``print`` + flush；若设 ``LMP_MP_SELFTEST_DIAG=/path`` 则同时追加到文件（不依赖 shell 重定向）。"""
    print(msg, flush=True)
    path = (os.environ.get("LMP_MP_SELFTEST_DIAG") or "").strip()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


@dataclass
class CPUExpertsInput:
    layer_idx: int
    expert_idx_list: list[int]
    expert_indices_map: dict[int, tuple[int, int]]
    flat_hidden_states: torch.Tensor
    idxs: torch.Tensor
    if_decode: bool = False
    fused: bool = False
    # ``True``：``einsum_with_group_tensors_mp(..., fused=True)`` 内用 ``group_fused_experts_tensor`` 取权再 einsum；
    # ``False``：``group_experts_tensor`` + 同一套 einsum。
    use_bmm: bool = False
    # ``True``：走 ``bmm_with_group_tensors_mp_1``（融合专家 + CPU BMM）；与 ``fused`` 独立，BMM 路径固定用 ``group_fused_experts_tensor``。
    # Correlate submit/wait when multiple workers are used.
    request_id: int = -1
    device: str = "cuda:0",

def _cpu_experts_worker(
    input_queue: Queue,
    output_queue: Queue,
    model_path: str,
    model_name_type: str,
    exit_event,
    bootstrap_ready_queue: Queue,
    worker_id: int,
    num_workers: int,
):
    """
    CPU experts worker 进程主函数

    - ``use_bmm=True``：``bmm_with_group_tensors_mp_1``（融合权重 BMM）。
    - ``use_bmm=False``：``experts_func_einsum_mp`` → ``einsum_with_group_tensors_mp``，其中 ``fused`` 控制
      ``group_fused_experts_tensor`` 与 ``group_experts_tensor`` 取权。结果经 ``output_queue`` 回传 ``ExpertEinsumResult``。

    Args:
        input_queue: 输入队列（从主进程接收计算请求）
        output_queue: 输出队列（向主进程传递计算结果）
        model_path: 模型路径
        model_name_type: 模型名称类型
        exit_event: 退出事件（主进程设置此事件来通知工作进程退出）
        bootstrap_ready_queue: 各 worker 完成自检后 ``put(True)``，供主进程 ``wait_worker_bootstrap_ready`` 同步
        worker_id: 本进程 worker 下标，用于 ``sched_setaffinity`` 划分 CPU，减轻多 worker 间抢占
        num_workers: worker 总数（与 ``CPUExpertsManagerMP.num_workers`` 一致）
    """
    import sys
    import time
    import queue as queue_module
    import torch

    # 子进程侧第三方 / 项目依赖：集中在一处，便于增删改
    from utils.logger import init_logger
    from utils.cuda_h import cuda_hook_time, cuda_hook_time_end
    from sllm_store._C import allocate_cuda_memory

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    logger = init_logger(__name__)
    if num_workers > 1:
        _apply_worker_cpu_affinity(worker_id=int(worker_id), num_workers=int(num_workers))
        try:
            aff = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        except Exception:
            aff = []
        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] worker_id={int(worker_id)}/{int(num_workers)} "
            f"entered (model_path={model_path!r} type={model_name_type!r}) sched_getaffinity={aff}"
        )
        _intra = _resolve_mp_worker_intraop_threads()
        torch.set_num_threads(_intra)
        logger.debug(
            "CPU experts MP worker: torch.set_num_threads(%d) "
            "(LMP_MP_CPU_INTRAOP_THREADS / OMP_NUM_THREADS / default cap 32)",
            _intra,
        )
        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] torch.set_num_threads({_intra}) done"
        )

        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] importing MLPModuleWrapper / HostMemoryView ..."
        )

    if torch.cuda.is_available():
        # 方法1: 先检查 CUDA 是否已初始化
        # 在子进程中，CUDA 通常还没有初始化
        if not torch.cuda.is_initialized():
            # 尝试初始化 CUDA
            # 注意：如果父进程在使用 CUDA，这里可能会失败
            try:
                logger.debug(f"初始化")
                torch.cuda.init()
            except RuntimeError as e:
                logger.warning(f"Init process {os.getpid()}: torch.cuda.init() failed: {e}")
    
    
    from models.mlpmodule import MLPModuleWrapper
    from lmp.cuda_memory_view import HostMemoryView
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] imports done")

    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] MLPModuleWrapper(...) ...")
    mlpm = MLPModuleWrapper(model_name_type=model_name_type, model_path=model_path)
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] init_chmv_meta_model(...) ...")
    cm = mlpm.init_chmv_meta_model(device="cpu")
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] HostMemoryView(...) ...")
    hmv = HostMemoryView(mlpm=mlpm, empty_model=cm)
    _mp_worker_diag_line(f"[cpu_experts_worker pid={os.getpid()}] HostMemoryView ready")

    # 子进程启动自检会显著增加冷启动时间；默认跳过，仅在需要诊断时开启。
    run_selftest_raw = os.environ.get("LMP_MP_WORKER_SELFTEST", "0").strip()
    run_selftest = run_selftest_raw in ("1", "true", "True", "yes", "Yes", "on", "ON")
    if run_selftest:
        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] begin _test_group_bmm_fused_experts()"
        )
        try:
            hmv._test_group_bmm_fused_experts()
        except Exception:
            import traceback

            _mp_worker_diag_line(traceback.format_exc())
            raise
        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] end _test_group_bmm_fused_experts()"
        )
        logger.info("CPU experts worker: _test_group_bmm_fused_experts() finished.")
    else:
        _mp_worker_diag_line(
            f"[cpu_experts_worker pid={os.getpid()}] skip _test_group_bmm_fused_experts() "
            "(set LMP_MP_WORKER_SELFTEST=1 to enable)"
        )

    try:
        bootstrap_ready_queue.put(True)
    except Exception:
        pass

    group_list_list = []
    while not exit_event.is_set():
      
        input_data: CPUExpertsInput = input_queue.get()

        # 清空group_list_list, 释放unmap
        if input_data.layer_idx == -1:
            group_list_list.clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result = ExpertEinsumResult(
                final_hidden_states=None,
                time_einsum_end=0.0,
                request_id=input_data.request_id,
            )
            output_queue.put(result)
            continue

        cuda_hook_time(f"cpu_experts_worker_bmm_or_einsum_{worker_id}")

        expert_idx_list = input_data.expert_idx_list
        expert_indices_map = input_data.expert_indices_map
        idxs = input_data.idxs
        flat_hidden_states = input_data.flat_hidden_states

        if input_data.use_bmm:
            # Keep copies synchronous in worker process to avoid reading incompletely transferred CPU tensors.
            flat_hidden_states_on_cpu = flat_hidden_states.to(device="cpu", non_blocking=False)
            idxs_on_cpu = idxs.to(device="cpu", non_blocking=False)
            # _, group_list = mlpm.bmm_with_group_tensors_mp_cpu_para(
            #     hmv=hmv,
            #     layer_idx=input_data.layer_idx,
            #     expert_idx_list=expert_idx_list,
            #     expert_indices_map=expert_indices_map,
            #     flat_hidden_states_on_cpu=flat_hidden_states_on_cpu,
            #     idxs_on_cpu=idxs_on_cpu,
            #     output_queue=output_queue,
            #     request_id=input_data.request_id,
            #     device=input_data.device,
            # )
            # _, group_list = mlpm.bmm_with_group_tensors_mp(
            #     hmv=hmv,
            #     layer_idx=input_data.layer_idx,
            #     expert_idx_list=expert_idx_list,
            #     expert_indices_map=expert_indices_map,
            #     flat_hidden_states=flat_hidden_states,
            #     idxs=idxs,
            #     output_queue=output_queue,
            #     request_id=input_data.request_id,
            # )
            _, group_list = mlpm.bmm_with_group_tensors_mp_cpu_pure(
                hmv=hmv,
                layer_idx=input_data.layer_idx,
                expert_idx_list=expert_idx_list,
                expert_indices_map=expert_indices_map,
                flat_hidden_states_on_cpu=flat_hidden_states_on_cpu,
                idxs_on_cpu=idxs_on_cpu,
                output_queue=output_queue,
                request_id=input_data.request_id,
                device=input_data.device,
            )
        else:
            _, group_list = mlpm.experts_func_einsum_mp(
                hmv=hmv,
                layer_idx=input_data.layer_idx,
                expert_idx_list=expert_idx_list,
                expert_indices_map=expert_indices_map,
                flat_hidden_states=flat_hidden_states,
                idxs=idxs,
                output_queue=output_queue,
                fused=input_data.fused,
                request_id=input_data.request_id,
            )
        group_list_list.append(group_list)
        # 两条路径均会向 ``output_queue`` 放入 ``ExpertEinsumResult``；``group_list`` 供延迟 unmap。

        cuda_hook_time_end(f"cpu_experts_worker_bmm_or_einsum_{worker_id}")
        # 注意：不要 return，继续循环处理下一个任务


class CPUExpertsManagerMP:
    """
    多进程版本的 CPU Experts Manager
    使用 torch.multiprocessing 启动多个独立进程处理 CPU experts 计算
    主进程从这些进程获取计算结果
    torch.multiprocessing 支持 CUDA 张量的共享
    支持配置进程数量，多个进程共享任务队列（工作窃取模式）

    每个子进程启动时会按 ``LMP_MP_CPU_AFFINITY*`` 环境变量设置 CPU 亲和性，使各 ``num_workers``
    尽量落在不相交的 CPU 集合上，减轻彼此与主进程的算力抢占（Linux ``sched_setaffinity``，失败则忽略）。
    """
    def __init__(self, num_workers: int = 1, model_path: str = "deepseek-moe-16b-base-bfloat16", model_name_type: str = "Deepseek"):
        self.num_workers = num_workers
        self.input_queues = []
        for i in range(num_workers):
            self.input_queues.append(Queue())
        self.output_queue = Queue()
        self._bootstrap_ready_queue: Queue = Queue()
        self.exit_event = mp.Event()
        self.model_path = model_path
        self.model_name_type = model_name_type
        self.processes = []

    def submit(
        self,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        idxs: torch.Tensor,
    ):
        """
        提交任务到第一个 worker 的队列
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            idxs=idxs,
        )
        self.input_queues[0].put(input_data)
    
    def submit_worker(
        self,
        worker_idx: int,
        layer_idx: int,
        expert_idx_list: list[int],
        expert_indices_map: dict[int, tuple[int, int]],
        flat_hidden_states: torch.Tensor,
        idxs: torch.Tensor,
        fused: bool = False,
        use_bmm: bool = False,
        request_id: int | None = None,
        device: str = "cuda:0",
    ):
        """
        提交任务到指定 worker 的队列
        
        Args:
            worker_idx: worker 索引
            fused: 仅 ``use_bmm=False`` 时生效，传给 ``einsum_with_group_tensors_mp``。
            use_bmm: ``True`` 时走 ``bmm_with_group_tensors_mp_1``，否则走 ``experts_func_einsum_mp``。
        """
        input_data = CPUExpertsInput(
            layer_idx=layer_idx,
            expert_idx_list=expert_idx_list,
            expert_indices_map=expert_indices_map,
            flat_hidden_states=flat_hidden_states,
            idxs=idxs,
            fused=fused,
            use_bmm=use_bmm,
            request_id=int(request_id) if request_id is not None else -1,
            device=device,
        )
        
        if worker_idx >= len(self.input_queues):
            raise ValueError(f"Worker index {worker_idx} out of range [0, {len(self.input_queues)})")
        self.input_queues[worker_idx].put(input_data)
        return input_data

    def wait_worker_bootstrap_ready(self, timeout: float | None = None) -> None:
        """
        阻塞直到每个 worker 完成 ``HostMemoryView`` 构建与 ``_test_group_bmm_fused_experts``。

        避免主进程在 ``start()`` 后立即 ``mp_stop()`` 导致子进程尚未自检就被杀掉、重定向日志截断。

        超时秒数：参数 ``timeout``，或环境变量 ``LMP_MP_BOOTSTRAP_TIMEOUT_SEC``（默认 600），按 **每个** worker 一次 ``get`` 使用。
        """
        import queue as queue_module

        if timeout is None:
            raw = os.environ.get("LMP_MP_BOOTSTRAP_TIMEOUT_SEC", "600").strip()
            try:
                timeout = float(raw)
            except ValueError:
                timeout = 600.0
        for _ in range(self.num_workers):
            try:
                self._bootstrap_ready_queue.get(timeout=timeout)
            except queue_module.Empty as e:
                raise RuntimeError(
                    "CPU experts MP worker bootstrap timed out "
                    f"(>{timeout}s per worker). Check model/shm or set LMP_MP_BOOTSTRAP_TIMEOUT_SEC."
                ) from e

    def wait(self):
        """
        阻塞直到 worker 在 ``output_queue`` 上放入 ``ExpertEinsumResult``（由 ``bmm_with_group_tensors_mp`` 写入）。

        Returns:
            ``ExpertEinsumResult.final_hidden_states``（通常为 GPU 张量）；清空任务时为 ``None``。
        """
        result: ExpertEinsumResult = self.output_queue.get()
        return result
    
    def start(self):
        """
        启动多进程 CPU experts 管理器
        """
        # 验证启动方法是否为 'spawn'
        current_method = mp.get_start_method(allow_none=True)
        if current_method != 'spawn':
            raise RuntimeError(
                f"Multiprocessing start method must be 'spawn' for CUDA, but got '{current_method}'. "
                "Please set it before importing this module or creating CPUExpertsManagerMP instance. "
                "Add this at the beginning of your main script (before any imports):\n"
                "  import torch.multiprocessing as mp\n"
                "  mp.set_start_method('spawn')\n"
            )
        
        # 重要：在创建子进程之前，确保父进程释放所有 CUDA 资源
        import torch
        if torch.cuda.is_available():
            # 同步所有 CUDA 操作，确保没有未完成的操作
            torch.cuda.synchronize()
        
        self.processes = []
        for i in range(self.num_workers):
            process = Process(
                target=_cpu_experts_worker,
                args=(
                    self.input_queues[i],
                    self.output_queue,
                    self.model_path,
                    self.model_name_type,
                    self.exit_event,
                    self._bootstrap_ready_queue,
                    i,
                    self.num_workers,
                ),
            )
            process.start()
            self.processes.append(process)
    
    def stop(self):
        """
        停止多进程 CPU experts 管理器
        """
        # 设置退出事件，通知所有进程退出
        self.exit_event.set()
        
        # 等待所有进程结束
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                # 如果进程还在运行，强制终止
                process.terminate()
                process.join()
        
        self.processes = []