import torch
import threading
import threading
import time

from utils.logger import init_logger
logger = init_logger(__name__)

class PinnedMemoryPool:
    def __init__(self,
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 2, chunk_size_mb: int=1024):
        self.dtype = dtype
        self.pool_size = pool_size  # GB
        print(f"here pin")
        # 计算总字节数和元素数量
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = pool_size * 1024 * 1024 * 1024
        self.total_elements = total_bytes // self.element_size

        # 分块分配内存，每次分配1GB
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        chunk_elements = chunk_size_bytes // self.element_size

        time_start = time.time()
        self.pool_memory = []  # 改为列表存储多个内存块

        logger.info(f"Initializing PinnedMemoryPool with {pool_size}GB total, allocating in {chunk_size_mb}MB chunks...")

        remaining_elements = self.total_elements
        current_offset = 0

        while remaining_elements > 0:
            # 计算当前块的大小（最后一个块可能小于1GB）
            current_chunk_elements = min(chunk_elements, remaining_elements)

            try:
                chunk = torch.empty(current_chunk_elements, dtype=dtype, pin_memory=True, device="cpu")
                self.pool_memory.append(chunk)
                logger.debug(f"Allocated chunk {len(self.pool_memory)}: {current_chunk_elements} elements ({current_chunk_elements * self.element_size / (1024*1024):.1f} MB)")

                remaining_elements -= current_chunk_elements
                current_offset += current_chunk_elements

            except Exception as e:
                logger.error(f"Failed to allocate chunk: {e}")
                break

        total_allocated = sum(chunk.numel() for chunk in self.pool_memory)
        alloc_time = time.time() - time_start

        logger.info(f"Successfully allocated {len(self.pool_memory)} chunks, total {total_allocated} elements ({total_allocated * self.element_size / (1024*1024):.1f} MB) in {alloc_time:.3f}s")

        # 内存管理数据结构
        self.lock = threading.RLock()
        self.allocated_blocks = []  # 记录已分配的内存块 (chunk_idx, start, size)
        self.free_list = []  # 空闲块列表 [(chunk_idx, start, size), ...]

        # 初始化空闲列表
        for chunk_idx, chunk in enumerate(self.pool_memory):
            self.free_list.append((chunk_idx, 0, chunk.numel()))

    def alloc_same_pin_tensor(self, tensor: torch.Tensor):
        """分配与给定Tensor形状和类型匹配的内存块"""
        required_elements = tensor.numel()
        required_bytes = required_elements * self.element_size
        # print(tensor.element_size(), tensor.numel(), size_kb)
        block = self.alloc(required_bytes)
        return self.reshape(block, tensor.shape)
    def alloc(self, size_bytes: int):
        """分配指定大小的连续内存块"""
        if size_bytes <= 0:
            raise ValueError("分配大小必须为正数")

        # 计算需要的元素数量
        required_bytes = size_bytes
        required_elements = (required_bytes + self.element_size - 1) // self.element_size
        required_elements = int(required_elements)

        with self.lock:
            # 寻找合适的空闲块（首次适应算法）
            for i, (chunk_idx, start, size) in enumerate(self.free_list):
                if size >= required_elements:
                    # 分配这块内存
                    allocated_chunk_idx = chunk_idx
                    allocated_start = start
                    allocated_size = required_elements

                    # 更新空闲列表
                    if size == required_elements:
                        # 完全匹配，移除这个空闲块
                        self.free_list.pop(i)
                    else:
                        # 部分使用，更新空闲块起始位置和大小
                        self.free_list[i] = (chunk_idx, start + required_elements, size - required_elements)

                    # 记录已分配块
                    self.allocated_blocks.append((allocated_chunk_idx, allocated_start, allocated_size))

                    # 返回内存视图（不拷贝数据）
                    chunk = self.pool_memory[allocated_chunk_idx]
                    return chunk[allocated_start:allocated_start+allocated_size]
            logger.error(f"Memory allocation failed: ")
            logger.error(f"  Required: {required_elements} elements ({required_bytes} bytes)")
            logger.error(f"  Available free blocks: {len(self.free_list)}")
            for i, (chunk_idx, start, size) in enumerate(self.free_list):
                logger.error(f"    Block {i}: chunk={chunk_idx}, start={start}, size={size}")
            logger.error(f"  Current usage: {self.get_usage_info()}")
            raise MemoryError("内存池中没有足够的连续空间")
    
    def free(self, memory_block):
        """释放之前分配的内存块"""
        with self.lock:
            # 找到这个内存块对应的分配记录
            block_chunk_idx = None
            block_start = None
            block_size = None

            for i, (chunk_idx, start, size) in enumerate(self.allocated_blocks):
                chunk = self.pool_memory[chunk_idx]
                if (chunk[start:start+size].data_ptr() ==
                    memory_block.data_ptr()):
                    block_chunk_idx = chunk_idx
                    block_start = start
                    block_size = size
                    self.allocated_blocks.pop(i)
                    break

            if block_start is None:
                # if for test
                return
                raise ValueError("尝试释放未分配的内存块")

            # 将释放的块加入空闲列表并合并相邻空闲块
            self.free_list.append((block_chunk_idx, block_start, block_size))
            self.free_list.sort(key=lambda x: (x[0], x[1]))  # 按chunk_idx和start排序
            
            # 合并相邻空闲块
            merged_list = []
            current_chunk_idx, current_start, current_size = self.free_list[0]

            for i in range(1, len(self.free_list)):
                chunk_idx, start, size = self.free_list[i]
                # 只有在同一个chunk内的相邻块才能合并
                if chunk_idx == current_chunk_idx and current_start + current_size == start:
                    # 相邻块，合并
                    current_size += size
                else:
                    # 不同chunk或不相邻，添加当前块并开始新的块
                    merged_list.append((current_chunk_idx, current_start, current_size))
                    current_chunk_idx, current_start, current_size = chunk_idx, start, size

            merged_list.append((current_chunk_idx, current_start, current_size))
            self.free_list = merged_list
    
    def reshape(self, memory_block, new_shape):
        """将内存块reshape为新形状，不进行内存拷贝"""
        # 计算新形状需要的总元素数
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        # 检查新形状是否与原始内存块大小兼容
        if new_size != memory_block.numel():
            raise ValueError(f"新形状 {new_shape} 需要 {new_size} 个元素，但内存块有 {memory_block.numel()} 个元素")
        
        # 返回reshape后的视图（不拷贝数据）
        return memory_block.view(new_shape)
    
    def get_usage_info(self):
        """获取内存池使用情况信息"""
        with self.lock:
            total_elements = self.total_elements
            allocated_elements = sum(size for _, _, size in self.allocated_blocks)
            free_elements = sum(size for _, _, size in self.free_list)

            return {
                "total_gb": self.pool_size,
                "allocated_mb": (allocated_elements * self.element_size) / (1024 * 1024),
                "free_mb": (free_elements * self.element_size) / (1024 * 1024),
                "fragmentation": len(self.free_list)
            }


gpinpool = PinnedMemoryPool(pool_size=2, chunk_size_mb=1024)