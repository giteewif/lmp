import os
import uuid
from typing import Optional, Union


STORAGE_PATH = os.getenv("STORAGE_PATH", "/mnt/zhengcf3/models/sllm_models")
TENSOR_INDEX_RESIZE_PATH = os.getenv("TENSOR_INDEX_RESIZE_PATH", "tensor_index_resize.json")
SLLM_ADDRESS = os.getenv("SLLM_ADDRESS", "127.0.0.1:8073")

from sllm_store.client import SllmStoreClient


from utils.logger import init_logger
logger = init_logger(__name__)


def _get_uuid():
    return str(uuid.uuid4())
def load_into_gpu_async(
        client: SllmStoreClient,
        device_uuid_map: dict,
        model_path: Optional[Union[str, os.PathLike]],
        tensor_copy_chunks,
        cuda_memory_handles,
        use_fixed_gpu_ptrs=False
    ):
        logger.debug(f"get device uuid map")
        replica_uuid = _get_uuid()
        logger.debug(f"call client load into gpu")
        ret = client.load_into_gpu(
            model_path,
            replica_uuid,
            {
                device_uuid_map[device_id]: v
                for device_id, v in tensor_copy_chunks.items()
            },
            {
                device_uuid_map[device_id]: [v]
                for device_id, v in cuda_memory_handles.items()
            },
            use_fixed_gpu_ptrs=use_fixed_gpu_ptrs,
        )
        return ret, replica_uuid

def load_into_cpu(
    client: SllmStoreClient,
    model_path: str,
):
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    return ret