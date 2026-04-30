import asyncio
import json
import grpc
import sllm_store
from sllm_store.proto import storage_pb2, storage_pb2_grpc
from sllm_store.logger import init_logger

# this is necessary to avoid libtorch.so not found error
import torch  # noqa: F401

import ctypes
import os

_PACKAGE_DIR = sllm_store.__path__[0]


ctypes.CDLL(os.path.join(_PACKAGE_DIR, "libglog.so"))

from sllm_store._checkpoint_store import (  # noqa: E402
    CheckpointStore,
    MemCopyChunk,
)

logger = init_logger(__name__)


class StorageServicer(storage_pb2_grpc.StorageServicer):
    def __init__(
        self,
        storage_path,
        mem_pool_size,
        num_thread,
        chunk_size,
        registration_required,
        use_shared_memory=False,
        shm_name_prefix="/sllm_pinned_pool",
    ):
        if not storage_path:
            logger.error("storage_path is empty")
            raise ValueError("storage_path is empty")

        if mem_pool_size <= 0:
            logger.error("mem_pool_size must be greater than 0")
            raise ValueError("Invalid mem_pool_size")

        logger.info(
            f"StorageServicer: storage_path={storage_path}, "
            f"mem_pool_size={mem_pool_size}, num_thread={num_thread}, "
            f"chunk_size={chunk_size}, "
            f"registration_required={registration_required}"
        )

        # Backward compatibility: older/native builds expose only the
        # 4-argument constructor and do not support shared-memory options.
        try:
            self.storage = CheckpointStore(
                storage_path,
                mem_pool_size,
                num_thread,
                chunk_size,
                use_shared_memory,
                shm_name_prefix,
            )
        except TypeError:
            logger.warning(
                "CheckpointStore in current native extension does not support "
                "shared-memory constructor args; falling back to legacy mode."
            )
            self.storage = CheckpointStore(
                storage_path, mem_pool_size, num_thread, chunk_size
            )
        self.registration_required = registration_required
        self.storage_path = storage_path

    def _load_tensor_index_files(self, model_path):
        tensor_index_path = os.path.join(
            self.storage_path, model_path, "tensor_index.json"
        )
        tensor_index_resize_path = os.path.join(
            self.storage_path, model_path, "tensor_index_resize.json"
        )

        tensor_index = {}
        tensor_index_resize = {}

        if os.path.exists(tensor_index_path):
            try:
                with open(tensor_index_path, "r", encoding="utf-8") as f:
                    tensor_index = json.load(f)
            except Exception as e:
                raise ValueError(
                    f"Failed to load tensor index file: {tensor_index_path}, error: {e}"
                ) from e
        else:
            logger.warning(
                f"tensor_index.json not found for model {model_path}, fallback to full load"
            )

        # tensor_index_resize.json is optional
        if os.path.exists(tensor_index_resize_path):
            try:
                with open(tensor_index_resize_path, "r", encoding="utf-8") as f:
                    tensor_index_resize = json.load(f)
            except Exception as e:
                raise ValueError(
                    "Failed to load tensor index resize file: "
                    f"{tensor_index_resize_path}, error: {e}"
                ) from e

        return tensor_index, tensor_index_resize

    async def LoadModelAsync(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.LoadModelResponse()


        device_type = request.target_device_type
        if device_type == storage_pb2.DEVICE_TYPE_CPU:
            if not self.registration_required:
                try:
                    tensor_index, tensor_index_resize = self._load_tensor_index_files(
                        model_path
                    )
                except ValueError as e:
                    logger.error(str(e))
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return storage_pb2.LoadModelResponse()
                model_size = self.storage.register_model_info(model_path, tensor_index, tensor_index_resize)
                if model_size < 0:
                    logger.error("RegisterModel failed")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return storage_pb2.LoadModelResponse()

            ret = self.storage.load_model_from_disk_async(model_path)
        elif device_type == storage_pb2.DEVICE_TYPE_GPU:
            replica_uuid = request.replica_uuid
            if not replica_uuid:
                logger.error("replica_uuid is empty")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return storage_pb2.LoadModelResponse()

            for device_uuid, chunk_list in request.chunks.items():
                for chunk in chunk_list.chunks:
                    if chunk.task_id == 0:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details(
                            "task_id is required for GPU load chunks and must be non-zero"
                        )
                        logger.error(
                            "LoadModelAsync rejected: device_uuid=%s has chunk with task_id=0",
                            device_uuid,
                        )
                        return storage_pb2.LoadModelResponse()

            gpu_memory_handles = {
                device_uuid: [
                    handle.cuda_ipc_handle for handle in handle_list.handles
                ]
                for device_uuid, handle_list in request.handles.items()
            }

            def create_mem_copy_chunk(chunk):
                mem_copy_chunk = MemCopyChunk()
                mem_copy_chunk.src_offset = chunk.src_offset
                mem_copy_chunk.size = chunk.size
                mem_copy_chunk.dst_offset = chunk.dst_offset
                mem_copy_chunk.handle_idx = chunk.handle_idx
                mem_copy_chunk.task_id = chunk.task_id
                mem_copy_chunk.priority = (
                    1
                    if chunk.priority == storage_pb2.COPY_PRIORITY_HIGH
                    else 0
                )
                mem_copy_chunk.reorder_hint = chunk.reorder_hint
                return mem_copy_chunk

            mem_copy_chunks = {
                device_uuid: [
                    create_mem_copy_chunk(chunk) for chunk in chunk_list.chunks
                ]
                for device_uuid, chunk_list in request.chunks.items()
            }
            # logger.debug(
            #     f"LoadModelAsync: {model_path}, {replica_uuid}, "
            #     f"{gpu_memory_handles}, {mem_copy_chunks}"
            # )
            ret = self.storage.load_model_from_mem_async(
                model_path, replica_uuid, gpu_memory_handles, mem_copy_chunks
            )
        else:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.LoadModelResponse()

        if ret != 0:
            logger.error("LoadModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.LoadModelResponse()

        logger.info(
            f"LoadModel: success {model_path} with target {device_type}"
        )
        return storage_pb2.LoadModelResponse(model_path=model_path)

    async def SubmitHighPriorityTasks(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        task_ids = list(dict.fromkeys(request.task_ids))
        if not model_path or not replica_uuid:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("model_path and replica_uuid are required")
            return storage_pb2.SubmitHighPriorityTasksResponse()

        try:
            code, pending_task_ids = self.storage.submit_high_priority_tasks(
                model_path, replica_uuid, task_ids
            )
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return storage_pb2.SubmitHighPriorityTasksResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return storage_pb2.SubmitHighPriorityTasksResponse()

        if code != 0:
            context.set_code(grpc.StatusCode.NOT_FOUND)
        return storage_pb2.SubmitHighPriorityTasksResponse(
            code=code, pending_task_ids=pending_task_ids
        )

    async def SetReorderBitmap(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        if not model_path or not replica_uuid:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("model_path and replica_uuid are required")
            return storage_pb2.SetReorderBitmapResponse()

        try:
            code = self.storage.set_reorder_bitmap(
                model_path, replica_uuid, list(dict.fromkeys(request.task_ids))
            )
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return storage_pb2.SetReorderBitmapResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return storage_pb2.SetReorderBitmapResponse()

        if code != 0:
            context.set_code(grpc.StatusCode.NOT_FOUND)
        return storage_pb2.SetReorderBitmapResponse(code=code)

    async def WaitCopyTasks(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        task_ids = list(dict.fromkeys(request.task_ids))
        if not model_path or not replica_uuid:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("model_path and replica_uuid are required")
            return storage_pb2.WaitCopyTasksResponse()

        for task_id in task_ids:
            if task_id < 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("task_ids must be uint64")
                return storage_pb2.WaitCopyTasksResponse()
        timeout_ms = request.timeout_ms if request.timeout_ms > 0 else 1
        try:
            code, pending_task_ids = self.storage.wait_copy_tasks(
                model_path,
                replica_uuid,
                task_ids,
                timeout_ms,
            )
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return storage_pb2.WaitCopyTasksResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return storage_pb2.WaitCopyTasksResponse()

        if code == 1:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
        elif code != 0:
            context.set_code(grpc.StatusCode.NOT_FOUND)
        return storage_pb2.WaitCopyTasksResponse(
            code=code, pending_task_ids=pending_task_ids
        )

    async def ConfirmModel(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.ConfirmModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_GPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.ConfirmModelResponse()

        for i in range(5):
            ret = self.storage.wait_model_in_gpu(model_path, replica_uuid)
            if ret == 0:
                logger.info(
                    f"Confirm model {model_path} replica {replica_uuid} success"
                )
                return storage_pb2.ConfirmModelResponse(model_path=model_path)
            logger.info(f"Confirm model failed, retry {i + 1}")

            await asyncio.sleep(0.05)

        logger.error(
            f"Confirm model {model_path} replica {replica_uuid} failed"
        )
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ConfirmModelResponse()

    async def UnloadModel(self, request, context):
        model_path = request.model_path
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.UnloadModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_CPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.UnloadModelResponse()

        for i in range(5):
            ret = self.storage.unload_model_from_host(model_path)
            if ret == 0:
                logger.info(f"UnloadModel: success {model_path}")
                return storage_pb2.UnloadModelResponse(model_path=model_path)
            logger.info(f"UnloadModel failed, retry {i + 1}")

            await asyncio.sleep(0.01)

        logger.error(f"UnloadModel failed for model {model_path}")
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.UnloadModelResponse()

    async def ClearMem(self, request, context):
        ret = self.storage.clear_mem()
        if ret != 0:
            logger.error("ClearMem failed")
            context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ClearMemResponse()

    async def RegisterModel(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.RegisterModelResponse()

        try:
            tensor_index, tensor_index_resize = self._load_tensor_index_files(
                model_path
            )
        except ValueError as e:
            logger.error(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.RegisterModelResponse()

        model_size = self.storage.register_model_info(
            model_path, tensor_index, tensor_index_resize
        )
        if model_size < 0:
            logger.error("RegisterModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.RegisterModelResponse()

        return storage_pb2.RegisterModelResponse(
            model_path=model_path, model_size=model_size
        )

    async def GetServerConfig(self, request, context):
        return storage_pb2.GetServerConfigResponse(
            mem_pool_size=self.storage.get_mem_pool_size(),
            chunk_size=self.storage.get_chunk_size(),
        )

    async def GetModelSharedMemoryNames(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.GetModelSharedMemoryNamesResponse()

        try:
            shm_names, chunk_size = self.storage.get_model_shared_memory_names(
                model_path
            )
        except Exception as e:
            logger.error(
                f"GetModelSharedMemoryNames failed for model {model_path}: {e}"
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.GetModelSharedMemoryNamesResponse()

        if not shm_names:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(
                "No shared memory names available; verify shared-memory mode and model state."
            )
            return storage_pb2.GetModelSharedMemoryNamesResponse()

        return storage_pb2.GetModelSharedMemoryNamesResponse(
            model_path=model_path,
            shm_names=shm_names,
            chunk_size=chunk_size,
        )


async def serve(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    registration_required,
    use_shared_memory=False,
    shm_name_prefix="/sllm_pinned_pool",
):
    server = grpc.aio.server()
    storage_pb2_grpc.add_StorageServicer_to_server(
        StorageServicer(
            storage_path,
            mem_pool_size,
            num_thread,
            chunk_size,
            registration_required,
            use_shared_memory,
            shm_name_prefix,
        ),
        server,
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gRPC server")
        await server.stop(5)
