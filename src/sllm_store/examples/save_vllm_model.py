import argparse
import os
import shutil
from typing import Optional


class VllmModelDownloader:
    def __init__(self):
        pass

    def download_vllm_model(
        self,
        model_name: str,
        torch_dtype: str,
        tensor_parallel_size: int = 1,
        cpu_offload_gb: float = 0,
        gpu_memory_utilization: float = 0.6,
        storage_path: str = os.path.expanduser("~/models"),
        local_model_path: Optional[str] = None,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ):
        import gc
        import sys
        from tempfile import TemporaryDirectory

        import torch
        from huggingface_hub import snapshot_download
        # vLLM relies on Transformers' AutoConfig mapping for `model_type`.
        # For local/custom checkpoints (e.g., gemma4) we register a minimal
        # config shim so Transformers can parse `config.json` and vLLM can read
        # fields from `config.text_config` (as attributes).
        try:
            from transformers import AutoConfig  # type: ignore
            from transformers.models.auto.configuration_auto import (  # type: ignore
                CONFIG_MAPPING,
            )
            if "gemma4" not in CONFIG_MAPPING:
                # If a local shim package is installed, importing it will
                # register `gemma4` into AutoConfig.
                import transformers.models.gemma4.configuration_gemma4  # noqa: F401
        except Exception:
            # Best-effort: if registration fails, downstream will raise a
            # clearer error when trying to load the checkpoint.
            pass

        # Must be set before importing vLLM / torch.cuda initialization.
        # Default to all 4 GPUs so --tensor-parallel-size 4 can work.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

        from vllm import LLM

        # set the model storage path
        storage_path = os.getenv(
            "STORAGE_PATH", storage_path
        ) or os.path.expanduser("~/models")

        def _run_writer(input_dir, model_name):
            # load models from the input directory
            llm_writer = LLM(
                model=input_dir,
                download_dir=input_dir,
                trust_remote_code=True,
                dtype=torch_dtype,
                num_gpu_blocks_override=1,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=True,
                max_model_len=1,
            )
       
            model_path = os.path.join(storage_path, model_name)
            # vLLM v0.19+ (V1 engine) exposes ServerlessLLM export via the
            # EngineCore client, not via `llm_engine.model_executor`.
            llm_writer.llm_engine.engine_core.save_serverless_llm_state(
                path=model_path, pattern=pattern, max_size=max_size
            )
            for file in os.listdir(input_dir):
                # Copy the metadata files into the output directory
                if os.path.splitext(file)[1] not in (
                    ".bin",
                    ".pt",
                    ".safetensors",
                ):
                    src_path = os.path.join(input_dir, file)
                    dest_path = os.path.join(model_path, file)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path)
                    else:
                        shutil.copy(src_path, dest_path)
            del llm_writer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        try:
            with TemporaryDirectory() as cache_dir:
                input_dir = local_model_path
                # download from huggingface
                if local_model_path is None:
                    input_dir = snapshot_download(
                        model_name,
                        cache_dir=cache_dir,
                        allow_patterns=[
                            "*.safetensors",
                            "*.bin",
                            "*.json",
                            "*.txt",
                        ],
                    )
                _run_writer(input_dir, model_name)
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            # remove the output dir
            out = os.path.join(storage_path, model_name)
            if os.path.isdir(out):
                shutil.rmtree(out)
            raise RuntimeError(
                f"Failed to save {model_name} for vllm backend: {e}"
            ) from e


parser = argparse.ArgumentParser(
    description="Save a model from HuggingFace model hub."
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Model name from HuggingFace model hub.",
)
parser.add_argument(
    "--local-model-path",
    type=str,
    required=False,
    help="Local path to the model snapshot.",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default=os.path.expanduser("~/models"),
    help="Local path to save the model.",
)
parser.add_argument(
    "--tensor-parallel-size",
    type=int,
    default=1,
    help="Tensor parallel size.",
)

args = parser.parse_args()

model_name = args.model_name
local_model_path = args.local_model_path
storage_path = args.storage_path
tensor_parallel_size = args.tensor_parallel_size

downloader = VllmModelDownloader()
downloader.download_vllm_model(
    model_name,
    "bfloat16",
    tensor_parallel_size=tensor_parallel_size,
    storage_path=storage_path,
    local_model_path=local_model_path,
)
