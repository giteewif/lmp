from transformers import AutoConfig
from accelerate import init_empty_weights
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


class GptOssModule:
    def create_empty_model(self, config):
        with init_empty_weights():
            return GptOssForCausalLM(config)

    def get_config(self, path):
        return AutoConfig.from_pretrained(path, trust_remote_code=True)
