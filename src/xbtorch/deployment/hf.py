from xbtorch.patches import replace_all_layers_stateless
from transformers import LlamaForCausalLM, LlamaConfig

from .base import ACCELERATOR_REGISTRY


class NoisyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, accelerator_name=None, accelerator_kwargs=None, exclude=None):
        super().__init__(config)
        print("BOOP", accelerator_name)
        exit()
        if accelerator_name is not None:
            if accelerator_name not in ACCELERATOR_REGISTRY:
                raise ValueError(f"Unknown layer {accelerator_name}")
            layer_cls = ACCELERATOR_REGISTRY[accelerator_name]
            replace_all_layers_stateless(self, layer_cls, accelerator_kwargs, exclude)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Pop the Python class / type out before passing kwargs to HF
        accelerator_name = kwargs.pop("accelerator_name", None)
        accelerator_kwargs = kwargs.pop("accelerator_kwargs", None)
        exclude = kwargs.pop("exclude", None)

        # HF will call __init__, which uses the string to map to class
        return super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            accelerator_name=accelerator_name,
            accelerator_kwargs=accelerator_kwargs,
            exclude=exclude,
            **kwargs
        )