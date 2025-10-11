from transformers import AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from transformers import AutoTokenizer

from xbtorch.deployment import NoisyLlamaForCausalLM, SimpleFixedPoint
import xbtorch
from xbtorch.patches import xbtorch_model

import torch

def noisy_lm_evaluate(model_path,
                    eval_tasks, 
                    eval_limit=None,
                    num_fewshot=0,
                    accelerator_name=None, 
                    accelerator_kwargs=None,
                    exclude=None,
                    batch_size=32,
                    **config_kwargs):

    print("ASD0")
    # Fix mutable defaults
    if accelerator_kwargs is None:
        accelerator_kwargs = {}
    if exclude is None:
        exclude = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("ASD1")
    acc = SimpleFixedPoint(
        g_min=100,
        g_max=200,
        device=device,
        stateful=False,
        adc_bits=8,
        dac_bits=8
    )

    print("ASD2")
    xbtorch.initialize(inference_accelerator=acc, pytorch_device=device)

    # Default kwargs for model loading
    default_config_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    # Let user override defaults
    config_kwargs = {**default_config_kwargs, **config_kwargs}

    print("ASD3")
    model = NoisyLlamaForCausalLM.from_pretrained(model_path,
                                                  accelerator_name=accelerator_name,
                                                  accelerator_kwargs=accelerator_kwargs,
                                                  **config_kwargs,
                                                  exclude=exclude)

    # model = LlamaForCausalLM(model_path,
    #                              **config_kwargs)
    
    model = xbtorch_model(model, replace_all=True)

    model.xb_eval(enable=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    patched_lm_eval_model = HFLM(pretrained=model,
                                tokenizer=tokenizer,
                                batch_size=batch_size
                                )

    patched_results = evaluator.simple_evaluate(
        model=patched_lm_eval_model,
        tasks=eval_tasks,
        num_fewshot=num_fewshot,
        limit=eval_limit
    )

    return patched_results['results']

if __name__ == "__main__":
    eval_tasks = ["piqa"]
    num_fewshot = 0
    eval_limit = 10#None
    batch_size = 40

    models = [
                # f'facebook/{model_family}-BF16', 
                # f'facebook/{model_family}-1-bit', 
                # f'facebook/{model_family}-1.58-bit', 
                # f'facebook/{model_family}-2-bit', 
                # f'facebook/{model_family}-3-bit', 
                # f'facebook/{model_family}-4-bit', 
                "SpectraSuite/TriLM_830M_Unpacked"
              ]

    for model_path in models:
        result = noisy_lm_evaluate(model_path=model_path,
                                    eval_limit=eval_limit,
                                    batch_size=batch_size,
                                    num_fewshot=num_fewshot,
                                    accelerator_name=None,
                                    accelerator_kwargs=None,
                                    eval_tasks=eval_tasks,
                                    exclude=["lm_head"]
                                    )

        print(result)