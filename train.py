import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import logging
import os

import torch
import torch.distributed
import transformers
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset
import datasets
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from lift_dataset import load_lift_dataset
from lift_sft_trainer import LIFTSFTTrainer
from lift_args import LIFTDataArguments
from typing import Union

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Base model or residual model setting
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B")
    attn_implementation: Optional[str] = field(default="flash_attention_2")
    # Lora or PiSSA setting
    full_finetune: Optional[bool] = field(default=True)
    adapter_name_or_path: Optional[str] = field(default=None,metadata={"help": ("Pre-initialized PiSSA adapter path; when this is not None, the following arguments are ignored."),},)
    init_weights: Union[bool, str] = field(default=True,metadata={"help": ("True -> LoRA; `pissa` -> PiSSA; `pissa_niter_16` -> Fast SVD PiSSA"),},)
    target_modules: Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank: Optional[int] = field(default=8)
    lora_alpha: Optional[float] = field(default=32.)
    lora_dropout: Optional[float] = field(default=0.,metadata={"help": ("Must be set to 0 when using PiSSA."),},)
    # Quantization setting
    bits: int = field(default=16,metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True,metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4",metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    # TrainingArguments
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512,metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    merge : Optional[bool] = field(default=False,metadata={"help": "Merge the PiSSA adapter to the residual model or LoRA to the base model"},)

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)
        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, 'completed'))
        if is_completed: return None # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and filename.startswith(PREFIX_CHECKPOINT_DIR):
                max_step = max(max_step, int(filename.replace(PREFIX_CHECKPOINT_DIR + '-', '')))
        if max_step == 0: return None
        latest_ckpt_dir = os.path.join(checkpoint_dir, f'{PREFIX_CHECKPOINT_DIR}-{max_step}')
        logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
        return latest_ckpt_dir
    return None # first training

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [tokenizer(text, max_length=tokenizer.model_max_length,truncation=True,)for text in strings]
    input_ids = labels = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def train_tokenize_function(examples, tokenizer, query, response):
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}\n{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def build_model(script_args, checkpoint_dir):
    if script_args.full_finetune:
        assert script_args.bits in [16, 32]
    compute_dtype = (torch.bfloat16 if script_args.bf16 else torch.float32)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.bits == 4,
            load_in_8bit=script_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=script_args.double_quant,
            bnb_4bit_quant_type=script_args.quant_type,
        ) if script_args.bits in [4, 8] else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # Tokenizer
    
    if not script_args.full_finetune:
        if script_args.bits < 16:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=script_args.gradient_checkpointing)

        if checkpoint_dir is not None:
            logger.info(f"Loading adapters from {checkpoint_dir}.")
            # os.path.join(checkpoint_dir, 'adapter_model')
            model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
        elif script_args.adapter_name_or_path is not None:
            logger.info(f"Initilize LoRA/PiSSA/CLOVER adapters from {script_args.model_name_or_path}/{script_args.adapter_name_or_path}.")
            model = PeftModel.from_pretrained(model, script_args.model_name_or_path, subfolder = script_args.adapter_name_or_path, is_trainable=True)
        else:
            logger.info(f'Init LoRA/PiSSA modules...')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=script_args.target_modules.split(','),
                inference_mode=False,
                r=script_args.lora_rank, 
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                init_lora_weights=script_args.init_weights,
            )
            model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    return model

def train():
    parser = transformers.HfArgumentParser((TrainingArguments, LIFTDataArguments))
    script_args, data_args = parser.parse_args_into_dataclasses()
    log_level = script_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
        
    if script_args.local_rank == 0:
        logger.info('='*100)
        logger.info(script_args)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.local_rank == 0:
        logger.info("Load tokenizer from {} over.".format(script_args.model_name_or_path))
    
    resume_from_checkpoint_dir = get_last_checkpoint(script_args.output_dir)
    model = build_model(script_args, resume_from_checkpoint_dir)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if script_args.local_rank > 0: 
        torch.distributed.barrier()

    data_modules = load_lift_dataset(tokenizer, data_args, script_args.model_max_length)

    if script_args.local_rank == 0:
        torch.distributed.barrier()
        print(model)
        logger.info("Training dataset samples:", len(data_modules['train_dataset']))
        for index in random.sample(range(len(data_modules['train_dataset'])), 3):
            logger.info(f"Sample {index} of the training set: {data_modules['train_dataset'][index]['input_ids']}, {data_modules['train_dataset'][index]['labels']}.")
            logger.info(f"Sample {index} of the training set: {tokenizer.decode(list(data_modules['train_dataset'][index]['input_ids']))}.")

    script_args.accelerator_config.even_batches = False
    trainer = LIFTSFTTrainer(model=model, tokenizer=tokenizer, args=script_args, **data_modules)
    if not script_args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    trainer.train(resume_from_checkpoint = resume_from_checkpoint_dir)
    trainer.save_state()
    if not script_args.full_finetune and script_args.merge:
        model = model.merge_and_unload()
        model.save_pretrained(script_args.output_dir)
        tokenizer.save_pretrained(script_args.output_dir)
    if script_args.full_finetune:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=script_args.output_dir)
        

if __name__ == "__main__":
    train()

