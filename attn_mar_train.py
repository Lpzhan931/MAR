import os
import json
import pathlib
from typing import Dict, Optional
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from safetensors.torch import save_file
import transformers
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from attn_mar_model import MARModel, MARConfig
from train_settings import Config


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Customized for training MARModel
class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            medusa_num_heads = model.module.medusa_num_heads
        else:
            medusa_num_heads = model.medusa_num_heads

        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]

        # # Shift so that tokens < n predict n
        # loss = 0
        # 修复：不要写 loss = 0
        # 而是用 0.0 乘以 logits 的总和。
        # 这会创造一个值为 0 的 Tensor，并且它连接着整个模型的计算图
        # 这样即使整个 batch 全是 -100，.backward() 也能安全执行，梯度全为 0。
        loss = 0.0 * logits.sum() 

        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(medusa_num_heads):
            mar_logits = logits[i, :, : -(2 + i)].contiguous()
            mar_labels = labels[..., 2 + i :].contiguous()
            mar_logits = mar_logits.view(-1, logits.shape[-1])
            mar_labels = mar_labels.view(-1)
            mar_labels = mar_labels.to(mar_logits.device)

            not_ignore = mar_labels.ne(IGNORE_TOKEN_ID)

            # NOTE 拦截空标签，防止后面除以 0 产生 NaN (preprocess 可能产生空标签)
            # 如果全都是 -100，直接跳过计算
            if not_ignore.sum() == 0:
                log[f"mar_head{i}_loss"] = 0.0
                log[f"mar_head{i}_top1"] = 0.0
                continue

            loss_i = loss_fct(mar_logits, mar_labels)
            loss += loss_i
            
            mar_labels = mar_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 2):
                _, topk = mar_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(mar_labels.unsqueeze(-1)).any(-1)
                log[f"mar_head{i}_top{k}"] = correct.float().mean().item()

            log[f"mar_head{i}_loss"] = loss_i.item()
        self.log(log)   # NOTE transformers 版本问题 log 可能有问题
        return (loss, logits) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model        
        
        # 提取所有 requires_grad=True 的参数
        trainable_state_dict = {
            k: v.cpu() for k, v in model_to_save.named_parameters() if v.requires_grad
        }
        save_file(trainable_state_dict, os.path.join(output_dir, "mar.safetensors"))    

        if hasattr(model_to_save, "config"):
            model_to_save.config.save_pretrained(output_dir)
            
        print(f"\n[Trainer] 已保存 mar.safetensors 到: {output_dir}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=Config.BASE_MODEL_PATH)
    small_model_name_or_path: Optional[str] = field(
        default=Config.SMALL_MODEL_PATH,
        metadata={"help": "Path to the small auxiliary model."}
    )
    freeze_base_model: bool = field(
        default=True, metadata={"help": "whether to freeze the base model)"}
    )
    freeze_small_model: bool = field(
        default=True, metadata={"help": "whether to freeze the small model)"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = "none"
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads in MAR."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head in MAR."},
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, conversation in enumerate(sources):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompts.append(prompt)
        conversations.append(conversation)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):

        search_start = 0

        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                stripped_content = content.strip()
                
                # 如果内容为空，直接跳过
                if not stripped_content:
                    continue

                try:
                    # 获取去掉所有空格/换行符的版本，纯净地匹配
                    clean_prompt = prompt.replace(" ", "").replace("\n", "").replace("\t", "")
                    clean_content = stripped_content.replace(" ", "").replace("\n", "").replace("\t", "")
                    
                    # 在纯净版中寻找相对起始位置
                    clean_start = clean_prompt.index(clean_content, search_start)
                    clean_stop = clean_start + len(clean_content)
                    search_start = clean_stop # 下次从这里开始找
                    
                    # 利用双指针，将 clean 版本的 index 映射回真实 prompt 的 index
                    start = 0
                    clean_ptr = 0
                    while clean_ptr < clean_start and start < len(prompt):
                        if prompt[start] not in [" ", "\n", "\t"]:
                            clean_ptr += 1
                        start += 1
                        
                    stop = start
                    while clean_ptr < clean_stop and stop < len(prompt):
                        if prompt[stop] not in [" ", "\n", "\t"]:
                            clean_ptr += 1
                        stop += 1

                    # 映射 Tokens
                    indices = []
                    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                        # 只要 token 的范围和内容的范围有交集，就计入
                        if tok_stop > start and tok_start < stop:
                            indices.append(tok_index)
                    
                    if indices:
                        target[indices] = encoding.input_ids[conv_index][indices]
                        
                except ValueError:
                    # 如果这都找不到，说明模板彻底改变了原句内容
                    # print(f"Warning: Skipped masking for a turn in conv {conv_index} due to severe string mismatch.")
                    continue

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class JsonlLogCallback(TrainerCallback):
    """
    训练日志实时写入 JSONL 文件的 Callback
    """
    def __init__(self, log_path):
        self.log_path = log_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # 加入当前的全局步数，方便对齐
            logs["global_step"] = state.global_step
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    # Llama-2 模型的 Tokenizer 没有自带 chat_template，需要手动注入 Llama-2 的标准模板
    if tokenizer.chat_template is None:
        print("Warning: chat_template not found. Injecting default Llama-2 template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "<<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n"
                "{% elif message['role'] == 'user' or message['role'] == 'human' %}"
                    "[INST] {{ message['content'] }} [/INST] "
                "{% elif message['role'] == 'assistant' or message['role'] == 'gpt' %}"
                    "{{ message['content'] }} </s>"
                "{% endif %}"
            "{% endfor %}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    if model_args.freeze_base_model:
        print("Freezing Base Model...")
        # for param in model.base_model.parameters():
        for param in model.parameters():    # also freeze the lm_head
            param.requires_grad = False
    else:
        print("Base Model is UNFROZEN! Will Train base model parameters.")

    # The small model is frozen in MARModel

    mar_model = MARModel(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
        small_model_name_or_path=model_args.small_model_name_or_path,
        freeze_small_model=model_args.freeze_small_model,
    )

    # Generate MAR config for pushing to HF hub
    mar_config = MARConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
        small_model_name_or_path=model_args.small_model_name_or_path
    )

    
    # Save MAR config
    mar_config.save_pretrained(training_args.output_dir)

    # 初始化 Callback 并传入 Trainer
    log_file_path = os.path.join(training_args.output_dir, "training_logs.jsonl")
    jsonl_callback = JsonlLogCallback(log_file_path)

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = CustomizedTrainer(
        model=mar_model, 
        args=training_args, 
        callbacks=[jsonl_callback],
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print("训练结束.")

    model.config.use_cache = True
    actual_model = mar_model.module if hasattr(mar_model, "module") else mar_model
    
    if local_rank == 0 or local_rank == -1:
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        
        # 提取所有 requires_grad=True 的自定义参数
        trainable_state_dict = {
            k: v.cpu() for k, v in actual_model.named_parameters() if v.requires_grad
        }
        
        save_file(
            trainable_state_dict,
            os.path.join(training_args.output_dir, "mar.safetensors"),
        )
        print(f"MAR 混合权重已成功导出至: {training_args.output_dir}")


if __name__ == "__main__":
    train()
