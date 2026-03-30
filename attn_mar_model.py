import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from train_settings import Config


class MARConfig(PretrainedConfig):
    """
    Configuration class for MAR model.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=Config.BASE_MODEL_PATH,
        small_model_name_or_path=Config.SMALL_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.small_model_name_or_path = small_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    """

    def __init__(self, hidden_size, hidden_size_sm=None):
        super().__init__()
        if hidden_size_sm is not None:
            self.linear = nn.Linear(hidden_size, hidden_size_sm)
        else:
            self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

        if hidden_size_sm is not None:
            self.shortcut = nn.Linear(hidden_size, hidden_size_sm, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the ResBlock.
        """
        return self.shortcut(x) + self.act(self.linear(x))


class MARModel(nn.Module):
    """The MAR Language Model Head.
    """

    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=Config.BASE_MODEL_PATH,
        small_model_name_or_path=Config.SMALL_MODEL_PATH,
        freeze_small_model=True,
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            medusa_num_layers (int, optional): Number of ResBlock layers for each MAR head. Defaults to 0.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.small_model_name_or_path = small_model_name_or_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        except:
            self.tokenizer = None
        
        print(f"Loading small model: {self.small_model_name_or_path}")
        self.small_model = AutoModelForCausalLM.from_pretrained(
            self.small_model_name_or_path,
            torch_dtype=base_model.dtype,
            # device_map=base_model.device,
        )
        # Freeze the small model
        if freeze_small_model:
            print("Freezing Small Model...")
            for param in self.small_model.parameters():
                param.requires_grad = False
        else:
            print("Small Model is UNFROZEN! Will train small model parameters.")
        
        self.hidden_size_sm = self.small_model.config.hidden_size

        # 初始化 decoder_layer
        self.extra_decoder_layer = Qwen3DecoderLayer(self.config, layer_idx=0)
        # 零初始化
        nn.init.zeros_(self.extra_decoder_layer.self_attn.o_proj.weight)
        nn.init.zeros_(self.extra_decoder_layer.mlp.down_proj.weight)


        self.medusa_head = nn.ModuleList()
        for _ in range(medusa_num_heads):
            layers = []            
            layers.append(ResBlock(self.hidden_size, self.hidden_size_sm))  # 第一层降维
            for _ in range(medusa_num_layers - 1):  # 后面的层在小维度下运算
                layers.append(ResBlock(self.hidden_size_sm, self.hidden_size_sm))
            self.medusa_head.append(nn.Sequential(*layers))

        # FC层与 SpS 预测头, 将大模型特征(H_i)与小模型特征(h_i)拼接映射回 hidden_size_sm
        self.fc_layer = nn.Linear(self.hidden_size_sm + self.hidden_size_sm, self.hidden_size_sm)
        # 零初始化, 初始时不干扰大模型的特征
        torch.nn.init.zeros_(self.fc_layer.weight)
        if self.fc_layer.bias is not None:
            torch.nn.init.zeros_(self.fc_layer.bias)

        self.mar_lm_head = nn.Linear(self.hidden_size_sm, self.vocab_size, bias=False)
        
        # 初始化策略：拷贝小模型 lm_head 的权重给 mar_lm_head (原名 lm_head_sps)
        self.mar_lm_head.weight.data.copy_(self.small_model.lm_head.weight.data)

        # Ensure these dtype and device align with the base_model
        self.extra_decoder_layer.to(self.base_model.dtype).to(self.base_model.device)
        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)
        self.fc_layer.to(self.base_model.dtype).to(self.base_model.device)
        self.mar_lm_head.to(self.base_model.dtype).to(self.base_model.device)
        

    def get_tokenizer(self):
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        mar_name_or_path,
        **kwargs,
    ):
        mar_config = MARConfig.from_pretrained(mar_name_or_path)

        # 使用 pop 从 kwargs 中取出自定义的参数，不传给下一层
        base_model_override = kwargs.pop('base_model', None)
        if base_model_override:
            print(f"Overriding base_model from config with CLI argument: {base_model_override}")
            mar_config.base_model_name_or_path = base_model_override
        small_model_override = kwargs.pop('small_model', None)
        if small_model_override:
            print(f"Overriding small_model from config with CLI argument: {small_model_override}")
            mar_config.small_model_name_or_path = small_model_override

        print(f"Loading base model: {mar_config.base_model_name_or_path}")
        base_model_instance = AutoModelForCausalLM.from_pretrained(
            mar_config.base_model_name_or_path,
            torch_dtype="auto",
            **kwargs
        )

        model = cls(
            base_model=base_model_instance,
            medusa_num_heads=mar_config.medusa_num_heads,
            medusa_num_layers=mar_config.medusa_num_layers,
            base_model_name_or_path=mar_config.base_model_name_or_path,
            small_model_name_or_path=mar_config.small_model_name_or_path,
            freeze_small_model=True # 评测时总是冻结
        )

        filename = os.path.join(mar_name_or_path, "mar.safetensors")
        if not os.path.exists(filename):
            filename = os.path.join(mar_name_or_path, "mar.pt")
            if not os.path.exists(filename):
                raise FileNotFoundError(f"MAR weight not found in {mar_name_or_path}")
        
        print(f"Loading MAR weight from {filename}")

        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file
            mar_state_dict = load_file(filename, device="cpu")
        else:
            mar_state_dict = torch.load(filename, map_location="cpu")
        
        # mar_state_dict 中 small_model 和其它的分别处理
        small_model_state_dict = {}
        mar_only_state_dict = {}
        for key, value in mar_state_dict.items():
            if key.startswith("small_model."):
                small_model_state_dict[key.replace("small_model.", "")] = value
            elif not key.startswith("base_model."):
                mar_only_state_dict[key] = value
        
        # 1. mar-only 加载
        missing_mar, unexpected_mar = model.load_state_dict(mar_only_state_dict, strict=False)
        missing_mar_true = [s for s in missing_mar if not s.startswith(("base_model.", "small_model."))]
        print(f"MAR-Only weight loaded. Missing keys: {missing_mar_true}, Unexpected keys: {unexpected_mar}")

        # 2. fine-tuned small_model 加载
        if small_model_state_dict:
            print("Found and loading fine-tuned small_model weights...")
            missing_sm, unexpected_sm = model.small_model.load_state_dict(small_model_state_dict, strict=False)
            print(f"Small_model weights loaded. Missing keys: {missing_sm}, Unexpected keys: {unexpected_sm}")
            model.small_model.to(model.base_model.device)
        else:
            print("No fine-tuned small_model weights found in the file. Using the original small_model.")
        
        return model


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        return_latencies=False, # profile
    ):
        """Forward pass of the MARModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all MAR heads.
            (Optional) Original predictions from the base model's LM head.
        """

        with torch.no_grad():
            # Pass input through the base model
            outputs_lm = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=False,     # 不需要输出全部层的 hidden_states
            )
            last_hidden_states_lm = outputs_lm.last_hidden_state
            if output_orig:
                orig = self.base_model.lm_head(last_hidden_states_lm)

        # 以下这些放在 no_grad 外面
        # 为 extra_decoder_layer 准备 Attention Mask 与 Position IDs
        batch_size, seq_length = input_ids.shape[:2]

        # 1. Position IDs (如果没有传入，基于 kv_cache 长度推断)
        if position_ids is None:
            # 兼容 HuggingFace 最新的 DynamicCache API
            past_length = past_key_values.get_usable_length(seq_length) if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # 2. Attention Mask (兼容 FlashAttention2 和 SDPA)
        layer_attention_mask = attention_mask
        is_flash_attn = getattr(self.config, "_attn_implementation", "") == "flash_attention_2"
        if not is_flash_attn and last_hidden_states_lm.shape[1] > 1 and attention_mask is not None and attention_mask.dim() == 2:
            seq_len = last_hidden_states_lm.shape[1]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=last_hidden_states_lm.device, dtype=torch.bool))
            causal_4d = causal_mask[None, None, :, :].expand(last_hidden_states_lm.shape[0], 1, seq_len, seq_len)
            
            padding_mask = attention_mask[:, None, None, :].expand(-1, 1, seq_len, -1).bool()
            combined_mask = causal_4d & padding_mask
            
            layer_attention_mask = torch.zeros_like(combined_mask, dtype=last_hidden_states_lm.dtype)
            layer_attention_mask.masked_fill_(~combined_mask, torch.finfo(last_hidden_states_lm.dtype).min)


        # 3. 前向传播 Extra Decoder Layer (这部分要允许计算梯度，所以移出 no_grad)
        # extra_decoder_layer 会返回一个 tuple: (hidden_states, present_key_value, ...)
        position_embeddings = self.base_model.model.rotary_emb(last_hidden_states_lm, position_ids)

        layer_outputs = self.extra_decoder_layer(
            hidden_states=last_hidden_states_lm,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            past_key_value=None,    # 训练时不使用 Cache
            output_attentions=False,
            use_cache=False,        # 训练时不使用 Cache
            position_embeddings=position_embeddings,
        )
        
        final_concat_hidden_lm = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs    # 5.3.0 版本实际返回 tensor

        # 小模型要解冻的话必须将这里移出 no_grad, 让 PyTorch 追踪计算图
        # pass input through the small model
        outputs_sm = self.small_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            output_hidden_states=False,
        )

        hidden_states_sm = outputs_sm.last_hidden_state  # (batch_size, seq_len, hidden_size_sm)
        
        mar_logits = []
        batch_size, seq_len, d_sm = hidden_states_sm.shape


        for i in range(self.medusa_num_heads):
            # 大模型分支提取
            m_hidden = self.medusa_head[i](final_concat_hidden_lm.clone())
            
            # 小模型分支特征平移 (Teacher Forcing)
            # 头 i 预测 t+i+2 步，需要小模型输入 t+i+1 步后的特征
            shift = i + 1 

            # 为了支持反向传播, 平移时不要直接对 s_hidden 切片赋值, 而是采用下面的方式
            if seq_len > shift:
                shifted_hidden = hidden_states_sm[:, shift:, :] # 提取平移后的特征
                # 在 seq_len 维度（倒数第2个维度）末尾补 shift 个 0，其它维度补 0
                s_hidden = F.pad(shifted_hidden, pad=(0, 0, 0, shift)) 
            else:
                s_hidden = torch.zeros_like(hidden_states_sm)

            # [B, T, D_LM + D_SM]
            concat_hidden = torch.cat([m_hidden, s_hidden], dim=-1)
            
            # # FC 残差连接
            # fc_out = m_hidden + self.fc_layer(concat_hidden)
            fc_out = self.fc_layer(concat_hidden)

            mlogits = self.mar_lm_head(fc_out)
            mar_logits.append(mlogits)

        if output_orig:
            return torch.stack(mar_logits, dim=0), outputs_lm, orig
        return torch.stack(mar_logits, dim=0)

    