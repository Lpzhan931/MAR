"""
python mar_benchmark.py \
    --mar-model-path /home/pzli/Project/Spec/model/mar_model/mar_full_data/checkpoint-4928 \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k \
    --num-samples 80 
"""
import torch
import time
import json
import os
from tqdm import tqdm
from datetime import datetime

from mar_model import MARModel

DATA_DIR = "/home/pzli/Project/Spec/SpS/260327_attn_medusa/commit/data"
LOG_DIR = "./evaluation/logs"

def trim_kv_cache(past_key_values, keep_len):
    """
    安全裁剪 KV Cache, 兼容新版 DynamicCache 与 Tuple
    """
    if past_key_values is None: 
        return None
    
    # 1. Transformers 新版 DynamicCache (最佳做法：调用官方 crop)
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_len)
        return past_key_values
    
    # 2. 传统 tuple 格式 (基础模型的多层)
    if isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
        new_past = []
        for layer in past_key_values:
            k = layer[0][:, :, :keep_len, :]
            v = layer[1][:, :, :keep_len, :]
            new_past.append((k, v))
        return tuple(new_past)
        
    return past_key_values


@torch.no_grad()
def benchmark_mar_generate(model, input_ids, max_new_tokens=256, debug_log_file=None):
    stats = {
        "draft_time": 0,      
        "verify_time": 0, 
        "draft_count": 0,      
        "verify_count": 0,     
        "accepted_lengths": [],
        "position_attempts": torch.zeros(model.medusa_num_heads).cuda(), 
        "position_accepts": torch.zeros(model.medusa_num_heads).cuda(),
        "new_tokens": 0
    }

    device = model.base_model.device
    tokenizer = model.tokenizer

    terminators = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and isinstance(im_end, int):
            terminators.append(im_end)

    prompt_len = input_ids.shape[1]
    
    if debug_log_file:
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write("🔍 [DEBUG] MAR 投机解码详细过程追踪 (Sample 0)\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(f"📝 Prompt 长度: {prompt_len} tokens\n\n")

    seq = input_ids[0].tolist()
    
    # ==========================================
    # 0. 预填充 (Prefill)
    # ==========================================
    t0 = time.perf_counter()
    
    # Base Model Prefill
    t_out = model.base_model.model(input_ids.to(device), use_cache=True, output_hidden_states=True)
    target_kv = t_out.past_key_values
    
    # 获取特征进行 MAR 初始化
    last_hidden_states_lm = t_out.hidden_states[-1][:, -1:, :]
    orig_logits = model.base_model.lm_head(last_hidden_states_lm)
    base_token = torch.argmax(orig_logits[0, -1, :]).item()
    
    seq.append(base_token)
    stats["new_tokens"] += 1
    stats["verify_time"] += (time.perf_counter() - t0)
    stats["verify_count"] += 1
    
    # Small Model Prefill
    d_out = model.small_model.model(input_ids.to(device), use_cache=True, output_hidden_states=True)
    draft_kv = d_out.past_key_values

    # Token Mixer 历史特征缓冲池 (保留近 kernel_size 步的融合特征)
    feature_buffer = []

    step_counter = 0

    while stats["new_tokens"] < max_new_tokens:
        if seq[-1] in terminators:
            break
        step_counter += 1
        
        ctx_len_before = len(seq)
        L_before = ctx_len_before - 1 # target_kv 里面此时包含的 token 数量 (不含 base_token)

        # ==========================================
        # 1. Drafting 阶段 (MAR Head + Small Model)
        # ==========================================
        torch.cuda.synchronize()
        t_d_start = time.perf_counter()
        
        # a. 提取上一阶段 Base Model 的最后一步特征进行 Layer Fusion
        # (如果是 step 1，提取的是 prefill 最后一个 token；之后是上一个 step 的 verify 最后一个 token)
        selected_hiddens_lm = [t_out.hidden_states[idx][:, -1:, :] for idx in model.extract_layers]
        fused_hidden = model.layer_fusion(selected_hiddens_lm) # [1, 1, H]
        
        # b. 维护 Token Mixer Buffer
        feature_buffer.append(fused_hidden)
        if len(feature_buffer) > model.conv_kernel_size:
            feature_buffer.pop(0)
            
        # 拼接近期的特征送入 Token Mixer
        buffer_tensor = torch.cat(feature_buffer, dim=1) # [1, seq_len <= kernel_size, H]
        mixed_hidden = model.token_mixer(buffer_tensor)[:, -1:, :] # [1, 1, H] (只取最后时刻)

        # c. 迭代生成草稿
        draft_tokens = []
        curr_d_in = torch.tensor([[base_token]], device=device)
        
        for i in range(model.medusa_num_heads):
            # 小模型前向
            s_out = model.small_model.model(curr_d_in, past_key_values=draft_kv, use_cache=True, output_hidden_states=True)
            draft_kv = s_out.past_key_values
            s_hidden = s_out.hidden_states[-1][:, -1:, :] # [1, 1, d_sm]
            
            # 大模型特征分支
            m_hidden = model.medusa_head[i](mixed_hidden)
            
            # 融合预测
            concat_hidden = torch.cat([m_hidden, s_hidden], dim=-1)
            # fc_out = m_hidden + model.fc_layer(concat_hidden)
            fc_out = model.fc_layer(concat_hidden)
            mlogits = model.mar_lm_head(fc_out)
            
            next_tok = torch.argmax(mlogits, dim=-1).item()
            draft_tokens.append(next_tok)
            
            stats["position_attempts"][i] += 1
            
            if next_tok in terminators:
                break
            curr_d_in = torch.tensor([[next_tok]], device=device)
            
        torch.cuda.synchronize()
        step_draft_time = time.perf_counter() - t_d_start
        stats["draft_time"] += step_draft_time
        stats["draft_count"] += 1
        
        actual_gamma = len(draft_tokens)

        # ==========================================
        # 2. Verification 阶段 (Base Model)
        # ==========================================
        torch.cuda.synchronize()
        t_v_start = time.perf_counter()
        
        # Base Model 需要验证 [Base_Token] + [Draft_Tokens]
        candidates = [base_token] + draft_tokens
        t_in = torch.tensor([candidates], device=device)
        
        t_out = model.base_model.model(t_in, past_key_values=target_kv, use_cache=True, output_hidden_states=True)
        target_kv = t_out.past_key_values
        
        # 通过 LM Head 得到这批序列产生的预测
        v_logits = model.base_model.lm_head(t_out.hidden_states[-1])
        target_preds = torch.argmax(v_logits[0], dim=-1).tolist()
        
        # 贪心匹配 (Matching)
        accept_length = 0
        for i in range(actual_gamma):
            if target_preds[i] == draft_tokens[i]:
                accept_length += 1
                stats["position_accepts"][i] += 1
            else:
                break
                
        stats["accepted_lengths"].append(accept_length)
        
        # 被接受的 Token + 1个大模型纠正的 Bonus Token
        accepted_ids = draft_tokens[:accept_length]
        bonus_token = target_preds[accept_length]
        
        tokens_to_add = accepted_ids + [bonus_token]
        seq.extend(tokens_to_add)
        stats["new_tokens"] += len(tokens_to_add)

        # ==========================================
        # 3. KV Cache 回滚 (Rollback) & 状态更新
        # ==========================================
        # Base Model 需要保留到 L_before + 1(base_token) + accept_length
        keep_len_base = L_before + 1 + accept_length
        target_kv = trim_kv_cache(target_kv, keep_len_base)
        
        # Small Model 同样进行裁剪
        keep_len_small = keep_len_base
        if accept_length == actual_gamma and seq[-1] not in terminators:
            # 如果全接受，小模型还需要吃下最后一个草稿的 token 来对齐状态 (类似于 sps_generator 逻辑)
            missing_d_in = torch.tensor([[draft_tokens[-1]]], device=device)
            dummy_out = model.small_model.model(missing_d_in, past_key_values=draft_kv, use_cache=True)
            draft_kv = dummy_out.past_key_values
        else:
            draft_kv = trim_kv_cache(draft_kv, keep_len_small)
            
        # 更新下一个 Step 的起点
        # t_out.hidden_states 包含了验证时走过的所有状态，选取被接受序列的最后一个状态用于下一轮 TokenMixer
        last_accepted_idx = accept_length
        # 替换 t_out 里的 hidden_states，以便下一阶段 (Phase 1) 提取特征
        t_out.hidden_states = tuple(h[:, last_accepted_idx:last_accepted_idx+1, :] for h in t_out.hidden_states)
        base_token = bonus_token

        torch.cuda.synchronize()
        step_verify_time = time.perf_counter() - t_v_start
        stats["verify_time"] += step_verify_time
        stats["verify_count"] += 1

        # ==========================================
        # 4. 打印 Debug 信息
        # ==========================================
        if debug_log_file:
            # 1. 真正被拒绝的 Token
            truly_rejected_id = []
            if accept_length < actual_gamma:
                truly_rejected_id = [draft_tokens[accept_length]]
            
            # 2. 因前者被拒而从未被评估的后续草稿 Token
            unseen_draft_ids = []
            if accept_length + 1 < actual_gamma:
                unseen_draft_ids = draft_tokens[accept_length + 1:]

            bonus_id = [bonus_token]

            # 解码为文本
            draft_str = tokenizer.decode(draft_tokens)
            accepted_str = tokenizer.decode(accepted_ids) if accepted_ids else "(None)"
            rejected_str = tokenizer.decode(truly_rejected_id) if truly_rejected_id else "(None - Draft fully matched)"
            unseen_str = tokenizer.decode(unseen_draft_ids) if unseen_draft_ids else "(None)"
            bonus_str = tokenizer.decode(bonus_id)
            merged_str = tokenizer.decode(tokens_to_add)
            
            debug_log_file.write(f"Step {step_counter} | Accept Length: {accept_length}/{actual_gamma}\n")
            debug_log_file.write(f"\tTime: Verify {step_verify_time*1000:.2f} ms | Draft {step_draft_time*1000:.2f} ms\n")
            debug_log_file.write(f"\t[Draft]   '{draft_tokens}' | {draft_str}\n")
            debug_log_file.write(f"\t[Accept]  '{accepted_ids}' | {accepted_str}\n")
            debug_log_file.write(f"\t[Reject]  '{truly_rejected_id}' | {rejected_str}\n")
            debug_log_file.write(f"\t[Bonus]   '{bonus_id}' | {bonus_str}\n")
            debug_log_file.write(f"\t[Unseen]  '{unseen_draft_ids}' | {unseen_str}\n")
            debug_log_file.write(f"\t[Final]   '{tokens_to_add}' | {merged_str}\n")
            debug_log_file.write("-" * 80 + "\n")
            debug_log_file.flush()

    if debug_log_file:
        debug_log_file.write("\n" + "="*60 + "\n")
        debug_log_file.write("📄 [DEBUG] 最终完整生成的回答:\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(tokenizer.decode(seq[prompt_len:]) + "\n")
        debug_log_file.write("="*60 + "\n\n")

    return stats


def run_benchmark(args):
    # 1. 加载 MAR 模型 (会同时加载 base_model 和 small_model)
    model = MARModel.from_pretrained(
        mar_name_or_path=args.mar_model_path,
        base_model=args.base_model_path,
        device_map="auto",
        # small_model 是如果在 MARConfig 里写死的话可以不传，而且一般会从 mar 中加载训练后的参数，如果需要覆盖可加参数
    )
    model.eval()

    questions = []
    # question_file = f"{DATA_DIR}/{args.bench_name}/question.jsonl"
    question_file = os.path.join(DATA_DIR, f"{args.bench_name}/question.jsonl")
    if not os.path.exists(question_file):
        question_file = f"/home/pzli/Project/Spec/EAGLE/eagle/data/{args.bench_name}/question.jsonl"
    with open(question_file, "r") as f:
        for line in f: questions.append(json.loads(line))

    # --- 1. 预热阶段 ---
    print("🔥 Warming up...")
    dummy_input = model.tokenizer(["Hello, who are you?"], return_tensors="pt").input_ids.cuda()
    benchmark_mar_generate(model, dummy_input, max_new_tokens=16)

    # --- 2. 纯粹的 Debug 阶段 ---
    if args.show_first_sample:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"mar_benchmark_{timestamp}.log")
        os.makedirs(LOG_DIR, exist_ok=True)
        print(f"\n📝 正在进行单样本详细解码追踪 (耗时不计入最终评测)...")
        print(f"📂 详细过程将写入日志文件: \033[96m{log_filename}\033[0m")
        
        q0 = questions[0]
        messages = [{"role": "user", "content": q0["turns"][0]}]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids.cuda()
        
        with open(log_filename, "w", encoding="utf-8") as log_file:
            benchmark_mar_generate(model, input_ids, max_new_tokens=args.max_new_tokens, debug_log_file=log_file)
            
        print(f"✅ 追踪完成！\n")

    # --- 3. 正式的评测阶段 ---
    all_stats = []
    print("🚀 Starting benchmark...")
    
    for q in tqdm(questions[:args.num_samples], total=args.num_samples):
        messages = [{"role": "user", "content": q["turns"][0]}]
        prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = model.tokenizer([prompt], return_tensors="pt").input_ids.cuda()
        
        start_w = time.perf_counter()
        stat = benchmark_mar_generate(model, input_ids, max_new_tokens=args.max_new_tokens, debug_log_file=None)
        stat["total_wall_time"] = time.perf_counter() - start_w
        all_stats.append(stat)

    # --- 4. 结果统计 ---
    total_steps = sum(len(s["accepted_lengths"]) for s in all_stats)
    total_accepts = sum(sum(s["accepted_lengths"]) for s in all_stats)
    avg_accept_len = total_accepts / total_steps if total_steps > 0 else 0
    
    total_tokens = sum(s["new_tokens"] for s in all_stats)
    total_time = sum(s["total_wall_time"] for s in all_stats)
    speed = total_tokens / total_time
    
    total_draft_time = sum(s["draft_time"] for s in all_stats)
    total_verify_time = sum(s["verify_time"] for s in all_stats)
    draft_ratio = total_draft_time / (total_draft_time + total_verify_time)

    avg_draft_time = total_draft_time / sum(s["draft_count"] for s in all_stats)
    avg_verify_time = total_verify_time / sum(s["verify_count"] for s in all_stats)

    print(f"\n====== MAR 测试结果 ======")
    print(f"Base Model: {args.base_model_path}")
    print(f"MAR Model: {args.mar_model_path}")
    print(f"1. 解码速度: {speed:.2f} tokens/s")
    print(f"2. 平均接受长度: {avg_accept_len:.2f}")
    print(f"3. 平均延时: 投机模块 {avg_draft_time*1000:.2f}ms ({draft_ratio*100:.1f}%) | 验证模块 {avg_verify_time*1000:.2f}ms ({(1-draft_ratio)*100:.1f}%)")
    

    # 提取绝对接受次数
    pos_accepts_list = torch.stack([s["position_accepts"] for s in all_stats]).sum(dim=0).cpu().tolist()
    print("4. 逐位置接受率 (条件接受率 / Conditional Acceptance Rate):")
    # Token 1 的分母是总验证步数 (Total Steps)
    current_denominator = total_steps
    for i, accepts in enumerate(pos_accepts_list):
        if current_denominator > 0:
            rate = accepts / current_denominator
            print(f"   Token {i+1}: {rate*100:.2f}% ({int(accepts)}/{int(current_denominator)})")
        else:
            print(f"   Token {i+1}: 0.00% (0/0)")
        # 核心逻辑：下一个位置的条件尝试分母，等于当前位置的接受分子
        current_denominator = accepts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--mar-model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--show-first-sample", action="store_true", help="打印第一条数据的投机解码详细过程并保存到日志")
    args = parser.parse_args()
    run_benchmark(args)