import torch
import torch.nn.functional as F
import time
import json

# Load config
with open("/mnt/zhengcf3/lmp/src/models/Gemma/config.json", "r") as f:
    config = json.load(f)

text_config = config["text_config"]
hidden_size = text_config["hidden_size"]  # 2816
num_experts = text_config["num_experts"]  # 128
top_k_experts = text_config["top_k_experts"]  # 8
moe_intermediate_size = text_config["moe_intermediate_size"]  # 704

# Test parameters
batch_size = 4
seq_len = 256
num_tokens = batch_size * seq_len

# Multi-device settings
num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"Number of available GPUs: {num_devices}")
num_devices = min(num_devices, 4)  # Use up to 4 devices
experts_per_device = num_experts // num_devices

print(f"Config: hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k_experts}, moe_intermediate_size={moe_intermediate_size}")
print(f"Test params: batch_size={batch_size}, seq_len={seq_len}, num_tokens={num_tokens}")
print(f"Multi-device: {num_devices} devices, {experts_per_device} experts per device")


def prepare_data(num_tokens, hidden_size, num_experts, top_k_experts):
    """Prepare test data"""
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size).cuda()
    
    # Generate expert weights - split across devices
    gate_up_weight = []
    down_weight = []
    for d in range(num_devices):
        start = d * experts_per_device
        end = (d + 1) * experts_per_device if d < num_devices - 1 else num_experts
        local_num_experts = end - start
        
        gu_w = torch.randn(local_num_experts, hidden_size, 2 * moe_intermediate_size).cuda(d)
        d_w = torch.randn(local_num_experts, moe_intermediate_size, hidden_size).cuda(d)
        
        gate_up_weight.append(gu_w)
        down_weight.append(d_w)
    
    # Generate random expert assignments (top-k)
    expert_ids = torch.randint(0, num_experts, (num_tokens, top_k_experts)).cuda()
    gate_scores = torch.rand(num_tokens, top_k_experts).cuda()
    gate_scores = gate_scores.softmax(dim=-1)
    
    return x, gate_up_weight, down_weight, expert_ids, gate_scores


def run_single_device_non_graph(x, expert_ids, gate_scores, gate_up_weight, down_weight, 
                                device_id=0, num_runs=10):
    """Run MOE on single device without torch.compile"""
    start_idx = device_id * experts_per_device
    end_idx = (device_id + 1) * experts_per_device if device_id < num_devices - 1 else num_experts
    
    start = time.perf_counter()
    for _ in range(num_runs):
        expert_cache = torch.zeros(num_tokens, hidden_size).cuda(device_id)
        
        for k in range(top_k_experts):
            eids = expert_ids[:, k]
            scores = gate_scores[:, k].unsqueeze(-1)
            
            # Filter tokens assigned to this device's experts
            in_range = (eids >= start_idx) & (eids < end_idx)
            if not in_range.any():
                continue
            
            local_eids = eids[in_range] - start_idx
            local_x = x[in_range].to(device_id)
            local_scores = scores[in_range].to(device_id)
            
            # One-hot encoding
            eid_onehot = F.one_hot(local_eids, num_classes=experts_per_device).float()
            
            # Expand inputs
            x_expanded = local_x.unsqueeze(1).expand(-1, experts_per_device, hidden_size)
            
            # Gate-up projection
            gate_up_out = torch.einsum('teh,ehi->tei', x_expanded, gate_up_weight[device_id])
            
            # Split and activate
            gate_out = gate_up_out[:, :, :moe_intermediate_size]
            up_out = gate_up_out[:, :, moe_intermediate_size:]
            up_out = F.gelu(up_out)
            gate_out = torch.sigmoid(gate_out)
            hidden = gate_out * up_out
            
            # Down projection
            down_out = torch.einsum('tei,eih->teh', hidden, down_weight[device_id])
            
            # Apply mask and scores
            masked_out = down_out * eid_onehot.unsqueeze(-1)
            masked_out = masked_out * local_scores.unsqueeze(-1)
            
            # Accumulate
            expert_cache[in_range] += masked_out.sum(dim=1)
    
    torch.cuda.synchronize(device_id)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_runs
    return avg_time, expert_cache


def run_multi_device_non_graph(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs=10):
    """Run MOE with multiple devices without torch.compile"""
    start = time.perf_counter()
    
    # Pre-move data to each device
    x_devices = [x.to(d) for d in range(num_devices)]
    expert_ids_devices = [expert_ids.to(d) for d in range(num_devices)]
    gate_scores_devices = [gate_scores.to(d) for d in range(num_devices)]
    
    for _ in range(num_runs):
        results = []
        
        # Process each device
        for d in range(num_devices):
            start_idx = d * experts_per_device
            end_idx = (d + 1) * experts_per_device if d < num_devices - 1 else num_experts
            
            device_cache = torch.zeros(num_tokens, hidden_size, device=f'cuda:{d}')
            local_x = x_devices[d]
            local_eids_all = expert_ids_devices[d]
            local_scores_all = gate_scores_devices[d]
            
            for k in range(top_k_experts):
                eids = local_eids_all[:, k]
                scores = local_scores_all[:, k].unsqueeze(-1)
                
                in_range = (eids >= start_idx) & (eids < end_idx)
                if not in_range.any():
                    continue
                
                local_eids = eids[in_range] - start_idx
                selected_x = local_x[in_range]
                selected_scores = scores[in_range]
                
                eid_onehot = F.one_hot(local_eids, num_classes=experts_per_device).float()
                x_expanded = selected_x.unsqueeze(1).expand(-1, experts_per_device, hidden_size)
                
                gate_up_out = torch.einsum('teh,ehi->tei', x_expanded, gate_up_weight[d])
                
                gate_out = gate_up_out[:, :, :moe_intermediate_size]
                up_out = gate_up_out[:, :, moe_intermediate_size:]
                up_out = F.gelu(up_out)
                gate_out = torch.sigmoid(gate_out)
                hidden = gate_out * up_out
                
                down_out = torch.einsum('tei,eih->teh', hidden, down_weight[d])
                
                masked_out = down_out * eid_onehot.unsqueeze(-1)
                masked_out = masked_out * selected_scores.unsqueeze(-1)
                
                device_cache[in_range] += masked_out.sum(dim=1)
            
            results.append(device_cache)
        
        # Aggregate results on main device (device 0)
        final_result = torch.zeros(num_tokens, hidden_size, device='cuda:0')
        for d in range(num_devices):
            final_result += results[d].to(0)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_runs
    print(f"Multi-device non-graph mode: {avg_time*1000:.2f} ms per iteration")
    return avg_time


def run_multi_device_graph(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs=10):
    """Run MOE with multiple devices with torch.compile"""
    
    # Create compiled functions for each device
    compiled_funcs = []
    for d in range(num_devices):
        start_idx = d * experts_per_device
        end_idx = (d + 1) * experts_per_device if d < num_devices - 1 else num_experts
        local_gate_up = gate_up_weight[d]
        local_down = down_weight[d]
        
        @torch.compile(fullgraph=True)
        def compiled_moe_device(x_local, eids_local, scores_local, gate_up, down, 
                                start_idx=start_idx, end_idx=end_idx, device=d):
            device_cache = torch.zeros(num_tokens, hidden_size, device=device)
            
            for k in range(top_k_experts):
                eids = eids_local[:, k]
                scores = scores_local[:, k].unsqueeze(-1)
                
                # Use mask instead of if statement
                in_range = (eids >= start_idx) & (eids < end_idx)
                mask_float = in_range.float().unsqueeze(-1)
                
                local_eids = (eids - start_idx) * in_range.long()
                local_x = x_local.to(device)
                
                # One-hot encoding (with zero padding for out-of-range)
                eid_onehot = F.one_hot(local_eids, num_classes=experts_per_device).float()
                
                # Only process tokens in range
                x_expanded = local_x.unsqueeze(1).expand(-1, experts_per_device, hidden_size)
                
                gate_up_out = torch.einsum('teh,ehi->tei', x_expanded, gate_up)
                
                gate_out = gate_up_out[:, :, :moe_intermediate_size]
                up_out = gate_up_out[:, :, moe_intermediate_size:]
                up_out = F.gelu(up_out)
                gate_out = torch.sigmoid(gate_out)
                hidden = gate_out * up_out
                
                down_out = torch.einsum('tei,eih->teh', hidden, down)
                
                masked_out = down_out * eid_onehot.unsqueeze(-1)
                masked_out = masked_out * scores.to(device).unsqueeze(-1)
                masked_out = masked_out * mask_float
                
                device_cache = device_cache + masked_out.sum(dim=1)
            
            return device_cache
        
        compiled_funcs.append(compiled_moe_device)
    
    # Warm up
    for d in range(num_devices):
        _ = compiled_funcs[d](x, expert_ids, gate_scores, gate_up_weight[d], down_weight[d])
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        results = []
        for d in range(num_devices):
            result = compiled_funcs[d](x, expert_ids, gate_scores, gate_up_weight[d], down_weight[d])
            results.append(result)
        
        # Aggregate
        final_result = torch.zeros(num_tokens, hidden_size).cuda()
        for d in range(num_devices):
            final_result += results[d].to(0)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_runs
    print(f"Multi-device graph mode (compiled): {avg_time*1000:.2f} ms per iteration")
    return avg_time


def run_multi_device_batched_bmm(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs=10):
    """Run MOE with multiple devices using batched BMM - similar to reference code"""
    
    @torch.compile(fullgraph=True)
    def compiled_multi_device_bmm(x, expert_ids, gate_scores, gate_up_weight, down_weight):
        final_result = torch.zeros(num_tokens, hidden_size, device='cuda:0')
        
        for d in range(num_devices):
            start_idx = d * experts_per_device
            end_idx = (d + 1) * experts_per_device if d < num_devices - 1 else num_experts
            device_id = f'cuda:{d}'
            
            device_cache = torch.zeros(num_tokens, hidden_size, device=device_id)
            local_gate_up = gate_up_weight[d]
            local_down = down_weight[d]
            
            for k in range(top_k_experts):
                eids = expert_ids[:, k]
                scores = gate_scores[:, k]
                
                # Filter tokens for this device
                in_range = (eids >= start_idx) & (eids < end_idx)
                if not in_range.any():
                    continue
                
                # Get tokens assigned to this device
                device_tokens = torch.where(in_range)[0]
                local_eids = eids[in_range] - start_idx
                local_x = x[in_range].to(device_id)
                local_scores = scores[in_range].to(device_id)
                
                # Sort by expert
                sorted_indices = local_eids.argsort()
                eids_sorted = local_eids[sorted_indices]
                x_sorted = local_x[sorted_indices]
                scores_sorted = local_scores[sorted_indices]
                
                # Get unique experts
                unique_eids, inverse_indices, counts = torch.unique(eids_sorted, return_inverse=True, return_counts=True)
                num_active = unique_eids.size(0)
                max_count = counts.max()
                
                # Create batched inputs
                batched_inputs = torch.zeros(num_active, max_count, hidden_size, device=device_id)
                batch_indices = torch.arange(device_tokens.size(0), device=device_id) - torch.cumsum(torch.nn.functional.pad(counts, (1, 0)), dim=0)[inverse_indices]
                batched_inputs[inverse_indices, batch_indices] = x_sorted
                
                # Get weights
                w_gu = local_gate_up[unique_eids]
                w_down = local_down[unique_eids]
                
                # BMM operations
                gate_up_out = torch.bmm(batched_inputs, w_gu)
                gate_out = gate_up_out[:, :, :moe_intermediate_size]
                up_out = gate_up_out[:, :, moe_intermediate_size:]
                up_out = F.gelu(up_out)
                gate_out = torch.sigmoid(gate_out)
                hidden = gate_out * up_out
                
                down_out = torch.bmm(hidden, w_down)
                
                # Gather results
                result = down_out[inverse_indices, batch_indices] * scores_sorted.unsqueeze(-1)
                
                # Scatter to output
                device_cache[device_tokens.to(device_id)] += result
            
            final_result += device_cache.to('cuda:0')
        
        return final_result
    
    # Warm up
    _ = compiled_multi_device_bmm(x, expert_ids, gate_scores, gate_up_weight, down_weight)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = compiled_multi_device_bmm(x, expert_ids, gate_scores, gate_up_weight, down_weight)
    torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_runs
    print(f"Multi-device batched BMM mode (compiled): {avg_time*1000:.2f} ms per iteration")
    return avg_time


if __name__ == "__main__":
    print("=" * 70)
    print("MOE Multi-Device BMM Performance Comparison")
    print("=" * 70)
    
    # Prepare data
    x, gate_up_weight, down_weight, expert_ids, gate_scores = prepare_data(
        num_tokens, hidden_size, num_experts, top_k_experts
    )
    
    print(f"\nMemory usage per device:")
    for d in range(num_devices):
        gu_size = gate_up_weight[d].numel() * 2 / 1024**2
        d_size = down_weight[d].numel() * 2 / 1024**2
        print(f"  Device {d}: gate_up={gu_size:.2f} MB, down={d_size:.2f} MB")
    
    # Run tests
    num_runs = 10
    
    print("\nRunning multi-device non-graph mode...")
    non_graph_time = run_multi_device_non_graph(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs)
    
    print("\nRunning multi-device graph mode...")
    graph_time = run_multi_device_graph(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs)
    
    print("\nRunning multi-device batched BMM mode...")
    batched_time = run_multi_device_batched_bmm(x, expert_ids, gate_scores, gate_up_weight, down_weight, num_runs)
    
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"Multi-device non-graph: {non_graph_time*1000:.2f} ms")
    print(f"Multi-device graph: {graph_time*1000:.2f} ms")
    print(f"Multi-device batched BMM: {batched_time*1000:.2f} ms")
    print(f"\nGraph speedup vs non-graph: {non_graph_time/graph_time:.2f}x")
    print(f"Batched BMM speedup vs non-graph: {non_graph_time/batched_time:.2f}x")
    print(f"Batched BMM speedup vs graph: {graph_time/batched_time:.2f}x")