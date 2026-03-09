# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "pillow",
#     "huggingface-hub",
#     "numpy",
# ]
# ///

"""
VLM 推論速度基準測試 (Benchmark)
測量：每張圖片推論耗時、每秒可處理幾張 Frame (FPS)
"""

import torch
import time
import argparse
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login

login()

def make_dummy_frame(width=640, height=360):
    """產生一張隨機 RGB 測試圖片 (不需要真實影片)"""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def run_benchmark(model_id, num_frames_per_query, num_queries, prompt_text):
    print(f"\n{'='*60}")
    print(f"模型: {model_id}")
    print(f"每次查詢的 Frame 數: {num_frames_per_query}")
    print(f"執行次數: {num_queries}")
    print(f"Prompt: {prompt_text}")
    print(f"{'='*60}\n")

    print("載入模型中...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()

    # 預先生成好假的測試圖片 (避免讓圖片生成影響計時)
    dummy_frames = [make_dummy_frame() for _ in range(num_frames_per_query)]

    # 準備 Message 結構
    content_items = [{"type": "image"} for _ in range(num_frames_per_query)]
    content_items.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content_items}]

    # 準備 Prompt (只需要做一次)
    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # ---- 預熱 (Warmup) ---- #
    # 前幾次推論因為 GPU 暖機通常較慢，先跑一次不計入結果
    print("GPU 預熱中 (Warmup)...")
    inputs = processor(text=formatted_prompt, images=dummy_frames, return_tensors="pt").to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=None, top_p=None)
    torch.cuda.synchronize()
    print("預熱完畢，開始計時...\n")

    # ---- 正式計時 ---- #
    latencies = []  # 每次查詢的耗時 (秒)
    
    for i in range(num_queries):
        inputs = processor(text=formatted_prompt, images=dummy_frames, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        torch.cuda.synchronize()  # 確保 GPU 計算完畢再開始計時
        start = time.perf_counter()
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        torch.cuda.synchronize()
        end = time.perf_counter()

        elapsed = end - start
        latencies.append(elapsed)

        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        print(f"  [{i+1}/{num_queries}] 耗時: {elapsed:.3f}s | 回答: {response.strip()[:60]}")

    # ---- 統計結果 ---- #
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # Effective FPS: 假設每次查詢看 num_frames_per_query 張圖，
    # 以平均耗時計算「每秒能看幾張圖」
    fps = num_frames_per_query / avg_latency

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"📊 基準測試結果")
    print(f"{'='*60}")
    print(f"  平均每次查詢耗時:  {avg_latency*1000:.1f} ms")
    print(f"  最快:              {min_latency*1000:.1f} ms")
    print(f"  最慢:              {max_latency*1000:.1f} ms")
    print(f"  等效 FPS (每秒看幾張圖): {fps:.2f} FPS")
    print(f"  等效即時延遲:      {avg_latency:.2f} 秒/次查詢")
    print(f"  GPU 最高記憶體佔用: {peak_vram:.2f} GB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM 推論速度基準測試")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it",
                        help="Hugging Face model ID")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="每次查詢傳入幾張圖片 (e.g., 8, 16, 32)")
    parser.add_argument("--num_queries", type=int, default=10,
                        help="執行幾次查詢取平均值 (建議至少 5)")
    parser.add_argument("--prompt", type=str, default="Describe the main action accurately in under 10 words.",
                        help="傳給模型的 Prompt 文字")
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model_id,
        num_frames_per_query=args.num_frames,
        num_queries=args.num_queries,
        prompt_text=args.prompt
    )
