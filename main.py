import os
import random
import glob
import time
import logging
import argparse

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import decord
import numpy as np
from PIL import Image


print("torch version =", torch.__version__)
print("cuda available =", torch.cuda.is_available())
print("torch cuda version =", torch.version.cuda)

def sample_frames(video_path, num_frames=8):
    """
    使用 decord 讀取影片並均勻抽幀
    """
    try:
        # decord 預設為 cpu 讀取，也可以指定 ctx=decord.gpu(0) 如果有需要
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as e:
        logging.error(f"⚠️ 無法讀取影片 {video_path}: {e}")
        return None
    
    total_frames = len(vr)
    if total_frames == 0:
        return None
    
    # 均勻取樣 num_frames 個幀
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    
    # 轉換為 PIL Image 以符合 Transformers 預設處理器要求
    pil_frames = [Image.fromarray(f) for f in frames]
    return pil_frames

def main():
    parser = argparse.ArgumentParser(description="Run VLM inference on videos")
    parser.add_argument("--video_dir", type=str, default="./dataset/climbing_stair", help="Directory containing mp4/mkv files")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID")
    args = parser.parse_args()

    # 取得 ground_truth_name (將路徑最後的資料夾名稱視為 ground_truth)
    # 例如：傳入 "three_classes/falling_off_chair"，則 ground_truth_name 為 "falling_off_chair"
    # 若 args.video_dir 若有結尾斜線，需用 os.path.normpath 處理
    clean_video_dir = os.path.normpath(args.video_dir)
    ground_truth_name = os.path.basename(clean_video_dir)
    if not ground_truth_name or ground_truth_name == ".":
        ground_truth_name = "default_ground_truth"

    # 取得 model_id
    model_id = args.model_id
    model_name = model_id.split("/")[-1] # 取出模型名稱作為資料夾名稱

    # 建立輸出目錄: model_name
    os.makedirs(model_name, exist_ok=True)
    
    # 定義輸出的 log 檔案名稱
    # model_name/ground_truth_name.log
    main_log_file = os.path.join(model_name, f"{ground_truth_name}.log")

    # 設定 logging，同時將結果輸出到終端機與 log 檔案
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', # 簡化輸出格式，移除 asctime 避免太冗長
        handlers=[
            logging.FileHandler(main_log_file, encoding="utf-8", mode='a'),
            logging.StreamHandler()
        ]
    )
    
    hf_token = os.environ.get("HF_TOKEN")

    logging.info(f"載入模型與處理器: {model_id} ...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        # 使用 bfloat16 以節省 VRAM 並加速
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token = hf_token,
            dtype = torch.bfloat16, 
            device_map = "auto"
        )
    except Exception as e:
        logging.error(f"載入模型失敗，請確認已安裝最新版 transformers 並且有 Gemma 3 的存取權限: {e}")
        return
    
    # 尋找本地影片檔案，直接從 mp4 資料夾讀取
    video_dir = args.video_dir
    video_paths = glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True) + glob.glob(os.path.join(video_dir, "**/*.mkv"), recursive=True)
        
    logging.info(f"從 {video_dir} 尋找，共找到 {len(video_paths)} 支影片。")
    print("video amounts: ", len(video_paths))
    if len(video_paths) == 0:
        logging.error("❌ 找不到任何影片檔案，請確認下載已完成且解壓縮。")
        return
        
    # 前面您修改成隨機抽取 1 支，若要測 100 支請將此處數字改為 100
    sample_size = min(50, len(video_paths))
    sampled_videos = random.sample(video_paths, sample_size)
    logging.info(f"開始測試，隨機抽取 {sample_size} 支影片進行推論...")
    
    prompt_text = (
        "Describe the main action accurately in under 10 words."
    )
    
    results = []
    total_time = 0.0
    total_generated_tokens = 0
    successful_runs = 0
    num_sampled_frames = 8
    
    for i, v_path in enumerate(sampled_videos):
        logging.info(f"{'='*50}")
        logging.info(f"[{i+1}/{sample_size}] 處理影片: {v_path}")
        
        frames = sample_frames(v_path, num_frames=num_sampled_frames)
        if frames is None:
#	    logging.warning(f"⏩ 影片讀取失敗，可能是檔案損毀或下載不完全 (例如 missing moov atom) ➜ 略過: {v_path}")
            continue
            
        # 建立 Gemma 3 多模態對話格式所需的 Message 結構
        # 將 num_sampled_frames 張圖片傳入
        content_items = [{"type": "image"} for _ in range(len(frames))]
        content_items.append({"type": "text", "text": prompt_text})
        
        messages = [
            {
                "role": "user",
                "content": content_items
            }
        ]
        
        try:
            # 使用 Chat Template 建立正確的 Prompt 格式
            formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            # 使用 Device Map 將 Input 自動放到適合的 GPU/CPU
            inputs = processor(
                text=formatted_prompt, 
                images=frames, 
                return_tensors="pt"
            ).to(model.device)
            
            # 將 bfloat16 用於 pixel_values（如果可用）以避免型別錯誤
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            logging.info("開始生成回答...")
            start_time = time.time()
            
            # 模型推論
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False, # 設為 False 進行 Greedy Decode 確保測試的穩定性
                    temperature=None,
                    top_p=None
                )
                
            # 去除輸入 prompt 部分的 Token
            generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True)
            
            end_time = time.time()
            elapsed = end_time - start_time
            num_generated_tokens = len(generated_ids)
            tps = num_generated_tokens / elapsed if elapsed > 0 else 0
            
            logging.info(f"⏱️ 耗時: {elapsed:.2f} 秒")
            logging.info(f"⚡ 速度: {tps:.2f} tokens/sec ({num_generated_tokens} tokens)")
            logging.info(f"🤖 模型回答:\n{response.strip()}")
            logging.info("-" * 30) # 新增分隔線，取代原本寫入 details log 的動作
                
            results.append({
                "video": v_path,
                "time": elapsed,
                "tokens": num_generated_tokens,
                "tps": tps,
                "response": response.strip()
            })
            
            total_time += elapsed
            total_generated_tokens += num_generated_tokens
            successful_runs += 1
            
        except Exception as e:
            logging.error(f"❌ 推論過程中發生錯誤: {e}")
            
    # 統計並輸出最終結果
    if successful_runs > 0:
        avg_time = total_time / successful_runs
        avg_tps = total_generated_tokens / total_time if total_time > 0 else 0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        logging.info(f"\n{'='*20} 測試總結 {'='*20}")
        logging.info(f"成功處理影片數  : {successful_runs} / {sample_size}")
        logging.info(f"總耗費時間      : {total_time:.2f} 秒")
        logging.info(f"總生成 Tokens 數: {total_generated_tokens}")
        logging.info(f"平均每支影片耗時: {avg_time:.2f} 秒")
        logging.info(f"整體生成速度    : {avg_tps:.2f} tokens/sec")
        if torch.cuda.is_available():
            logging.info(f"GPU 最高記憶體佔用: {peak_vram:.2f} GB")
        logging.info(f"詳細測試結果已隨時記錄於 {main_log_file}")

if __name__ == "__main__":
    main()
