# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
#     "python-dotenv",
# ]
# ///

import os
import re
import csv
import logging
import argparse
from dotenv import load_dotenv
from openai import OpenAI

def setup_logging(output_filename):
    # 設定日誌格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(output_filename, encoding="utf-8", mode='w'),
            logging.StreamHandler()
        ]
    )

def parse_log_file(log_path):
    """
    從 gemma3-4b_details.log 中解析出每支影片及其對應的模型回答
    並根據路徑中的資料夾名稱作為 ground truth
    """
    if not os.path.exists(log_path):
        logging.error(f"找不到日誌檔案: {log_path}")
        return []

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按照分隔線切出每一筆紀錄
    blocks = content.split("------------------------------")
    
    parsed_items = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # 使用正則表達式提取影片路徑和回答內容
        video_match = re.search(r"影片:\s*(.+)", block)
        answer_match = re.search(r"回答:\s*([\s\S]+)", block) # \s\S 匹配包含換行的所有內容
        
        if video_match and answer_match:
            video_path = video_match.group(1).strip()
            clean_answer = answer_match.group(1).strip()
            
            # 從檔案路徑解析資料夾名稱做為 label
            # 例如: three_classes/falling_off_chair/xxx.mp4 -> falling_off_chair
            # 取得影片檔案所在的上一層資料夾名稱
            label = os.path.basename(os.path.dirname(video_path))
            
            # 若路徑沒有包含資料夾（或者與當前目錄相同），則顯示不明
            if not label or label == "." or label == "..":
                label = "Unknown Action"
            else:
                # 將底線替換為空白，以符合一般 ground truth 的格式
                label = label.replace("_", " ")
            
            parsed_items.append({
                "video": video_path,
                "answer": clean_answer,
                "label": label
            })
            
    return parsed_items

def evaluate_with_gpt5(client, answer, label):
    judge_prompt = f"""
You are an expert, impartial evaluator for Vision-Language Models (VLMs).
A VLM was asked to "Describe the main action accurately in under 10 words" for a video.

[Ground Truth Action]
The true action happening in the video is: {label}

[VLM's Description]
"{answer}"

Please critically evaluate this description based on the following criteria:
1. Did the VLM successfully identify the core action ({label}) or relevant action?
2. Did the VLM follow the length constraint (under 10 words)?
3. Is it free from irrelevant background details or hallucinations?

Weigh strictly. Provide your evaluation in the following exact format:
Score: [0-10]
Reason: [Your brief explanation, max 20 words]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-5", 
            messages=[{"role": "user", "content": judge_prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Judge 評估失敗: {e}"

def main():
    parser = argparse.ArgumentParser(description="Use GPT-5 to judge VLM results from a log file.")
    parser.add_argument("--video_dir", type=str, default="three_classes", help="Directory containing mp4/mkv files (used to infer ground truth name)")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-4b-it", help="Hugging Face model ID")
    args = parser.parse_args()

    # 從 args 解析出和 main.py 一致的 log 目錄與檔名
    clean_video_dir = os.path.normpath(args.video_dir)
    ground_truth_name = os.path.basename(clean_video_dir)
    if not ground_truth_name or ground_truth_name == ".":
        ground_truth_name = "default_ground_truth"
        
    model_name = args.model_id.split("/")[-1]
    
    # 預期要讀取的 log 檔案路徑
    log_file_path = os.path.join(model_name, f"{ground_truth_name}.log")

    # 自動根據輸入檔名命名輸出的判斷結果 Log
    output_log_file = os.path.join(model_name, f"{ground_truth_name}_judge_results.log")
    setup_logging(output_log_file)
    
    # 載入 .env 檔案與 OpenAI 金鑰，強制覆蓋以免吃到系統舊變數
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("❌ 找不到 OPENAI_API_KEY！請確認是否有在根目錄建立 .env 檔案並寫入金鑰。")
        return
        
    client = OpenAI(api_key=api_key)
    
    # 讀取指定的 log 檔案
    logging.info(f"開始解析紀錄檔: {log_file_path}")
    
    items_to_judge = parse_log_file(log_file_path)
    logging.info(f"成功解析出 {len(items_to_judge)} 筆待評估紀錄。")
    
    if not items_to_judge:
        logging.warning("沒有找到任何可評估的內容。")
        return
        
    logging.info("\n🚀 開始執行 LLM-as-a-Judge 評估...\n")
    
    for i, item in enumerate(items_to_judge):
        video = item['video']
        answer = item['answer']
        label = item['label']
        
        logging.info(f"[{i+1}/{len(items_to_judge)}] 正在評估影片: {video} (Ground Truth: {label})")
        logging.info(f"🤖 VLM 回答內容: {answer}")
        
        judge_result = evaluate_with_gpt5(client, answer, label)
        
        logging.info(f"📝 GPT-5 評語:\n{judge_result}\n{'-'*50}")

if __name__ == "__main__":
    main()
