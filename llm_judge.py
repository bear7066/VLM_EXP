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

def load_ground_truth(csv_path):
    ground_truth = {}
    if not os.path.exists(csv_path):
        logging.error(f"找不到 CSV 檔案: {csv_path}")
        return ground_truth
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "youtube_id" in row and "label" in row:
                ground_truth[row["youtube_id"]] = row["label"]
    return ground_truth

def parse_log_file(log_path, ground_truth):
    """
    從 gemma3-4b_details.log 中解析出每支影片及其對應的模型回答
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
            
            # 從檔名解析 youtube_id (處理 ID 本身包含 '_' 的情況)
            basename = os.path.basename(video_path) # e.g. -1B_EOupmCE_000076_000086.mp4
            name_without_ext = os.path.splitext(basename)[0] # -1B_EOupmCE_000076_000086
            # 使用 rsplit 切割最後兩個時間戳記
            parts = name_without_ext.rsplit('_', 2)
            youtube_id = parts[0]
            label = ground_truth.get(youtube_id, "Unknown Action")
            
            parsed_items.append({
                "video": video_path,
                "answer": clean_answer,
                "label": label
            })
            
    return parsed_items

def evaluate_with_gpt5(client, answer, label):
    judge_prompt = f"""
You are an expert, impartial evaluator for Vision-Language Models (VLMs). 
A VLM was asked to describe what is happening in a video.

[Ground Truth Action]
The true action happening in the video is: {label}

[VLM's Description]
"{answer}"

Please critically evaluate this description based on the following criteria:
1. Is the description concise, coherent, and logically sound?
2. Does it accurately identify and describe the ground truth action ({label})? Does it avoid hallucinations?
3. Answer with concise and clear explanation(less than 20 words).

Weigh strictly. Provide your evaluation in the following exact format:
Score: [0-10]
Reason: [Your brief explanation]
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
    parser.add_argument("log_file", type=str, nargs="?", default="gemma3-4b_details.log",
                        help="Path to the .log file to evaluate (e.g. gemma3-4b_details.log)")
    parser.add_argument("--csv_file", type=str, default="gp.csv",
                        help="Path to the ground truth CSV file (e.g. gp.csv)")
    args = parser.parse_args()

    # 自動根據輸入檔名命名輸出的判斷結果 Log
    log_name = os.path.splitext(os.path.basename(args.log_file))[0]
    output_log_file = f"{log_name}_judge_results.log"
    setup_logging(output_log_file)
    
    # 載入 .env 檔案與 OpenAI 金鑰，強制覆蓋以免吃到系統舊變數
    load_dotenv(override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("❌ 找不到 OPENAI_API_KEY！請確認是否有在根目錄建立 .env 檔案並寫入金鑰。")
        return
        
    client = OpenAI(api_key=api_key)
    
    # 讀取 CSV
    csv_file_path = args.csv_file
    ground_truth = load_ground_truth(csv_file_path)
    
    # 讀取指定的 log 檔案
    log_file_path = args.log_file
    logging.info(f"開始解析紀錄檔: {log_file_path}")
    
    items_to_judge = parse_log_file(log_file_path, ground_truth)
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
