# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
#     "python-dotenv",
# ]
# ///

import os
import re
import logging
from dotenv import load_dotenv
from openai import OpenAI

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("llm_judge_results.log", encoding="utf-8", mode='w'),
        logging.StreamHandler()
    ]
)

def parse_log_file(log_path):
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
            # 確保乾淨的回答文字
            clean_answer = answer_match.group(1).strip()
            
            parsed_items.append({
                "video": video_path,
                "answer": clean_answer
            })
            
    return parsed_items

def evaluate_with_gpt5(client, answer):
    judge_prompt = f"""
You are an expert, impartial evaluator for Vision-Language Models (VLMs). 
A VLM was asked to describe what is happening in a video.

[VLM's Description]
"{answer}"

Please critically evaluate this description based on the following criteria:
1. Is the description concise, coherent, and logically sound?
2. Does it accurately portray a plausible and realistic action/event (avoiding hallucinations)?

Weigh strictly. Provide your evaluation in the following exact format:
Score: [0-10]
Reason: [Your brief explanation]
"""
    
    try:
        response = client.chat.completions.create(
            # 如果 gpt-5 模型名稱報錯，請換成 gpt-4o 或您的帳號有權限的最新模型
            model="gpt-5", 
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Judge 評估失敗: {e}"

def main():
    # 載入 .env 檔案與 OpenAI 金鑰
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("❌ 找不到 OPENAI_API_KEY！請確認是否有在根目錄建立 .env 檔案並寫入金鑰。")
        return
        
    client = OpenAI(api_key=api_key)
    
    # 讀取剛剛生成的 log
    log_file_path = "gemma3-4b_details.log"
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
        
        logging.info(f"[{i+1}/{len(items_to_judge)}] 正在評估影片: {video}")
        logging.info(f"🤖 VLM 回答內容: {answer}")
        
        judge_result = evaluate_with_gpt5(client, answer)
        
        logging.info(f"📝 GPT-5 評語:\n{judge_result}\n{'-'*50}")

if __name__ == "__main__":
    main()
