import re
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_scores(file_path):
    scores = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r'Score:\s*(\d+)', line)
                if match:
                    scores.append(int(match.group(1)))
    except Exception as e:
        print(f"讀取 {file_path} 發生錯誤: {e}")
    return scores

if __name__ == "__main__":
    file_4b = '/Users/harry/Desktop/VLM/VLM_EXP/gemma3-4b_judge.log'
    file_12b = '/Users/harry/Desktop/VLM/VLM_EXP/gemma3-12b_judge.log'

    scores_4b = parse_scores(file_4b)
    scores_12b = parse_scores(file_12b)
    
    if len(scores_4b) == 0 and len(scores_12b) == 0:
        print("未找到任何分數資料，請確認 log 檔案是否存在且格式正確。")
        exit(1)

    print(f"=== 分數統計 ===")
    if scores_4b:
        print(f"Gemma-3-4B  | 數量: {len(scores_4b)} | 平均分數: {np.mean(scores_4b):.2f}")
    if scores_12b:
        print(f"Gemma-3-12B | 數量: {len(scores_12b)} | 平均分數: {np.mean(scores_12b):.2f}")

    # 繪製直方圖
    bins = np.arange(-0.5, 11.5, 1) # 設定 bin 的範圍讓柱子置中於 0~10 的數字上方

    plt.figure(figsize=(10, 6))
    
    # 將兩組資料傳入 plt.hist 即可自動並排顯示
    data_to_plot = []
    labels = []
    colors = []
    
    if scores_4b:
        data_to_plot.append(scores_4b)
        labels.append('Gemma-3 4B')
        colors.append('skyblue')
    if scores_12b:
        data_to_plot.append(scores_12b)
        labels.append('Gemma-3 12B')
        colors.append('salmon')

    plt.hist(data_to_plot, bins=bins, label=labels, color=colors, edgecolor='black', align='mid')

    plt.xlabel('Score')
    plt.ylabel('Frequency (Count)')
    plt.title('LLM-as-a-Judge Score Comparison: Gemma-3 4B vs 12B')
    plt.xticks(range(0, 11))
    plt.legend()
    plt.grid(axis='y', alpha=0.75)

    output_file = '/Users/harry/Desktop/VLM/VLM_EXP/score_comparison.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n=> 比較長條圖已儲存至: {output_file}")
