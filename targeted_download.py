import pandas as pd
import subprocess
import os

def download_targeted_kinetics(csv_path, target_labels, output_dir, limit=None):
    os.makedirs(output_dir, exist_ok=True)    
    print("正在讀取 CSV 目錄...")
    df = pd.read_csv(csv_path)
    filtered_df = df[df['label'].isin(target_labels)]
    
    if limit is not None:
        filtered_df = filtered_df.groupby('label').head(limit)
        
    total_videos = len(filtered_df)
    print(f"找到 {total_videos} 支符合條件的影片，準備下載...")

    # 逐一處理並下載
    for index, row in enumerate(filtered_df.itertuples(), 1):
        youtube_id = row.youtube_id
        time_start = int(row.time_start)
        time_end = int(row.time_end)
        label_name = row.label.replace(" ", "_") # 把空白換成底線方便存檔
        
        # 組裝 YouTube 網址與輸出檔名
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        output_filename = os.path.join(output_dir, f"{label_name}_{youtube_id}.mp4")
        
        if os.path.exists(output_filename):
            print(f"[{index}/{total_videos}] 已存在，跳過: {output_filename}")
            continue

        print(f"[{index}/{total_videos}] 正在下載 {label_name} : {youtube_id}")
        
        # 使用 yt-dlp 進行下載 (透過 subprocess 呼叫)
        # --download-sections 用來只下載 CSV 中標註的特定時間段 (非常重要，省流量)
        command = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4', # 指定 mp4 格式
            '--download-sections', f'*{time_start}-{time_end}',
            '-o', output_filename,
            url
        ]
        
        try:
            # 執行系統指令
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  -> 下載失敗 (可能影片已遭 YouTube 下架): {url}")
        except FileNotFoundError:
             print("  -> 找不到 yt-dlp 指令，請確保已透過 pip 安裝。")
             break

# 定義我們精心挑選的高價值標籤
target_classes = [
    'falling off chair', 
    'falling off bike', 
    'faceplanting',
    # 'climbing ladder',
    # 'walking with crutches',
]

if __name__ == "__main__":
    # 執行函式 (請確保路徑正確)
    csv_file = 'k700-2020/annotations/train.csv'
    
    # 這裡我增加了一個 limit 參數，方便你先測試下載前 10 支影片
    download_targeted_kinetics(csv_file, target_classes, 'kinetics_targeted_dataset', limit=50)
