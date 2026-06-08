import os

import matplotlib.pyplot as plt
import pandas as pd


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Data file not found: {filepath}")
        return None
    # 讀取 CSV，設定 Date 為 index
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

def plot_normalized_trends(df, output_dir):
    """繪製 24/7 全球指標的正規化趨勢圖"""
    # 擷取 Close 價格
    close_cols = [col for col in df.columns if col.endswith('_Close')]
    df_close = df[close_cols].copy()
    
    # 重新命名欄位以便圖表顯示
    df_close.columns = [col.replace('_Close', '') for col in df_close.columns]
    
    # Z-score 正規化，以便放在同一張圖表上比較趨勢
    df_norm = (df_close - df_close.mean()) / df_close.std()
    
    plt.figure(figsize=(14, 7))
    for col in df_norm.columns:
        plt.plot(df_norm.index, df_norm[col], label=col, linewidth=2, alpha=0.8)
    
    plt.title('24/7 Global Macro Indicators Trend (Z-Score Normalized)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Value (Z-Score)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'macro_trend_overlay.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Trend plot saved to {output_path}")

def plot_correlation_heatmap(df, output_dir):
    """繪製 24/7 全球指標的相關性熱力圖"""
    close_cols = [col for col in df.columns if col.endswith('_Close')]
    df_close = df[close_cols].copy()
    df_close.columns = [col.replace('_Close', '') for col in df_close.columns]
    
    # 計算日報酬率
    df_returns = df_close.pct_change().dropna()
    
    # 計算皮爾森相關係數
    corr = df_returns.corr()
    
    plt.figure(figsize=(10, 8))
    # 簡單的手刻熱力圖 (因為不一定有安裝 seaborn)
    plt.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation')
    
    # 加入文字數值
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", 
                     ha="center", va="center", color="black" if abs(corr.values[i,j]) < 0.5 else "white")
    
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Global Macro Cross-Market Correlation (Daily Returns)', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'macro_correlation_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(base_dir, 'data', 'global_macro_data_1d.csv')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    
    df = load_data(data_file)
    if df is not None and not df.empty:
        print("Generating visualizations...")
        plot_normalized_trends(df, reports_dir)
        plot_correlation_heatmap(df, reports_dir)
        print("Done!")
    else:
        print("Please run global_macro_loader.py first to generate data.")
