import numpy as np


def calculate_metrics(
    portfolio_history,
    positions_history,
    cash_history,
    daily_returns,
    turnover_history=None,
    tickers=None
):
    """
    計算標準化的策略績效與分析指標。
    
    Parameters
    ----------
    portfolio_history : list or np.ndarray
        每日組合總資產歷史 (長度 T)
    positions_history : list or np.ndarray
        每日股票部位權重 (形狀 T-1 x num_stocks)
    cash_history : list or np.ndarray
        每日現金權重 (長度 T-1)
    daily_returns : list or np.ndarray
        每日組合報酬率 (長度 T-1)
    turnover_history : list or np.ndarray, optional
        每日換手率 (長度 T-1)
    tickers : list of str, optional
        股票代號列表
        
    Returns
    -------
    dict
        包含各種評估指標的字典
    """
    portfolio_history = np.array(portfolio_history)
    daily_returns = np.array(daily_returns)
    cash_history = np.array(cash_history) if cash_history else np.zeros_like(daily_returns)
    
    # 基本報酬與回撤
    initial_val = portfolio_history[0]
    final_val = portfolio_history[-1]
    total_return = (final_val / initial_val) - 1.0
    
    peak = portfolio_history[0]
    max_drawdown = 0.0
    for val in portfolio_history:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_drawdown:
            max_drawdown = dd
            
    # 風險調整後報酬 (年化)
    # 假設一年 252 個交易日
    mean_ret = np.mean(daily_returns) if len(daily_returns) > 0 else 0.0
    std_ret = np.std(daily_returns) if len(daily_returns) > 0 else 1e-8
    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252)
    
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-8
    sortino = (mean_ret / (downside_std + 1e-8)) * np.sqrt(252)
    
    # 勝率
    win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0.0
    
    # 現金與換手率
    avg_cash_weight = np.mean(cash_history) if len(cash_history) > 0 else 0.0
    avg_turnover = np.mean(turnover_history) if turnover_history else 0.0
    
    # 多空曝險比例 (Long/Short Exposure)
    long_exposure = 0.0
    short_exposure = 0.0
    short_history = np.zeros_like(cash_history)
    
    if positions_history:
        pos_matrix = np.array(positions_history)
        long_sums = np.sum(np.maximum(pos_matrix, 0), axis=1)
        short_sums = np.sum(np.abs(np.minimum(pos_matrix, 0)), axis=1)
        long_exposure = float(np.mean(long_sums))
        short_exposure = float(np.mean(short_sums))
        short_history = short_sums
        
    # 假避險檢測 (Fake Hedging Detection)
    cash_weight_std = np.std(cash_history) if len(cash_history) > 0 else 0.0
    
    cash_corr_next_return = 0.0
    short_corr_next_return = 0.0
    
    if len(cash_history) > 1 and len(daily_returns) > 1:
        cash_t = cash_history[:-1]
        ret_t_plus_1 = daily_returns[1:]
        if np.std(cash_t) > 1e-6 and np.std(ret_t_plus_1) > 1e-6:
            corr = np.corrcoef(cash_t, ret_t_plus_1)[0, 1]
            cash_corr_next_return = float(corr) if not np.isnan(corr) else 0.0
            
        short_t = short_history[:-1]
        if np.std(short_t) > 1e-6 and np.std(ret_t_plus_1) > 1e-6:
            corr_s = np.corrcoef(short_t, ret_t_plus_1)[0, 1]
            short_corr_next_return = float(corr_s) if not np.isnan(corr_s) else 0.0
            
    # 前幾大持股 (平均權重)
    top_holdings = {}
    if positions_history and tickers:
        pos_matrix = np.array(positions_history)
        avg_weights = np.mean(pos_matrix, axis=0)
        top_indices = np.argsort(avg_weights)[-5:][::-1] # 取前五大
        for idx in top_indices:
            if avg_weights[idx] > 0.01: # 平均權重 > 1% 才列入
                top_holdings[tickers[idx]] = float(avg_weights[idx])
                
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(win_rate),
        "avg_cash_weight": float(avg_cash_weight),
        "long_exposure": float(long_exposure),
        "short_exposure": float(short_exposure),
        "cash_weight_std": float(cash_weight_std),
        "cash_corr_next_return": float(cash_corr_next_return),
        "short_corr_next_return": float(short_corr_next_return),
        "turnover": float(avg_turnover),
        "top_holdings": top_holdings
    }
