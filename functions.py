import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

# ---------------------------
# preprocess (giữ nguyên signature)
# ---------------------------
def preprocess(data, start=None, end=None):
    # Load
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("data phải là đường dẫn csv hoặc pandas.DataFrame")
    
    # Đổi tên cột
    rename_map = {'time': 'DATE', 'tmax': 'TMAX', 'tmin': 'TMIN'}
    df = df.rename(columns=rename_map)
    
    # Chuyển DATE thành datetime
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)

    # Lọc theo khoảng thời gian nếu có
    if start is not None:
        start = pd.to_datetime(start)
        df = df[df['DATE'] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        df = df[df['DATE'] <= end]
    df = df.reset_index(drop=True)

    # --- Thêm bước kiểm tra và điền ngày bị thiếu ---
    full_range = pd.date_range(df['DATE'].min(), df['DATE'].max(), freq='D')
    df_full = pd.DataFrame({'DATE': full_range})
    df = pd.merge(df_full, df, on='DATE', how='left')

    # --- Tính T ---
    df['T'] = (df['TMAX'].astype(float) + df['TMIN'].astype(float)) / 2.0

    # Hàm fill giá trị thiếu cho T (mean trong cửa sổ ±7, không dùng giá trị đang fill)
    def fill_missing_T(series):
        s = series.copy()
        nan_idx = np.where(s.isna())[0]
        for idx in nan_idx:
            start_idx = max(0, idx - 7)
            end_idx = min(len(s) - 1, idx + 7)
            window = s.iloc[start_idx:end_idx+1].drop(index=idx, errors='ignore')
            window = window[window.notna()]
            if len(window) > 0:
                s.iat[idx] = window.mean()
        return s

    df['T'] = fill_missing_T(df['T'])

    # --- Tách ngày/tháng/năm ---
    df['DAY'] = df['DATE'].dt.day
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year

    # --- Bỏ ngày 29/02 nếu có ---
    leap_mask = (df['MONTH'] == 2) & (df['DAY'] == 29)
    if leap_mask.any():
        df = df.loc[~leap_mask].reset_index(drop=True)
        
    # --- Chỉ giữ các cột cần thiết ---
    df = df[['DATE', 'TMAX', 'TMIN', 'T', 'DAY', 'MONTH', 'YEAR']]
    
    return df



# ---------------------------
# detrend_deseasonalize (giữ signature, sửa label thành LAMBDA(t))
# ---------------------------
def detrend_deseasonalize(df, T_col='T', LAMBDA_col='LAMBDA', X_col='X',
                          omega=2 * np.pi / 365, return_results=False):
    if T_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{T_col}' trong df")
    n = len(df)
    if n == 0:
        raise ValueError("DataFrame rỗng")

    # thời gian
    t = np.arange(n, dtype=float)

    # Vẽ line plot ban đầu
    plt.figure(figsize=(10, 4))
    plt.plot(t, df[T_col], label='Observed T', linestyle='-', marker='', alpha=0.8)
    plt.title('Chuỗi gốc trước khi fit')
    plt.xlabel('t')
    plt.ylabel(T_col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Ma trận thiết kế
    X = pd.DataFrame({
        'const': 1.0,
        't': t,
        'cos_wt': np.cos(omega * t),
        'sin_wt': np.sin(omega * t)
    }, index=df.index)

    # Fit OLS
    y = df[T_col].astype(float).values
    model = sm.OLS(y, X)
    results = model.fit()

    # Lưu fitted & residual
    df[LAMBDA_col] = results.predict(X)
    df[X_col] = df[T_col] - df[LAMBDA_col]

    # Vẽ scatter thực tế + line fitted với label LAMBDA(t)
    plt.figure(figsize=(10, 4))
    plt.scatter(t, df[T_col], label='Observed T', s=10, alpha=0.6)
    plt.plot(t, df[LAMBDA_col], label='Λ(t)', color='red', linewidth=2)
    plt.title('Dữ liệu thực tế (scatter) và fitted (line)')
    plt.xlabel('t')
    plt.ylabel(T_col)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # In kết quả hồi quy
    print(results.summary())

    return results if return_results else None


# ---------------------------
# ADF detailed (giữ nguyên)
# ---------------------------
def adf_detailed(y, regression='c'):
    y = y.dropna()

    # 1. Sử dụng adfuller để chọn số trễ theo AIC và lấy critical values
    result = adfuller(y, regression=regression, autolag='AIC')
    best_lag = result[2]
    critical_values = result[4]
    adf_statistic = result[0]

    print(f"\n=== ADF regression: '{regression}' | optimal lag (by AIC) = {best_lag} ===")

    # 2. Dựng lại mô hình OLS để lấy hệ số và p-value
    dy = y.diff().dropna()
    y_lag1 = y.shift(1).dropna().loc[dy.index]

    if best_lag > 0:
        dy_lags = pd.concat([dy.shift(i) for i in range(1, best_lag + 1)], axis=1)
        dy_lags.columns = [f'dy_lag{i}' for i in range(1, best_lag + 1)]
        dy_lags = dy_lags.dropna()
        y_lag1 = y_lag1.loc[dy_lags.index]
        dy = dy.loc[y_lag1.index]
        X = pd.concat([y_lag1.rename('y_lag1'), dy_lags], axis=1)
    else:
        dy = dy.loc[y_lag1.index]
        X = pd.DataFrame({'y_lag1': y_lag1})

    if regression == 'c':
        X = sm.add_constant(X)
    elif regression == 'ct':
        X['trend'] = np.arange(1, len(X) + 1)
        X = sm.add_constant(X)
    elif regression != 'n':
        raise ValueError("regression must be 'c', 'ct', or 'n'")

    model = sm.OLS(dy, X).fit()

    # 3. In hệ số và p-value
    for var in model.params.index:
        coef = model.params[var]
        pval = model.pvalues[var]
        print(f"{var:<10} | Coef: {coef: .4f} | p-value: {pval: .4f}")

    # 4. In ADF statistic và critical values
    print(f"\nADF test statistic: {adf_statistic:.4f}")
    print("Critical values:")
    for level, value in critical_values.items():
        print(f"  {level}: {value:.4f}")


def adf_test_full(df, col='X', title_city=None, show_output=True):

    def adf_get_result(y, regression='c'):
        y = y.dropna()
        result = adfuller(y, regression=regression, autolag='AIC')
        best_lag = result[2]
        critical_values = result[4]
        adf_stat = result[0]
        critical_5 = critical_values['5%']

        dy = y.diff().dropna()
        y_lag1 = y.shift(1).dropna().loc[dy.index]

        if best_lag > 0:
            dy_lags = pd.concat([dy.shift(i) for i in range(1, best_lag + 1)], axis=1)
            dy_lags.columns = [f'dy_lag{i}' for i in range(1, best_lag + 1)]
            dy_lags = dy_lags.dropna()
            y_lag1 = y_lag1.loc[dy_lags.index]
            dy = dy.loc[y_lag1.index]
            X = pd.concat([y_lag1.rename('y_lag1'), dy_lags], axis=1)
        else:
            dy = dy.loc[y_lag1.index]
            X = pd.DataFrame({'y_lag1': y_lag1})

        if regression == 'c':
            X = sm.add_constant(X)
        elif regression == 'ct':
            X['trend'] = np.arange(1, len(X) + 1)
            X = sm.add_constant(X)

        model = sm.OLS(dy, X).fit()

        const_p = model.pvalues['const'] if 'const' in model.pvalues else None
        trend_p = model.pvalues['trend'] if 'trend' in model.pvalues else None

        return {
            'reg': regression,
            'model': model,
            'const_p': const_p,
            'trend_p': trend_p,
            'adf_stat': adf_stat,
            'critical_5': critical_5
        }

    y = df[col]

    # --- Bước chọn mô hình ---
    ct = adf_get_result(y, 'ct')
    if ct['trend_p'] is not None and ct['trend_p'] <= 0.05:
        chosen = ct
    else:
        c = adf_get_result(y, 'c')
        if c['const_p'] is not None and c['const_p'] <= 0.05:
            chosen = c
        else:
            chosen = adf_get_result(y, 'n')

    # --- Kết luận stationarity ---
    if abs(chosen['adf_stat']) > abs(chosen['critical_5']):
        conclusion = f"{chosen['reg']} stationary"
    else:
        conclusion = f"{chosen['reg']} non-stationary"

    # --- Nếu không show_output: chỉ in kết luận ---
    if not show_output:
        print("→", conclusion)
        return conclusion

    # --- Ngược lại: in full output ---
    print("\n===== FULL ADF OUTPUT =====\n")
    adf_detailed(df[col], regression='ct')
    adf_detailed(df[col], regression='c')
    adf_detailed(df[col], regression='n')

    # --- Vẽ ACF / PACF ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(df[col], ax=axes[0], lags=365, markersize=1)
    plot_pacf(df[col], ax=axes[1], lags=10, method='ywm')
    axes[0].set_title(f'ACF ({title_city})')
    axes[1].set_title(f'PACF ({title_city})')
    plt.tight_layout()
    plt.show()

    print("\n===== FINAL DECISION =====")
    print("→", conclusion)

    return conclusion


# ---------------------------
# check_residuals (giữ nguyên)
# ---------------------------
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import matplotlib.pyplot as plt

def check_residuals(df, eps_col='eps', lags=10, title_city=None):
    if eps_col not in df.columns:
        raise ValueError(f"Cột '{eps_col}' không tồn tại trong DataFrame")
    
    eps = df[eps_col].dropna()
    
    print("=== 📊 Kiểm tra phân phối chuẩn ===")
    jb_stat, jb_p, skew, kurt = jarque_bera(eps)   # ← sửa đúng dòng này
    print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")


    
    print("\n=== 🔍 Kiểm tra tự tương quan (Ljung–Box) ===")
    lb_res = acorr_ljungbox(eps, lags=[lags], return_df=True)
    print(f"Ljung–Box test for {eps_col} (lag={lags}): statistic={lb_res['lb_stat'].iloc[0]:.4f}, "
          f"p-value={lb_res['lb_pvalue'].iloc[0]:.4f}")
    
    # ACF & PACF cho eps
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    plot_acf(eps, ax=ax[0], lags=365, markersize=1)
    plot_pacf(eps, ax=ax[1], lags=lags, method='ywm')
    ax[0].set_title(f'ACF ({title_city})')
    ax[1].set_title(f'PACF ({title_city})')
    plt.show()
    
    # --- Cho eps^2 ---
    eps2 = eps**2
    
    print("\n=== 🔍 Kiểm tra tự tương quan (Ljung–Box) cho bình phương ===")
    lb_res2 = acorr_ljungbox(eps2, lags=[lags], return_df=True)
    print(f"Ljung–Box test for {eps_col}² (lag={lags}): statistic={lb_res2['lb_stat'].iloc[0]:.4f}, "
          f"p-value={lb_res2['lb_pvalue'].iloc[0]:.4f}")
    
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    plot_acf(eps2, ax=ax[0], lags=365, markersize=1)
    plot_pacf(eps2, ax=ax[1], lags=lags, method='ywm')
    ax[0].set_title(f'ACF ({title_city})')
    ax[1].set_title(f'PACF ({title_city})')
    plt.show()

    # === 🌩 Kiểm định ARCH (Engle’s ARCH LM test) ===
    print("\n=== 🌩 Kiểm định ARCH ===")
    arch_stat, arch_p, f_stat, f_p = het_arch(eps, maxlag=lags)
    print(f"ARCH test (lag={lags}): LM-stat={arch_stat:.4f}, p-value={arch_p:.4f}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_degree_index(df, start, end, 
                      base_temp=65, index_type="HDD",
                      tick_size=20, 
                      contract="Futures", 
                      option_type=None, 
                      strike=None,
                      plot=False):
    """
    Tính weather index, payoff, và tùy chọn vẽ histogram.
    """

    if "DATE" not in df.columns:
        raise ValueError("DataFrame phải có cột 'DATE'.")

    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)

    df_period = df[(df["DATE"] >= start_ts) & (df["DATE"] <= end_ts)]
    path_cols = [c for c in df.columns if c.startswith("path_")]
    if not path_cols:
        raise ValueError("Không tìm thấy cột path_i nào trong DataFrame.")

    # Tính index
    idx = index_type.upper()
    if idx == "HDD":
        index_series = (base_temp - df_period[path_cols]).clip(lower=0).sum()
    elif idx == "CDD":
        index_series = (df_period[path_cols] - base_temp).clip(lower=0).sum()
    elif idx == "CAT":
        index_series = df_period[path_cols].sum()
    else:
        raise ValueError("index_type phải là 'HDD', 'CDD' hoặc 'CAT'.")

    # Tính payoff
    contract = contract.capitalize()
    if contract == "Futures":
        payoff = tick_size * index_series

    elif contract == "Options":
        if option_type is None or strike is None:
            raise ValueError("Cần truyền option_type và strike khi contract='Options'.")

        option_type = option_type.capitalize()
        if option_type == "Call":
            payoff = tick_size * np.maximum(index_series - strike, 0)
        elif option_type == "Put":
            payoff = tick_size * np.maximum(strike - index_series, 0)
        else:
            raise ValueError("option_type phải là 'Call' hoặc 'Put'.")
    else:
        raise ValueError("contract phải là 'Futures' hoặc 'Options'.")

    payoff.name = f"Payoff_{contract}_{idx}_{start_ts.date()}_{end_ts.date()}"

    # ==============================
    # PLOTTING (no subplots)
    # ==============================
    if plot:
        # ----- Index Distribution -----
        plt.figure(figsize=(7, 5))
        plt.hist(index_series, bins=30, alpha=0.7)
        plt.title(f"Phân phối của chỉ số {idx}")
        plt.xlabel("Chỉ số")
        plt.ylabel("Tần số")

        # Strike line (nếu Option)
        if contract == "Options" and strike is not None:
            plt.axvline(strike, color='red', linestyle='--', linewidth=2)
            plt.text(strike, plt.ylim()[1]*0.9,
                     f"Strike={strike}", color='red')

        plt.show()

        # ----- Payoff Distribution -----
        plt.figure(figsize=(7, 5))
        plt.hist(payoff, bins=30, alpha=0.7)
        plt.title("Phân phối của Payoff")
        plt.xlabel("Payoff")
        plt.ylabel("Tần số")
        plt.show()

    return payoff


