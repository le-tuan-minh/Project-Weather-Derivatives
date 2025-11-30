import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# rpy2 imports (dùng trong fit_gh + simulate)
try:
    from rpy2.robjects import r, IntVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except Exception:
    # sẽ báo lỗi rõ ràng khi gọi hàm GH nếu rpy2 / R không sẵn sàng
    R_AVAILABLE = False


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# ---------------------------
# detrend_deseasonalize
# ---------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def detrend_deseasonalize(df, T_col='T', LAMBDA_col='LAMBDA', X_col='X',
                          omega=2 * np.pi / 365, title_city=None, show_output=True):
    """
    Fit Lambda(t) = a0 + a1*t + a3*cos(omega t) + a4*sin(omega t)

    - Thêm 2 cột:
        + LAMBDA_col: fitted Lambda
        + X_col: phần dư
    - Nếu show_output=True:
        + Vẽ 3 biểu đồ với trục DATE:
            1) T (raw series)
            2) T và Lambda
            3) Residual X
        + In summary
    """

    if T_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{T_col}' trong df")

    if len(df) == 0:
        raise ValueError("DataFrame rỗng")

    # --- Trục thời gian t dạng numeric ---
    n = len(df)
    t = np.arange(n, dtype=float)

    # --- Trục DATE thực tế ---
    if 'DATE' in df.columns:
        date_axis = pd.to_datetime(df['DATE'])
    else:
        date_axis = df.index   # dùng index nếu không có cột DATE

    # --- Plot chuỗi gốc ---
    if show_output:
        plt.figure(figsize=(10, 4))
        plt.plot(date_axis, df[T_col], label='T', color='tab:blue')
        plt.title(f"{title_city}")
        #plt.xlabel('DATE')
        #plt.ylabel(T_col)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # --- Ma trận thiết kế ---
    Xmat = pd.DataFrame({
        'const': 1.0,
        't': t,
        'cos_wt': np.cos(omega * t),
        'sin_wt': np.sin(omega * t)
    }, index=df.index)

    # --- Fit OLS ---
    y = df[T_col].astype(float).values
    model = sm.OLS(y, Xmat)
    results = model.fit()

    # --- Fitted + residual ---
    df = df.copy()
    df[LAMBDA_col] = results.predict(Xmat)
    df[X_col] = df[T_col] - df[LAMBDA_col]

    # --- Plot T & fitted Lambda ---
    if show_output:
        plt.figure(figsize=(10, 4))
        plt.plot(date_axis, df[T_col], label='T', color='tab:blue', alpha=0.7)
        plt.plot(date_axis, df[LAMBDA_col], label='Λ(t)', color='red', linewidth=2)
        plt.title(f"{title_city}")
        #plt.xlabel('DATE')
        #plt.ylabel(T_col)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # --- Plot residual X ---
        plt.figure(figsize=(10, 4))
        plt.plot(date_axis, df[X_col], color='tab:blue')
        plt.title(f"{title_city}")
        #plt.xlabel('DATE')
        #plt.ylabel(X_col)
        #plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        print("=== Lambda fit summary (const + trend + cos + sin) ===")
        print(results.summary())

    # --- Params ---
    params = {
        'model_type': 'lambda_trend_harmonic',
        'omega': omega,
        'res': results,
        'coefs': np.array(results.params).astype(float)
    }

    return df, params





# ---------------------------
# fitCAR
# ---------------------------
def fitCAR(df, 
           X_col='X', 
           p=3, 
           date_col='DATE', 
           pacf_lag=10, 
           acf_lag=365,
           title_city=None,
           show_output=True):
    """
    Fit AR(p) trên X_col, tạo residuals u và u_sq.
    Trả về (df, params) với params chứa ar_res, ar_coefs, p.
    """
    if X_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{X_col}' trong df")

    df = df.copy()
    if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df.sort_values(by=date_col, inplace=True)

    y = df[X_col].astype(float)
    y_nonan = y.dropna().copy()
    if len(y_nonan) <= p:
        raise ValueError(f"Không đủ quan sát để fit AR(p): {len(y_nonan)} <= p={p}")

    original_idx = y_nonan.index.to_numpy()
    y_vals = y_nonan.values

    ar_model = AutoReg(y_vals, lags=p, trend='n', old_names=False)
    ar_res = ar_model.fit()
    fitted = ar_res.fittedvalues
    resid = ar_res.resid

    start_pos = p
    resid_positions = original_idx[start_pos:]
    df['u'] = np.nan
    df.loc[resid_positions, 'u'] = resid
    df['u_sq'] = df['u'] ** 2

    if show_output:
        print("=== AR Model Summary ===")
        print(ar_res.summary())

    def plot_corr(series, name):
        s = series.dropna()
        if len(s) == 0:
            if show_output:
                print(f"No data to plot ACF/PACF for {name}")
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sm.graphics.tsa.plot_acf(s, lags=acf_lag, ax=axes[0], markersize=1)
        sm.graphics.tsa.plot_pacf(s, lags=pacf_lag, ax=axes[1])
        axes[0].set_title(f'ACF ({title_city})')
        axes[1].set_title(f'PACF ({title_city})')
        plt.tight_layout()
        if show_output:
            plt.show()
        else:
            plt.close(fig)

    plot_corr(df['u'], 'u')
    plot_corr(df['u_sq'], 'u_sq')

    ar_coefs = np.array(ar_res.params).astype(float)
    params = {
        'model_type': 'AR',
        'p': p,
        'ar_res': ar_res,
        'ar_coefs': ar_coefs,
        'fittedvalues': fitted
    }

    return df, params


# ---------------------------
# seas_vol
# ---------------------------
def seas_vol(df, 
             K=4, 
             date_col='DATE', 
             omega=2 * np.pi / 365,
             clip_sigma_sq_min=1e-12,
             plot_last_n=365,
             title_city=None,
             show_output=True):
    """
    Fit seasonal variance model dựa trên u_sq_mean theo DAY (như trước).
    Trả về (df, params) với params chứa sig_res (OLS result), design matrix và sigma_sq series.
    """
    df = df.copy()
    if 'MONTH' not in df.columns and date_col in df.columns:
        df['MONTH'] = df[date_col].dt.month
    if 'DAY' not in df.columns:
        if date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['DAY'] = df[date_col].dt.day
        else:
            df['DAY'] = (np.arange(len(df)) % 31) + 1

    if 'u_sq' not in df.columns:
        df['u_sq'] = np.nan
    df['u_sq_mean'] = df.groupby(['MONTH', 'DAY'])['u_sq'].transform('mean')

    t_seq = np.arange(len(df), dtype=float)
    X_sig = pd.DataFrame({'const': 1}, index=df.index)
    for k in range(1, K+1):
        X_sig[f'cos_{k}'] = np.cos(omega * k * t_seq)
        X_sig[f'sin_{k}'] = np.sin(omega * k * t_seq)

    y_sig = df['u_sq_mean']
    mask = y_sig.notna() & X_sig.notna().all(axis=1)

    if mask.sum() == 0:
        df['sigma_sq'] = np.nan
        df['eps'] = np.nan
        if show_output:
            print("⚠ Không có dữ liệu hợp lệ để fit seasonal variance.")
        params = {
            'model_type': 'seasonal_variance',
            'K': K,
            'sig_res': None,
            'sigma_sq': df['sigma_sq']
        }
        return df, params

    sig_model = sm.OLS(y_sig[mask].values, X_sig[mask].values)
    sig_res = sig_model.fit()

    if show_output:
        print("=== Seasonal Variance Model Summary ===")
        print(sig_res.summary())

    sigma_sq_fitted = pd.Series(np.nan, index=df.index)
    sigma_sq_fitted.loc[mask] = sig_res.predict(X_sig[mask].values)
    sigma_sq_fitted = sigma_sq_fitted.clip(lower=clip_sigma_sq_min)
    df['sigma_sq'] = sigma_sq_fitted

    sigma = np.sqrt(df['sigma_sq'])
    df['eps'] = (df['u'] / sigma).replace([np.inf, -np.inf], np.nan)

    # Plot chỉ khi show_output=True
    if show_output:
        last_n = min(plot_last_n, len(df))
        df_plot = df.iloc[-last_n:]
        if date_col in df_plot.columns and pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
            x = df_plot[date_col]
            xlabel = 'DATE'
        else:
            x = df_plot.index
            xlabel = 'index'

        plt.figure(figsize=(10, 4))
        plt.plot(x, df_plot['u_sq_mean'], label='u^2 trung bình', linewidth=1)
        plt.plot(x, df_plot['sigma_sq'], label='σ^2(t)', linewidth=1)
        plt.legend()
        plt.xlabel(xlabel)
        #plt.ylabel('Value')
        plt.title(f"{title_city}")
        plt.tight_layout()
        plt.show()

    params = {
        'model_type': 'seasonal_variance',
        'K': K,
        'omega': omega,
        'sig_res': sig_res,
        'design_cols': X_sig.columns.tolist(),
        'sigma_sq': df['sigma_sq']
    }

    return df, params


# ---------------------------
# fit_gh (giữ nguyên behavior nhưng trả về đối tượng R và params)
# ---------------------------
def fit_gh(df, col='eps', title_city=None, show_output=True):
    """
    Fit GH distribution on residual eps using R package 'ghyp' via rpy2.
    Trả về dict {'gh_fit': r_object, 'params': params_dict}
    Yêu cầu: rpy2 + R package 'ghyp' cài sẵn.
    
    Thêm tham số show_output:
      - True: in kết quả và vẽ biểu đồ
      - False: chỉ fit và trả về kết quả, không in / không plot
    """
    if not R_AVAILABLE:
        raise ImportError("rpy2 không khả dụng. Cài đặt rpy2 và R package 'ghyp' nếu muốn fit GH.")

    ghyp = importr("ghyp")  # có thể raise error nếu ghyp chưa cài
    eps = df[col].dropna()
    if eps.empty:
        raise ValueError("Không có dữ liệu để fit GH.")

    with localconverter(pandas2ri.converter):
        eps_r = pandas2ri.py2rpy(eps)

    # fit.ghypuv : univariate fit from ghyp package
    fit = r['fit.ghypuv'](eps_r, silent=True)

    # lấy params bằng coef(fit)
    params_r = r['coef'](fit)
    params_dict = {}
    try:
        for name in params_r.names:
            val = np.array(params_r.rx2(name))[0]
            params_dict[name] = float(val)
    except Exception:
        try:
            params_dict = dict(zip(params_r.names, [float(np.array(params_r.rx2(n))[0]) for n in params_r.names]))
        except Exception:
            params_dict = {'raw_coef_object': params_r}

    if show_output:
        print("✅ Estimated GH parameters (R ghyp):")
        for k, v in params_dict.items():
            try:
                print(f"  {k}: {v:.6f}")
            except Exception:
                print(f"  {k}: {v}")

    # vẽ histogram + fitted pdf để kiểm tra
    x_grid = np.linspace(eps.min(), eps.max(), 500)
    with localconverter(pandas2ri.converter):
        x_r = pandas2ri.py2rpy(pd.Series(x_grid))
    pdf_vals = r['dghyp'](x_r, fit)
    with localconverter(pandas2ri.converter):
        pdf_py = np.array(pandas2ri.rpy2py(pdf_vals))

    if show_output:
        plt.figure(figsize=(8, 5))
        plt.hist(eps, bins=50, density=True, alpha=0.5, label="Dữ liệu")
        plt.plot(x_grid, pdf_py, 'r-', lw=2, label="Hàm mật độ ước lượng")
        #plt.xlabel(col)
        #plt.ylabel("Density")
        plt.title(f"{title_city}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {'gh_fit': fit, 'params': params_dict}



# Các import cần có (nếu chưa có ở file)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from rpy2.robjects import r, default_converter, conversion
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter

def simulate_forecast_T_using_GH(df,
                                 ar_params=None,
                                 sig_params=None,
                                 lambda_params=None,
                                 gh_fit=None,
                                 horizon=365,
                                 n_sims=200,
                                 p=3,
                                 date_col='DATE',
                                 omega=2 * np.pi / 365,
                                 clip_sigma_sq_min=1e-12,
                                 seed=None,
                                 plot=True):
    """
    Mô phỏng T với eps ~ GH, có seasonal λ(t), AR(p), sigma(t).
    Trả về:
      - sims: numpy (n_sims, horizon)
      - sims_df: DataFrame future-index
      - paths_df: DataFrame full-length (history + forecast), history đã được fill
      - median, mean, p05, p95: thống kê dự báo
    """

    rng = np.random.RandomState(seed)

    if not R_AVAILABLE:
        raise ImportError("rpy2 không khả dụng. Cần rpy2 và package R 'ghyp'.")
    if gh_fit is None:
        raise ValueError("Bạn phải cung cấp 'gh_fit' để sampling GH.")
    gh_fit_obj = gh_fit['gh_fit'] if isinstance(gh_fit, dict) and 'gh_fit' in gh_fit else gh_fit

    df_local = df.copy()
    n = len(df_local)
    t_full = np.arange(n + horizon, dtype=float)

    # --- LAMBDA(t)
    if lambda_params is not None and 'res' in lambda_params:
        lam_res = lambda_params['res']
        X_lambda_full = np.column_stack((
            np.ones(n + horizon),
            t_full,
            np.cos(omega * t_full),
            np.sin(omega * t_full)
        ))
        try:
            lambda_f = lam_res.predict(X_lambda_full)
        except:
            coefs = lambda_params.get('coefs', None)
            lambda_f = X_lambda_full.dot(coefs) if coefs is not None else np.zeros(n + horizon)
    else:
        if 'T' in df_local.columns:
            yT = df_local['T'].astype(float).values
            mask_T = ~np.isnan(yT)
            X_lambda_full = np.column_stack((
                np.ones(n + horizon),
                t_full,
                np.cos(omega * t_full),
                np.sin(omega * t_full)
            ))
            lam_res = sm.OLS(yT[mask_T], X_lambda_full[:n, :][mask_T]).fit()
            lambda_f = lam_res.predict(X_lambda_full)
        else:
            lambda_f = np.zeros(n + horizon)

    # --- sigma^2(t)
    if sig_params is not None and sig_params.get('sig_res', None) is not None:
        sig_res = sig_params['sig_res']
        K = sig_params.get('K', 4)
        X_sig_full = np.ones((n + horizon, 1))
        for k in range(1, K + 1):
            X_sig_full = np.hstack((X_sig_full,
                                    np.cos(omega * k * t_full).reshape(-1, 1),
                                    np.sin(omega * k * t_full).reshape(-1, 1)))
        try:
            sigma_sq_pred_full = sig_res.predict(X_sig_full)
        except:
            sigma_sq_pred_full = np.dot(X_sig_full, np.array(sig_res.params))
        sigma_sq_f = np.clip(sigma_sq_pred_full, clip_sigma_sq_min, None)
    else:
        if 'sigma_sq' in df_local.columns and df_local['sigma_sq'].dropna().size > 0:
            last = df_local['sigma_sq'].dropna().iloc[-1]
            sigma_sq_f = np.full(n + horizon, max(last, clip_sigma_sq_min))
        else:
            sigma_sq_f = np.full(n + horizon, 1.0)
    sigma_f = np.sqrt(sigma_sq_f)

    # --- AR coefficients
    if ar_params is not None and 'ar_coefs' in ar_params:
        ar_coefs = np.array(ar_params['ar_coefs']).astype(float)
        p_used = int(ar_params.get('p', len(ar_coefs)))
    else:
        if 'X' in df_local.columns and df_local['X'].dropna().shape[0] > p:
            x_series = df_local['X'].astype(float).dropna()
            ar_model = AutoReg(x_series.values, lags=p, trend='n', old_names=False)
            ar_res = ar_model.fit()
            ar_coefs = np.array(ar_res.params).astype(float)
            p_used = p
        else:
            ar_coefs = np.zeros(p)
            p_used = p

    if ar_params is not None and 'X_last' in ar_params:
        X_last = np.array(ar_params['X_last']).astype(float)
    else:
        if 'X' in df_local.columns and df_local['X'].dropna().shape[0] >= p_used:
            X_last = df_local['X'].dropna().values[-p_used:].astype(float)
        else:
            X_last = np.zeros(p_used)
    X_seed = list(X_last[-p_used:])

    # --- Draw GH eps
    total_draws = horizon * n_sims
    if seed is not None:
        try:
            r['set.seed'](int(seed))
        except:
            pass

    eps_r = r['rghyp'](total_draws, gh_fit_obj)

    with localconverter(default_converter + numpy2ri.converter):
        eps_all = np.asarray(conversion.rpy2py(eps_r), dtype=float)

    eps_all = eps_all.reshape((n_sims, horizon))

    # --- AR simulation
    sims = np.zeros((n_sims, horizon), dtype=float)
    for sim in range(n_sims):
        eps_draw = eps_all[sim, :]
        u_future = sigma_f[n: n + horizon] * eps_draw
        x_buf = X_seed.copy()
        x_out = np.zeros(horizon)
        for h in range(horizon):
            ar_part = sum(ar_coefs[i] * x_buf[-1 - i] for i in range(p_used))
            x_new = ar_part + u_future[h]
            x_out[h] = x_new
            x_buf.append(x_new)
            if len(x_buf) > p_used:
                x_buf.pop(0)
        sims[sim, :] = x_out + lambda_f[n: n + horizon]

    # --- Build DATE column
    if date_col in df_local.columns and pd.api.types.is_datetime64_any_dtype(df_local[date_col]):
        historical_dates = pd.to_datetime(df_local[date_col]).reset_index(drop=True)
        last_date = historical_dates.iloc[-1]
        future_index = pd.date_range(start=last_date + pd.Timedelta(1, unit='D'), periods=horizon, freq='D')
        full_dates = pd.concat([historical_dates, pd.Series(future_index)], ignore_index=True)
    else:
        full_dates = pd.Series(np.arange(n + horizon))

    # --- Fill historical path values
    if 'T' in df_local.columns:
        hist_vals = df_local['T'].values.reshape(-1, 1)  # (n,1)
        hist_vals = np.tile(hist_vals, (1, n_sims))      # (n,n_sims)
    else:
        hist_vals = (df_local['X'].values + lambda_f[:n]).reshape(-1, 1)
        hist_vals = np.tile(hist_vals, (1, n_sims))

    future_vals = sims.T  # (horizon, n_sims)
    paths_array = np.vstack([hist_vals, future_vals])

    paths_cols = [f'path_{i+1}' for i in range(n_sims)]
    paths_df = pd.DataFrame(paths_array, columns=paths_cols)
    paths_df.insert(0, date_col, np.asarray(full_dates))

    # --- sims_df (future only)
    if date_col in df_local.columns and pd.api.types.is_datetime64_any_dtype(df_local[date_col]):
        sims_df = pd.DataFrame(sims.T, index=future_index, columns=paths_cols)
    else:
        sims_df = pd.DataFrame(sims.T, index=np.arange(n, n + horizon), columns=paths_cols)

    # --- summary
    median = sims_df.median(axis=1)
    mean = sims_df.mean(axis=1)
    p05 = sims_df.quantile(0.05, axis=1)
    p95 = sims_df.quantile(0.95, axis=1)

    results = {
        'sims': sims,
        'sims_df': sims_df,
        'paths_df': paths_df,
        'median': median,
        'mean': mean,
        'p05': p05,
        'p95': p95
    }

    # --- Plot
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(median.index, median.values, label='Median forecast')
        plt.fill_between(median.index, p05.values, p95.values, alpha=0.25, label='5%-95% interval')
        plt.plot(median.index, lambda_f[n:n + horizon], linestyle='--', label='Lambda (extrap)')
        plt.title(f'Forecast of T for {horizon} days ({n_sims} sims)')
        plt.xlabel('Date')
        plt.ylabel('T')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    return results


