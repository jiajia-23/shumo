import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')
import os

class Config:
    """配置类：管理数据路径、存储路径、模型参数"""
    def __init__(self):
        # 核心数据路径
        self.tariff_data_path = "美国对华关税_清洗后数据.xlsx"  # 表格3：历史关税
        self.gdp_data_path = "us_monthly_gdp_history_data_sep.xlsx"  # 表格2：美国月度GDP
        self.forecast_tariff_path = "第一题/output/关税率预测结果.xlsx"  # 表格1：预测关税
        # 存储路径
        self.save_dir = "第一题/output_var/"
        # 模型参数
        self.max_ar_lag = 6  # AR-GARCH的AR阶数上限
        self.garch_order = (1, 2)  # GARCH(p,q)
        self.forecast_horizon = 24  # 关税预测月数
        self.simulations = 2000  # 模拟次数
        self.var_lag = 6  # VAR模型滞后阶数（适配月度数据）
        # 创建文件夹
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
        print(f"✅ 存储文件夹已准备：{os.path.dirname(self.save_dir)}")

class TariffRateAnalyzer:
    """关税率分析+GDP联动预测类（整合原有逻辑+新增VAR-EGARCH）"""
    def __init__(self, config):
        self.config = config
        # 数据容器（新增GDP相关字段）
        self.data = {
            "raw_tariff": None,       # 原始关税数据
            "rate_nonzero": None,     # 过滤0后的关税率
            "rate_clean": None,       # 异常值清洗后的关税率
            "dates": None,            # 关税时间索引
            "dates_clean": None,      # 清洗后关税时间索引
            "ln_rate": None,          # 对数关税率
            "dln_rate": None,        # 对数差分关税率（平稳）
            "dln_dates": None,        # 对数差分关税时间索引
            "raw_gdp": None,          # 原始GDP数据
            "gdp_monthly": None,      # 月度GDP（对齐后）
            "dln_gdp": None,          # 对数差分GDP（平稳）
            "combined_data": None     # 关税+GDP联动数据（双平稳序列）
        }
        # 模型容器（新增VAR-EGARCH）
        self.model = {
            "best_ar": None,          # AR-GARCH的最优AR阶数
            "garch_model": None,      # AR-GARCH模型
            "garch_result": None,     # AR-GARCH拟合结果
            "var_model": None,        # VAR-EGARCH联动模型
            "var_result": None        # VAR拟合结果
        }
        # 预测结果容器（新增GDP预测）
        self.forecast = {
            "tariff_mean": None,      # 关税预测均值
            "tariff_lower": None,     # 关税预测下边界
            "tariff_upper": None,     # 关税预测上边界
            "tariff_dates": None,     # 关税预测时间
            "tariff_rate": None,      # 最终关税预测值
            "gdp_forecast": None,     # GDP预测值
            "gdp_dates": None         # GDP预测时间
        }

    # ---------------------- 原有核心逻辑（保留+适配） ----------------------
    def load_and_preprocess_tariff(self):
        """加载并预处理历史关税数据"""
        self.data["raw_tariff"] = pd.read_excel(self.config.tariff_data_path).sort_values("Date").reset_index(drop=True)
        tariff_rate = self.data["raw_tariff"]["Tariff_Rate(%)"].dropna()
        self.data["dates"] = pd.to_datetime(self.data["raw_tariff"]["Date"]).dropna()
        
        # 过滤0值
        self.data["rate_nonzero"] = tariff_rate[tariff_rate > 0]
        self.data["dates"] = self.data["dates"][tariff_rate > 0]
        print(f"=== 关税数据加载完成 ===")
        print(f"原始观测值：{len(tariff_rate)} | 过滤后：{len(self.data['rate_nonzero'])}")

    def detect_outliers(self):
        """异常值检测与修正（3σ原则，修复dln_dates赋值顺序）"""
        mean = self.data["rate_nonzero"].mean()
        std = self.data["rate_nonzero"].std()
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std
        outliers = (self.data["rate_nonzero"] > upper_bound) | (self.data["rate_nonzero"] < lower_bound)
        
        print(f"\n=== 异常值检测结果 ===")
        print(f"异常值数量：{outliers.sum()}个 | 阈值：{lower_bound:.2f}% ~ {upper_bound:.2f}%")
        # 用前一个月数据填充异常值
        self.data["rate_clean"] = self.data["rate_nonzero"].copy()
        self.data["rate_clean"][outliers] = self.data["rate_clean"].shift(1)[outliers]
        self.data["rate_clean"] = self.data["rate_clean"].dropna()
        self.data["dates_clean"] = self.data["dates"][self.data["rate_clean"].index]
        
        # 核心修复：先计算dln_rate，再匹配dln_dates（避免None）
        self.data["ln_rate"] = np.log(self.data["rate_clean"])
        self.data["dln_rate"] = self.data["ln_rate"].diff().dropna()  # 先生成dln_rate（非None）
        # 再用dln_rate的长度匹配时间索引
        self.data["dln_dates"] = self.data["dates_clean"][len(self.data["dates_clean"]) - len(self.data["dln_rate"]):]
        print(f"清洗后：{len(self.data['rate_clean'])} | 对数差分后：{len(self.data['dln_rate'])}")

    def adf_test(self, series, title):
        """ADF平稳性检验（含可视化标注）"""
        result = adfuller(series)
        print(f"\n=== {title} ADF检验 ===")
        print(f"ADF统计量：{result[0]:.4f} | p值：{result[1]:.4f} | 1%临界值：{result[4]['1%']:.4f}")
        print(f"结论：{'平稳' if result[1] < 0.05 else '非平稳'}")
        return result[1]

    def plot_stationarity(self):
        """绘制关税平稳性检验图（ADF+ACF/PACF）"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle('Tariff Rate Stationarity Test (ADF + ACF/PACF)', fontsize=16, fontweight='bold')
        
        # 原始关税率
        axes[0,0].plot(self.data["dates"], self.data["rate_nonzero"], color='#1F77B4', linewidth=2)
        axes[0,0].set_title(f'Original Tariff Rate (ADF p={self.adf_test(self.data["rate_nonzero"], "原始关税"):.4f})', fontsize=12)
        axes[0,0].set_xlabel('Date')
        axes[0,0].grid(alpha=0.3)
        
        # 对数差分关税率（平稳）
        axes[0,1].plot(self.data["dln_dates"], self.data["dln_rate"], color='#FF7F0E', linewidth=2)
        axes[0,1].set_title(f'Log-Differenced Tariff (ADF p={self.adf_test(self.data["dln_rate"], "对数差分关税"):.4f})', fontsize=12)
        axes[0,1].set_xlabel('Date')
        axes[0,1].grid(alpha=0.3)
        
        # ACF/PACF
        plot_acf(self.data["dln_rate"], lags=24, ax=axes[1,0], alpha=0.05)
        plot_pacf(self.data["dln_rate"], lags=24, ax=axes[1,1], alpha=0.05)
        axes[1,0].set_title('ACF of Log-Differenced Tariff', fontsize=12)
        axes[1,1].set_title('PACF of Log-Differenced Tariff', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率平稳性检验图.png", bbox_inches='tight')
        plt.close()
        print(f"\n✅ 平稳性图已保存")

    def select_ar_order(self):
        """选择AR-GARCH的最优AR阶数（AIC准则）"""
        best_aic = np.inf
        self.model["best_ar"] = 1
        for ar in range(1, self.config.max_ar_lag + 1):
            try:
                model = arch_model(
                    self.data["dln_rate"],
                    mean='AR', lags=ar,
                    vol='EGARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],
                    dist='t'
                )
                res = model.fit(disp='off', options={'maxiter': 1000})
                if res.aic < best_aic:
                    best_aic = res.aic
                    self.model["best_ar"] = ar
            except:
                continue
        print(f"\n=== 最优AR阶数 == AR({self.model['best_ar']})")

    def fit_garch(self):
        """拟合AR-EGARCH模型（预测关税）"""
        self.model["garch_model"] = arch_model(
            self.data["dln_rate"],
            mean='AR', lags=self.model["best_ar"],
            vol='EGARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],
            dist='t'
        )
        # self.model["garch_result"] = self.model["garch_model"].fit(disp='off', options={'maxiter': 1000})

            # 移除bounds参数，增加迭代次数确保收敛
        self.model["garch_result"] = self.model["garch_model"].fit(disp='off', options={'maxiter': 2000})
        
        print(f"\n=== AR-EGARCH模型系数 ===")
        print(f"ARCH项(alpha)：{self.model['garch_result'].params['alpha[1]']:.4f}（p={self.model['garch_result'].pvalues['alpha[1]']:.4f}）")
        print(f"GARCH项(beta)：{self.model['garch_result'].params['beta[1]']:.4f}（p={self.model['garch_result'].pvalues['beta[1]']:.4f}）")
        print(f"AIC值：{self.model['garch_result'].aic:.4f}")

    def forecast_tariff(self):
        """预测未来关税率（平滑+合理性截断）"""
        fc = self.model["garch_result"].forecast(horizon=self.config.forecast_horizon, method='simulation', simulations=self.config.simulations)
        dln_mean = fc.mean.iloc[-1].values
        # 对数差分阶段平滑
        dln_mean_smoothed = pd.Series(dln_mean).rolling(window=3, min_periods=1).mean().values
        
        # 补全预测边界
        self.forecast["tariff_mean"] = dln_mean_smoothed
        self.forecast["tariff_lower"] = dln_mean_smoothed - 1.96 * (fc.variance.iloc[-1].values ** 0.5)
        self.forecast["tariff_upper"] = dln_mean_smoothed + 1.96 * (fc.variance.iloc[-1].values ** 0.5)
        
        # 还原为原始关税率
        last_ln = self.data["ln_rate"].iloc[-1]
        self.forecast["tariff_rate"] = np.exp(np.cumsum(dln_mean_smoothed) + last_ln)
        self.forecast["tariff_rate"] = np.clip(self.forecast["tariff_rate"], 0, 50)  # 0~50%合理范围
        
        # 预测时间索引
        last_date = self.data["dln_dates"].iloc[-1]
        self.forecast["tariff_dates"] = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=self.config.forecast_horizon, freq='M')
        print(f"\n=== 关税预测完成 ===")
        print(f"预测时间：{self.forecast['tariff_dates'].min().strftime('%Y-%m')} ~ {self.forecast['tariff_dates'].max().strftime('%Y-%m')}")

    # ---------------------- 新增：GDP数据加载+联动建模 ----------------------
    def load_and_preprocess_gdp(self):
        """加载并预处理美国月度GDP数据（适配无列名+标题行的格式）"""
        # 1. 读取GDP数据：跳过第1行标题，用第2行作为列名
        self.data["raw_gdp"] = pd.read_excel(
            self.config.gdp_data_path,
            sheet_name='Data',
            skiprows=1,  # 跳过第1行（标题行：Monthly Nominal GDP Index...）
            header=0     # 用第2行作为列名
        )
        
        # 2. 修复列名（第一列是日期，无列名，手动命名）
        print(f"\n=== GDP数据列名检测 ===")
        print(f"原始列名：{self.data['raw_gdp'].columns.tolist()}")
        # 手动重命名列（适配你的表格：A列=Date，B列=Nominal_GDP，C列=Real_GDP）
        self.data["raw_gdp"].columns = ["Date", "Nominal_GDP_Index", "Real_GDP_Index"]
        
        # 3. 解析日期（适配“1992 - Jan”格式）
        print(f"正在解析日期...")
        self.data["raw_gdp"]["Date"] = pd.to_datetime(
            self.data["raw_gdp"]["Date"],
            format='%Y - %b',  # 匹配“1992 - Jan”格式（年 - 月缩写）
            errors='coerce'
        )
        
        # 4. 清理无效数据
        self.data["raw_gdp"] = self.data["raw_gdp"].sort_values("Date").dropna(subset=["Date", "Real_GDP_Index"])
        if len(self.data["raw_gdp"]) == 0:
            raise ValueError("GDP数据解析失败，请检查Date列格式！")
        
        # 5. 提取实际GDP指数并平稳化（注意：这里是“指数”，需用对数差分）
        self.data["gdp_monthly"] = self.data["raw_gdp"][["Date", "Real_GDP_Index"]].copy()
        self.data["gdp_monthly"]["dln_gdp"] = np.log(self.data["gdp_monthly"]["Real_GDP_Index"]).diff().dropna()
        self.data["dln_gdp"] = self.data["gdp_monthly"]["dln_gdp"].dropna()
        
        print(f"\n=== GDP数据加载完成 ===")
        print(f"时间范围：{self.data['gdp_monthly']['Date'].min().strftime('%Y-%m')} ~ {self.data['gdp_monthly']['Date'].max().strftime('%Y-%m')}")
        print(f"有效观测值：{len(self.data['gdp_monthly'])}个月度数据")
        print(f"GDP对数差分平稳性：ADF p={self.adf_test(self.data['dln_gdp'], 'GDP对数差分'):.4f}")
    def combine_tariff_gdp(self):
        """对齐关税（历史+预测）和GDP数据（双月频）"""
        # 1. 整理历史关税+GDP（时间对齐）
        hist_tariff_df = pd.DataFrame({
            "Date": self.data["dln_dates"],
            "dln_tariff": self.data["dln_rate"].values
        })
        hist_gdp_df = self.data["gdp_monthly"][["Date", "dln_gdp"]].dropna()
        
        # 2. 合并历史数据（仅保留共同时间）
        combined_hist = pd.merge(hist_tariff_df, hist_gdp_df, on="Date", how="inner")
        
        # 3. 整理预测关税（用于GDP预测）
        forecast_tariff_df = pd.DataFrame({
            "Date": self.forecast["tariff_dates"],
            "dln_tariff": self.forecast["tariff_mean"]  # 用对数差分后的预测关税
        })
        
        self.data["combined_data"] = combined_hist
        self.forecast["gdp_dates"] = self.forecast["tariff_dates"]  # GDP与关税预测时间一致
        print(f"\n=== 数据对齐完成 ===")
        print(f"历史联动数据量：{len(combined_hist)}个月度观测值")
        print(f"GDP预测时间：{self.forecast['gdp_dates'].min().strftime('%Y-%m')} ~ {self.forecast['gdp_dates'].max().strftime('%Y-%m')}")

    def fit_var_egarch(self):
        """拟合VAR-EGARCH联动模型（关税→GDP影响量化）"""
        # VAR模型输入：平稳的双变量（对数差分关税、对数差分GDP）
        var_input = self.data["combined_data"][["dln_tariff", "dln_gdp"]].values
        
        # 拟合VAR模型（捕捉动态联动）
        self.model["var_model"] = VAR(var_input)
        self.model["var_result"] = self.model["var_model"].fit(self.config.var_lag)
        
        # 输出关税对GDP的影响系数（修复R²报错）
        print(f"\n=== VAR-EGARCH联动模型结果 ===")
        print(f"关税对GDP的短期影响系数（滞后1期）：{self.model['var_result'].coefs[0, 1, 0]:.6f}")
        print(f"关税对GDP的长期影响系数（累计）：{np.sum(self.model['var_result'].coefs[:, 1, 0]):.6f}")
        # 替换R²：用VAR模型的AIC值（越小越好）
        print(f"模型拟合优度：AIC={self.model['var_result'].aic:.4f}")
        
        # 预测GDP（基于关税预测值）
        # 构造VAR预测输入（最后6期历史数据+预测关税的对数差分）
        last_hist_data = var_input[-self.config.var_lag:]
        forecast_tariff_dln = self.forecast["tariff_mean"].reshape(-1, 1)
        
        # 逐期预测GDP的对数差分
        gdp_forecast_dln = []
        current_input = last_hist_data.copy()
        for i in range(self.config.forecast_horizon):
            # 用VAR模型预测下一期GDP差分
            next_gdp_dln = self.model["var_result"].forecast(current_input, steps=1)[0, 1]
            gdp_forecast_dln.append(next_gdp_dln)
            # 更新输入窗口（加入新预测的GDP差分和下一期关税差分）
            next_input = np.zeros_like(current_input)
            next_input[:-1] = current_input[1:]
            next_input[-1] = [forecast_tariff_dln[i, 0], next_gdp_dln]
            current_input = next_input
        
        # 还原为原始GDP预测值
        last_gdp_ln = np.log(self.data["gdp_monthly"]["Real_GDP_Index"].iloc[-1])
        self.forecast["gdp_forecast"] = np.exp(np.cumsum(gdp_forecast_dln) + last_gdp_ln)
        print(f"GDP预测完成：{len(self.forecast['gdp_forecast'])}个月度观测值")

    # ---------------------- 新增：联动结果可视化 ----------------------
    def plot_tariff_gdp_impact(self):
        """绘制关税对GDP的冲击效应+预测图"""
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=300)
        
        # 子图1：关税预测+历史对比
        axes[0].plot(self.data["dates_clean"][-60:], self.data["rate_clean"][-60:], color='#1F77B4', linewidth=2.5, label='Historical Tariff Rate', marker='o', markersize=4)
        axes[0].plot(self.forecast["tariff_dates"], self.forecast["tariff_rate"], color='#D62728', linewidth=3, label='Forecasted Tariff Rate', marker='s', markersize=5)
        axes[0].set_title('U.S. Tariff Rate (Historical + Forecast)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Tariff Rate (%)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # 子图2：GDP真实值+预测值
        axes[1].plot(self.data["gdp_monthly"]["Date"][-60:], self.data["gdp_monthly"]["Real_GDP_Index"][-60:], color='#2CA02C', linewidth=2.5, label='Historical Real GDP', marker='o', markersize=4)
        axes[1].plot(self.forecast["gdp_dates"], self.forecast["gdp_forecast"], color='#FF7F0E', linewidth=3, label='Forecasted Real GDP', marker='s', markersize=5)
        axes[1].set_title('U.S. Real GDP (Historical + Forecast) - Tariff Impact', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Real GDP (Billion USD)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税-GDP联动预测图.png", bbox_inches='tight')
        plt.close()
        print(f"✅ 关税-GDP联动预测图已保存")

    def plot_impulse_response(self):
        """绘制脉冲响应图（修复x与y维度不匹配）"""
        irf = self.model["var_result"].irf(12)  # 12期脉冲响应（实际生成0-12期，共13个值）
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        
        # 1. 提取脉冲响应值（长度13：0-12期）
        irf_data = irf.irfs[:, 1, 0]  # 维度：(13,)
        
        # 2. 调用stderr方法生成标准误（同样长度13）
        irf_stderr = irf.stderr()
        irf_std = irf_stderr[:, 1, 0]  # 维度：(13,)
        
        # 3. 修复x轴：生成与y匹配的长度（0-12期，共13个点）
        x_horizon = range(len(irf_data))  # 替代range(1,13)，自动匹配y长度
        
        # 4. 绘制脉冲响应曲线+95%置信区间
        ax.plot(x_horizon, irf_data, color='#D62728', linewidth=3, marker='o', markersize=5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(
            x_horizon,
            irf_data - 1.96 * irf_std,
            irf_data + 1.96 * irf_std,
            color='#D62728', alpha=0.2
        )
        
        # 5. 优化x轴标签（显示“第1期”到“第12期”，去掉第0期）
        ax.set_xticks(range(1, len(irf_data)))  # 只显示1-12期
        ax.set_xticklabels([f'{i}' for i in range(1, len(irf_data))])  # 标签为“1”到“12”
        
        ax.set_title('Impulse Response: Tariff Rate Shock → GDP Growth', fontsize=14, fontweight='bold')
        ax.set_xlabel('Horizon (Months)', fontsize=12)
        ax.set_ylabel('Response of Log-Differenced GDP Index', fontsize=12)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税对GDP脉冲响应图.png", bbox_inches='tight')
        plt.close()
        print(f"✅ 脉冲响应图已保存")
    # ---------------------- 新增：结果保存拓展 ----------------------
    def save_combined_results(self):
        """保存关税+GDP联动预测结果（修复VAR模型R²报错）"""
        # 1. 关税预测结果（原有）
        tariff_forecast_df = pd.DataFrame({
            "Forecast_Date": self.forecast["tariff_dates"].strftime('%Y-%m'),
            "Forecasted_Tariff_Rate(%)": self.forecast["tariff_rate"].round(4),
            "DLN_Tariff_Mean": self.forecast["tariff_mean"].round(6)
        })
        tariff_forecast_df.to_excel(f"{self.config.save_dir}关税率预测结果.xlsx", index=False)
        
        # 2. GDP预测结果（新增）
        gdp_forecast_df = pd.DataFrame({
            "Forecast_Date": self.forecast["gdp_dates"].strftime('%Y-%m'),
            "Forecasted_Real_GDP_Index(Billion USD)": self.forecast["gdp_forecast"].round(4),
            "Corresponding_Tariff_Rate(%)": self.forecast["tariff_rate"].round(4)
        })
        gdp_forecast_df.to_excel(f"{self.config.save_dir}GDP预测结果.xlsx", index=False)
        
        # 3. 联动模型系数（新增：删除rsquared，替换为VAR适配指标）
        with open(f"{self.config.save_dir}VAR-EGARCH联动模型结果.txt", 'w', encoding='utf-8') as f:
            f.write("VAR-EGARCH Model Summary (Tariff → GDP)\n")
            f.write(f"VAR Lag Order: {self.config.var_lag}\n")
            f.write(f"GARCH Order: {self.config.garch_order}\n\n")
            f.write("Tariff → GDP Impact Coefficients (Lag 1-6):\n")
            for i in range(self.config.var_lag):
                coef = self.model["var_result"].coefs[i, 1, 0]
                f.write(f"  Lag {i+1}: {coef:.6f}\n")
            f.write(f"\nLong-Term Impact Coefficient: {np.sum(self.model['var_result'].coefs[:, 1, 0]):.6f}\n")
            # 替换R²：用VAR模型的AIC、BIC（越小越好，适配VAR评估逻辑）
            f.write(f"VAR Model Fit Metrics:\n")
            f.write(f"  AIC: {self.model['var_result'].aic:.4f}\n")
            f.write(f"  BIC: {self.model['var_result'].bic:.4f}\n")
            f.write(f"  Log Likelihood: {self.model['var_result'].llf:.4f}\n")
        
        print(f"\n✅ 所有结果已保存至：{self.config.save_dir}")
        print(f"包含：关税预测.xlsx、GDP预测.xlsx、联动模型结果.txt、2张可视化图")

# ---------------------- 主流程（整合原有+新增） ----------------------
def main():
    # 初始化配置和分析器
    config = Config()
    analyzer = TariffRateAnalyzer(config)
    
    # 原有流程：关税预处理→建模→预测
    analyzer.load_and_preprocess_tariff()
    analyzer.detect_outliers()
    analyzer.plot_stationarity()
    analyzer.select_ar_order()
    analyzer.fit_garch()
    analyzer.forecast_tariff()
    
    # 新增流程：GDP预处理→联动建模→可视化→保存
    analyzer.load_and_preprocess_gdp()
    analyzer.combine_tariff_gdp()
    analyzer.fit_var_egarch()
    analyzer.plot_tariff_gdp_impact()
    analyzer.plot_impulse_response()
    analyzer.save_combined_results()

if __name__ == "__main__":
    main()