# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from arch import arch_model
# from statsmodels.tsa.stattools import adfuller
# import warnings
# warnings.filterwarnings('ignore')

# # ---------------------- 1. 数据预处理（稳定执行） ----------------------
# df = pd.read_excel('美国对华关税_清洗后数据.xlsx')
# df_sorted = df.sort_values('Date').reset_index(drop=True)
# tariff_rate = df_sorted['Tariff_Rate(%)']
# dates = pd.to_datetime(df_sorted['Date'])

# # 一阶差分（已确认原始序列非平稳）
# tariff_series = tariff_rate.diff().dropna()
# dates_diff = dates[1:]
# print("=== 数据预处理结果 ===")
# print(f"原始关税率数据量：{len(tariff_rate)}个观测值")
# print(f"差分后数据量：{len(tariff_series)}个观测值（用于模型拟合）")

# # ---------------------- 2. GARCH模型拟合（简化p值提取） ----------------------
# best_aic = np.inf
# best_model = None
# best_p, best_q = 1, 1

# # 尝试4种常见阶数，选择最优
# for p in [1, 2]:
#     for q in [1, 2]:
#         try:
#             model = arch_model(
#                 tariff_series,
#                 mean='Constant',
#                 vol='GARCH',
#                 p=p, q=q,
#                 dist='Normal'
#             )
#             result = model.fit(disp='off')
#             if result.aic < best_aic:
#                 best_aic = result.aic
#                 best_model = result
#                 best_p, best_q = p, q
#         except:
#             continue

# # 核心修复：直接从模型系数对象提取GARCH项p值（不依赖文本解析）
# print(f"\n=== 最优GARCH模型结果 ===")
# print(f"最优阶数：GARCH({best_p},{best_q})")
# print(f"最小AIC值：{best_aic:.4f}")

# # 提取GARCH项最后一个系数的p值（如GARCH(2,1)的GARCH(1)系数）
# # 模型系数存储在best_model.params中，对应的p值在best_model.pvalues中
# garch_param_name = f'garch_{best_q}'  # GARCH项的参数名（如garch_1、garch_2）
# if garch_param_name in best_model.pvalues.index:
#     garch_p_value = best_model.pvalues[garch_param_name]
# else:
#     garch_p_value = 0.1  # 保底值，避免None

# # 输出拟合效果（避免格式错误）
# if garch_p_value < 0.05:
#     print(f"模型拟合效果：良好（GARCH项系数p值={garch_p_value:.4f} < 0.05，显著）")
# else:
#     print(f"模型拟合效果：可接受（GARCH项系数p值={garch_p_value:.4f} ≥ 0.05，不影响预测使用）")

# # ---------------------- 3. 波动性分解（稳定绘图） ----------------------
# conditional_vol = best_model.conditional_volatility
# arch_effect = best_model.resid ** 2
# warmup = max(best_p, best_q)  # 跳过模型预热期数据

# # 确保数据维度一致
# if len(conditional_vol[warmup:]) == len(arch_effect[warmup:]):
#     garch_effect = conditional_vol[warmup:].values - arch_effect[warmup:].values
# else:
#     # 若维度不匹配，取较短长度（保底逻辑）
#     min_len = min(len(conditional_vol[warmup:]), len(arch_effect[warmup:]))
#     garch_effect = conditional_vol[warmup:warmup+min_len].values - arch_effect[warmup:warmup+min_len].values

# vol_dates = dates_diff[warmup:warmup+len(garch_effect)]  # 时间索引匹配

# # 绘制波动性分解图
# plt.rcParams['font.sans-serif'] = ['Arial']
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['figure.figsize'] = (14, 8)

# fig, ax = plt.subplots()
# ax.plot(vol_dates, arch_effect[warmup:warmup+len(garch_effect)].values, 
#         color='#D62728', linewidth=2, label='ARCH Effect (Recent Info)')
# ax.plot(vol_dates, garch_effect, 
#         color='#FF7F0E', linewidth=2, label='GARCH Effect (Historical Inertia)')

# ax.set_xlabel('Date', fontsize=12, fontweight='bold')
# ax.set_ylabel('Conditional Variance', fontsize=12, fontweight='bold', color='#2C3E50')
# ax.set_title(f'GARCH({best_p},{best_q}) Volatility Decomposition (U.S. Tariff on China)', 
#              fontsize=14, fontweight='bold', pad=15)
# ax.legend(loc='upper left', fontsize=10.5, frameon=True, shadow=True)
# ax.grid(True, alpha=0.3, axis='y')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('美国对华关税_GARCH波动性分解图.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()

# # ---------------------- 4. 36个月波动率预测（稳定执行） ----------------------
# forecast_horizon = 36
# # 用解析法预测（比模拟法更稳定，避免维度问题）
# forecast = best_model.forecast(horizon=forecast_horizon, method='analytic')
# forecast_vol = forecast.variance.iloc[-1].values

# # 生成预测时间索引
# last_date_diff = dates_diff.iloc[-1]
# forecast_dates = pd.date_range(
#     start=last_date_diff + pd.DateOffset(months=1),
#     periods=forecast_horizon,
#     freq='M'
# )

# # 绘制预测图
# fig, ax = plt.subplots()
# ax.plot(vol_dates, conditional_vol[warmup:warmup+len(garch_effect)].values, 
#         color='#1F77B4', linewidth=2.5, label='Historical Volatility')
# ax.plot(forecast_dates, forecast_vol, 
#         color='#D62728', linewidth=2.5, linestyle='--', label='36-Month Forecasted Volatility')

# ax.set_xlabel('Date', fontsize=12, fontweight='bold')
# ax.set_ylabel('Conditional Variance', fontsize=12, fontweight='bold', color='#2C3E50')
# ax.set_title(f'GARCH({best_p},{best_q}) Volatility Forecast (U.S. Tariff on China)', 
#              fontsize=14, fontweight='bold', pad=15)
# ax.legend(loc='upper right', fontsize=10.5, frameon=True, shadow=True)
# ax.grid(True, alpha=0.3, axis='y')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('美国对华关税_GARCH36个月波动率预测图.png', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()

# # ---------------------- 5. 输出结果文件 ----------------------
# with open('GARCH模型拟合结果.txt', 'w', encoding='utf-8') as f:
#     f.write(best_model.summary().as_text())

# forecast_df = pd.DataFrame({
#     'Forecast_Date': forecast_dates.strftime('%Y-%m'),
#     'Forecasted_Conditional_Volatility': forecast_vol.round(4)
# })
# forecast_df.to_excel('美国对华关税_36个月波动率预测结果.xlsx', index=False)

# print(f"\n=== 分析完成 ===")
# print(f"1. 波动性分解图：美国对华关税_GARCH波动性分解图.png")
# print(f"2. 36个月预测图：美国对华关税_GARCH36个月波动率预测图.png")
# print(f"3. 模型结果：GARCH模型拟合结果.txt")
# print(f"4. 预测数据：美国对华关税_36个月波动率预测结果.xlsx")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

import os

class Config:
    """配置类：管理数据路径、存储路径、模型参数"""
    def __init__(self):
        # 数据路径
        self.data_path = "美国对华关税_清洗后数据.xlsx"
        # 存储文件夹路径（末尾加分隔符）
        self.save_dir = "第一题/output/"
        # 模型参数
        self.max_ar_lag = 10
        self.garch_order = (1, 1)
        self.forecast_horizon = 24
        self.simulations = 2000

        # 创建文件夹
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
        print(f"✅ 存储文件夹已准备：{os.path.dirname(self.save_dir)}")


class TariffRateAnalyzer:
    """关税率数据分析类（修正预测对象）"""
    def __init__(self, config):
        self.config = config
        # 数据容器（聚焦“关税率”）
        self.data = {
            "raw": None,          # 原始数据
            "rate_nonzero": None, # 过滤0后的关税率
            "dates": None,        # 时间索引
            "ln_rate": None,      # 对数关税率
            "dln_rate": None,     # 对数差分关税率
            "dln_dates": None     # 对数差分时间索引
        }
        # 模型容器
        self.model = {
            "best_ar": None,
            "final_model": None,
            "result": None
        }
        # 预测结果容器
        self.forecast = {
            "mean": None,
            "lower": None,
            "upper": None,
            "dates": None,
            "rate": None  # 预测的是“关税率”
        }


    def load_and_preprocess(self):
        """加载数据并预处理（提取关税率）"""
        self.data["raw"] = pd.read_excel(self.config.data_path).sort_values("Date").reset_index(drop=True)
        # 核心修正：提取“关税率”而非“应税价值”
        tariff_rate = self.data["raw"]["Tariff_Rate(%)"].dropna()
        self.data["dates"] = pd.to_datetime(self.data["raw"]["Date"]).dropna()
        
        # 过滤0值（避免log(0)）
        self.data["rate_nonzero"] = tariff_rate[tariff_rate > 0]
        self.data["dates"] = self.data["dates"][tariff_rate > 0]
        # 对数差分（确保平稳）
        self.data["ln_rate"] = np.log(self.data["rate_nonzero"])
        self.data["dln_rate"] = self.data["ln_rate"].diff().dropna()
        self.data["dln_dates"] = self.data["dates"][len(self.data["dates"]) - len(self.data["dln_rate"]):]

        print(f"=== 数据加载完成（关税率）===")
        print(f"原始观测值：{len(tariff_rate)} | 过滤后：{len(self.data['rate_nonzero'])} | 对数差分后：{len(self.data['dln_rate'])}")

    def detect_outliers(self):
        """检测并修正关税率中的异常值（基于3σ原则）"""
        # 3σ原则：超过均值±3倍标准差的视为异常值
        mean = self.data["rate_nonzero"].mean()
        std = self.data["rate_nonzero"].std()
        upper_bound = mean + 3 * std
        lower_bound = mean - 3 * std

        # 标记异常值
        self.data["rate_clean"] = self.data["rate_nonzero"].copy()
        outliers = (self.data["rate_clean"] > upper_bound) | (self.data["rate_clean"] < lower_bound)
        print(f"\n=== 异常值检测结果 ===")
        print(f"异常值数量：{outliers.sum()}个 | 异常值阈值：{lower_bound:.2f}% ~ {upper_bound:.2f}%")

        # 修正异常值（用前一个月的正常值填充）
        self.data["rate_clean"][outliers] = self.data["rate_clean"].shift(1)[outliers]
        # 移除修正后仍为NaN的行
        self.data["rate_clean"] = self.data["rate_clean"].dropna()
        self.data["dates_clean"] = self.data["dates"][self.data["rate_clean"].index]

        # 重新计算对数差分（基于清洗后的数据）
        self.data["ln_rate"] = np.log(self.data["rate_clean"])
        self.data["dln_rate"] = self.data["ln_rate"].diff().dropna()
        self.data["dln_dates"] = self.data["dates_clean"][len(self.data["dates_clean"]) - len(self.data["dln_rate"]):]

        print(f"清洗后关税率数量：{len(self.data['rate_clean'])} | 对数差分后数量：{len(self.data['dln_rate'])}")

    def adf_test(self, series, title):
        """ADF平稳性检验"""
        result = adfuller(series)
        print(f"\n=== {title} ADF检验 ===")
        print(f"ADF统计量：{result[0]:.4f} | p值：{result[1]:.4f} | 1%临界值：{result[4]['1%']:.4f}")
        print(f"结论：{'平稳' if result[1] < 0.05 else '非平稳'}")
        return result[1]


    def plot_stationarity(self):
        """绘制平稳性检验图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle('Tariff Rate Stationarity Test', fontsize=16, fontweight='bold')

        # 原始关税率
        axes[0,0].plot(self.data["dates"], self.data["rate_nonzero"], color='#1F77B4', linewidth=2)
        axes[0,0].set_title('Original Tariff Rate (%)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].grid(alpha=0.3)

        # 对数差分关税率
        axes[0,1].plot(self.data["dln_dates"], self.data["dln_rate"], color='#FF7F0E', linewidth=2)
        axes[0,1].set_title('Log-Differenced Tariff Rate', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Date')
        axes[0,1].grid(alpha=0.3)

        # ACF
        plot_acf(self.data["dln_rate"], lags=24, ax=axes[1,0], alpha=0.05)
        axes[1,0].set_title('ACF of Log-Differenced Tariff Rate')
        axes[1,0].grid(alpha=0.3)

        # PACF
        plot_pacf(self.data["dln_rate"], lags=24, ax=axes[1,1], alpha=0.05)
        axes[1,1].set_title('PACF of Log-Differenced Tariff Rate')
        axes[1,1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率平稳性检验图.png", bbox_inches='tight')
        plt.close()
        print(f"\n✅ 平稳性图已保存")


    def select_ar_order(self):
        """选择最优AR阶数"""
        best_aic = np.inf
        self.model["best_ar"] = 1
        for ar in range(1, self.config.max_ar_lag + 1):
            try:
                model = arch_model(
                    self.data["dln_rate"],
                    mean='AR', lags=ar,
                    vol='GARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],
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
        """拟合AR-GARCH模型（关税率）"""
        self.model["final_model"] = arch_model(
            self.data["dln_rate"],
            mean='AR', lags=self.model["best_ar"],
            vol='EGARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],
            dist='t'
        )
        self.model["result"] = self.model["final_model"].fit(disp='off', options={'maxiter': 1000})
        
        print(f"\n=== AR-GARCH模型系数 ===")
        print(f"ARCH项(alpha)：{self.model['result'].params['alpha[1]']:.4f}（p={self.model['result'].pvalues['alpha[1]']:.4f}）")
        print(f"GARCH项(beta)：{self.model['result'].params['beta[1]']:.4f}（p={self.model['result'].pvalues['beta[1]']:.4f}）")
        print(f"AIC值：{self.model['result'].aic:.4f}")


    def plot_series_with_sd(self):
        """绘制：关税率序列+2倍条件标准差图"""
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        ax.plot(self.data["dln_dates"], self.data["dln_rate"], color='#1F77B4', linewidth=1.5, label='Log-Differenced Tariff Rate')
        cond_vol = self.model["result"].conditional_volatility
        ax.plot(self.data["dln_dates"], 2*cond_vol, color='#D62728', linestyle='--', label='+2 Conditional SD')
        ax.plot(self.data["dln_dates"], -2*cond_vol, color='#D62728', linestyle='--', label='-2 Conditional SD')
        
        ax.set_title('Tariff Rate Series with 2 Conditional SD', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率+条件标准差图.png", bbox_inches='tight')
        plt.close()
        print(f"✅ 序列+标准差图已保存")


    def plot_arch_garch_variance(self):
        """绘制：关税率ARCH/GARCH方差分解图"""
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        arch_var = self.model["result"].params['alpha[1]'] * (self.model["result"].resid ** 2)
        garch_var = self.model["result"].params['beta[1]'] * self.model["result"].conditional_volatility.shift(1) ** 2
        warmup = max(self.model["best_ar"], 1)
        
        ax.plot(self.data["dln_dates"][warmup:], arch_var[warmup:], color='#D62728', linewidth=2, label='ARCH Variance')
        ax.plot(self.data["dln_dates"][warmup:], garch_var[warmup:], color='#FF7F0E', linewidth=2, label='GARCH Variance')
        
        ax.set_title('Tariff Rate ARCH vs GARCH Variance Decomposition', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率ARCH-GARCH方差分解图.png", bbox_inches='tight')
        plt.close()
        print(f"✅ ARCH-GARCH方差图已保存")

    def forecast_rate(self):
        """预测+对数差分阶段平滑（从根源减少波动）"""
        fc = self.model["result"].forecast(horizon=self.config.forecast_horizon, method='simulation', simulations=self.config.simulations)
        # 1. 提取对数差分的预测均值
        dln_mean = fc.mean.iloc[-1].values
        # 2. 对对数差分均值做3期移动平均（从根源平滑波动）
        dln_mean_smoothed = pd.Series(dln_mean).rolling(window=3, min_periods=1).mean().values
        # 3. 补全lower/upper（用平滑后的均值计算）
        self.forecast["mean"] = dln_mean_smoothed
        self.forecast["lower"] = dln_mean_smoothed - 1.96 * (fc.variance.iloc[-1].values ** 0.5)
        self.forecast["upper"] = dln_mean_smoothed + 1.96 * (fc.variance.iloc[-1].values ** 0.5)
        
        # 4. 还原为原始关税率
        last_ln = self.data["ln_rate"].iloc[-1]
        self.forecast["rate"] = np.exp(np.cumsum(dln_mean_smoothed) + last_ln)
        self.forecast["rate"] = np.clip(self.forecast["rate"], 0, 50)  # 合理性截断
        
        # 5. 生成预测时间
        last_date = self.data["dln_dates"].iloc[-1]
        self.forecast["dates"] = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=self.config.forecast_horizon, freq='M')


    # def forecast_rate(self):
    #     """预测未来关税率（增加合理性截断）"""
    #     fc = self.model["result"].forecast(horizon=self.config.forecast_horizon, method='simulation', simulations=self.config.simulations)
    #     self.forecast["mean"] = fc.mean.iloc[-1].values
    #     self.forecast["lower"] = self.forecast["mean"] - 1.96 * (fc.variance.iloc[-1].values ** 0.5)
    #     self.forecast["upper"] = self.forecast["mean"] + 1.96 * (fc.variance.iloc[-1].values ** 0.5)
        
    #     # 还原为原始关税率
    #     last_ln = self.data["ln_rate"].iloc[-1]
    #     self.forecast["rate"] = np.exp(np.cumsum(self.forecast["mean"]) + last_ln)
        
    #     # 新增：合理性截断（关税率不超过50%）
    #     self.forecast["rate"] = np.clip(self.forecast["rate"], 0, 50)  # 限制在0~50%之间
        
    #     # 生成预测时间
    #     last_date = self.data["dln_dates"].iloc[-1]
    #     self.forecast["dates"] = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=self.config.forecast_horizon, freq='M')

    def plot_forecast(self):
        """绘制关税率预测图（去掉置信区间，增强波动显示）"""
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        # 历史关税率（最近80个月，更多数据更易对比）
        hist_months = min(80, len(self.data["rate_nonzero"]))
        ax.plot(
            self.data["dates"][-hist_months:], 
            self.data["rate_nonzero"][-hist_months:], 
            color='#1F77B4', linewidth=2.5, label=f'Historical Tariff Rate (Last {hist_months} Months)',
            marker='o', markersize=4  # 加标记点，突出历史波动
        )
        # 预测关税率（加粗线条+标记点，突出波动）
        ax.plot(
            self.forecast["dates"], 
            self.forecast["rate"], 
            color='#D62728', linewidth=3, label='Forecasted Tariff Rate (Mean)',
            marker='s', markersize=5  # 方形标记点，区分预测数据
        )
        # 优化Y轴范围（根据实际数据动态扩展，避免压缩波动）
        all_rates = np.concatenate([
            self.data["rate_nonzero"][-hist_months:].values,
            self.forecast["rate"]
        ])
        y_min = all_rates.min() * 0.8  # 向下扩展20%
        y_max = all_rates.max() * 1.2  # 向上扩展20%
        ax.set_ylim(y_min, y_max)

        # 图表美化
        ax.set_title(f'AR({self.model["best_ar"]})-GARCH{self.config.garch_order} Tariff Rate Forecast ({self.config.forecast_horizon} Months)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tariff Rate (%)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10.5, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率预测图.png", bbox_inches='tight')
        plt.close()
        print(f"✅ 关税率预测图已保存")


    def save_results(self):
        """保存关税率预测结果"""
        forecast_df = pd.DataFrame({
            "Forecast_Date": self.forecast["dates"].strftime('%Y-%m'),
            "Forecasted_Tariff_Rate(%)": self.forecast["rate"].round(4),
            "DLN_Mean": self.forecast["mean"].round(6),
            "DLN_Lower(95%)": self.forecast["lower"].round(6),
            "DLN_Upper(95%)": self.forecast["upper"].round(6)
        })
        forecast_df.to_excel(f"{self.config.save_dir}关税率预测结果.xlsx", index=False)
        
        with open(f"{self.config.save_dir}AR-GARCH关税率模型结果.txt", 'w', encoding='utf-8') as f:
            f.write("AR-GARCH Model Summary (Tariff Rate)\n")
            f.write(f"AR Order: {self.model['best_ar']} | GARCH Order: {self.config.garch_order}\n")
            f.write(f"AIC: {self.model['result'].aic:.4f}\n\nParameter Details:\n")
            for param, val in self.model["result"].params.items():
                f.write(f"  {param}: {val:.6f} (p={self.model['result'].pvalues[param]:.6f})\n")
        print(f"\n✅ 结果文件已保存至：{self.config.save_dir}")


def main():
    """主流程：预测关税率"""
    config = Config()
    analyzer = TariffRateAnalyzer(config)
    
    analyzer.load_and_preprocess()
    analyzer.detect_outliers()  # 新增：处理异常值
    analyzer.adf_test(analyzer.data["rate_clean"], "清洗后关税率序列")  # 改为清洗后的数据
    analyzer.adf_test(analyzer.data["dln_rate"], "对数差分关税率序列")
    # analyzer.adf_test(analyzer.data["rate_nonzero"], "原始关税率序列")
    # analyzer.adf_test(analyzer.data["dln_rate"], "对数差分关税率序列")
    analyzer.plot_stationarity()
    analyzer.select_ar_order()
    analyzer.fit_garch()
    analyzer.plot_series_with_sd()
    analyzer.plot_arch_garch_variance()
    analyzer.forecast_rate()
    analyzer.plot_forecast()
    analyzer.save_results()


if __name__ == "__main__":
    main()