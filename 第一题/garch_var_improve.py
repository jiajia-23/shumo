# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from arch import arch_model
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.api import VAR
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')
# import os
# from statsmodels.stats.diagnostic import ljungbox

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, acf, pacf  # 保留acf、pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

class Config:
    """配置类：管理数据路径、存储路径、模型参数（对齐GARCH标准流程）"""
    def __init__(self):
        # 核心数据路径
        self.tariff_data_path = "美国对华关税_清洗后数据.xlsx"  # 表格3：历史关税
        self.gdp_data_path = "us_monthly_gdp_history_data_sep.xlsx"  # 表格2：美国月度GDP
        # 存储路径（新增GARCH诊断图文件夹）
        self.save_dir = "第一题/output_var_improve/"
        self.garch_diag_dir = f"{self.save_dir}GARCH诊断图/"
        # 模型参数（按李东风教授流程设置）
        self.max_ar_lag = 10  # AR-GARCH的AR阶数上限
        self.garch_order = (1, 1)  # 标准GARCH(1,1)（李东风教授推荐基础模型）
        self.forecast_horizon = 24  # 关税预测月数
        self.simulations = 2000  # 模拟次数
        self.var_lag = 6  # VAR模型滞后阶数（适配月度数据）
        # 创建文件夹
        os.makedirs(os.path.dirname(self.save_dir), exist_ok=True)
        os.makedirs(self.garch_diag_dir, exist_ok=True)
        print(f"✅ 存储文件夹已准备：{os.path.dirname(self.save_dir)}")
        print(f"✅ GARCH诊断图文件夹已准备：{self.garch_diag_dir}")

class TariffRateAnalyzer:
    """关税率分析+GDP联动预测类（按李东风教授GARCH流程优化）"""
    def __init__(self, config):
        self.config = config
        # 数据容器（新增GARCH诊断相关字段）
        self.data = {
            "raw_tariff": None,       # 原始关税数据
            "rate_nonzero": None,     # 过滤0后的关税率
            "rate_clean": None,       # 异常值清洗后的关税率
            "dates": None,            # 关税时间索引
            "dates_clean": None,      # 清洗后关税时间索引
            "ln_rate": None,          # 对数关税率
            "dln_rate": None,         # 对数差分关税率（平稳）
            "dln_dates": None,        # 对数差分关税时间索引
            "raw_gdp": None,          # 原始GDP数据
            "gdp_monthly": None,      # 月度GDP（对齐后）
            "dln_gdp": None,          # 对数差分GDP（平稳）
            "combined_data": None,    # 关税+GDP联动数据（双平稳序列）
            "garch_resid": None,      # GARCH模型残差（新增）
            "garch_std_resid": None   # GARCH标准化残差（新增）
        }
        # 模型容器（新增GARCH诊断结果）
        self.model = {
            "best_ar": None,          # AR-GARCH的最优AR阶数
            "garch_model": None,      # AR-GARCH模型
            "garch_result": None,     # AR-GARCH拟合结果
            "garch_diag": {           # GARCH诊断结果（新增）
                "resid_acf_p": None,  # 残差ACF检验p值
                "resid_norm_p": None   # 残差正态性检验p值
            },
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
            "tariff_vol": None,       # 关税预测波动率（新增）
            "gdp_forecast": None,     # GDP预测值
            "gdp_dates": None         # GDP预测时间
        }

    # ---------------------- 原有核心逻辑（保留+适配GARCH标准流程） ----------------------
    def load_and_preprocess_tariff(self):
        """加载并预处理历史关税数据"""
        self.data["raw_tariff"] = pd.read_excel(self.config.tariff_data_path).sort_values("Date").reset_index(drop=True)
        tariff_rate = self.data["raw_tariff"]["Tariff_Rate(%)"].dropna()
        self.data["dates"] = pd.to_datetime(self.data["raw_tariff"]["Date"]).dropna()
        
        # 过滤0值（避免log(0)）
        self.data["rate_nonzero"] = tariff_rate[tariff_rate > 0]
        self.data["dates"] = self.data["dates"][tariff_rate > 0]
        print(f"=== 关税数据加载完成 ===")
        print(f"原始观测值：{len(tariff_rate)} | 过滤后：{len(self.data['rate_nonzero'])}")

    def ljung_box_test(self, series, lags=12):
        """手动实现Ljung-Box检验（兼容statsmodels 0.14.5，避免导入错误）"""
        n = len(series.dropna())
        acf_vals = acf(series.dropna(), nlags=lags, fft=False)
        q_stat = n * np.sum(np.square(acf_vals[1:lags+1]))  # 排除滞后0期
        p_val = 1 - stats.chi2.cdf(q_stat, df=lags)
        return q_stat, p_val
    
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
        self.data["dln_dates"] = self.data["dates_clean"][len(self.data["dates_clean"]) - len(self.data["dln_rate"]):]
        print(f"清洗后：{len(self.data['rate_clean'])} | 对数差分后：{len(self.data['dln_rate'])}")

    def adf_test(self, series, title):
        """ADF平稳性检验（含可视化标注，对齐李东风教授检验流程）"""
        result = adfuller(series)
        print(f"\n=== {title} ADF检验 ===")
        print(f"ADF统计量：{result[0]:.4f} | p值：{result[1]:.4f} | 1%临界值：{result[4]['1%']:.4f}")
        print(f"结论：{'平稳（可建模）' if result[1] < 0.05 else '非平稳（需差分）'}")
        return result[1]

    def plot_stationarity(self):
        """绘制关税平稳性检验图（ADF+ACF/PACF，优化学术样式）"""
        plt.rcParams['font.sans-serif'] = ['Arial']  # 学术图表常用字体
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条宽度
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle('Tariff Rate Stationarity Test (ADF + ACF/PACF)', fontsize=16, fontweight='bold', y=0.95)
        
        # 原始关税率（实线，蓝色）
        axes[0,0].plot(self.data["dates"], self.data["rate_nonzero"], color='#2F4F4F', linewidth=2.5)
        axes[0,0].set_title(f'Original Tariff Rate (ADF p={self.adf_test(self.data["rate_nonzero"], "原始关税"):.4f})', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Date', fontweight='bold')
        axes[0,0].set_ylabel('Tariff Rate (%)', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3, linestyle='--')
        axes[0,0].spines['top'].set_visible(False)  # 隐藏上边框
        axes[0,0].spines['right'].set_visible(False)  # 隐藏右边框
        
        # 对数差分关税率（平稳，红色）
        axes[0,1].plot(self.data["dln_dates"], self.data["dln_rate"], color='#DC143C', linewidth=2.5)
        axes[0,1].set_title(f'Log-Differenced Tariff (ADF p={self.adf_test(self.data["dln_rate"], "对数差分关税"):.4f})', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Date', fontweight='bold')
        axes[0,1].set_ylabel('Log-Differenced Tariff', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3, linestyle='--')
        axes[0,1].spines['top'].set_visible(False)
        axes[0,1].spines['right'].set_visible(False)
        
        # ACF（蓝色，置信区间灰色）
        plot_acf(self.data["dln_rate"], lags=24, ax=axes[1,0], alpha=0.05, color='#2F4F4F', linewidth=2)
        axes[1,0].set_title('ACF of Log-Differenced Tariff', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Lags', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3, linestyle='--')
        axes[1,0].spines['top'].set_visible(False)
        axes[1,0].spines['right'].set_visible(False)
        
        # PACF（红色，置信区间灰色）
        plot_pacf(self.data["dln_rate"], lags=24, ax=axes[1,1], alpha=0.05, color='#DC143C', linewidth=2)
        axes[1,1].set_title('PACF of Log-Differenced Tariff', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Lags', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3, linestyle='--')
        axes[1,1].spines['top'].set_visible(False)
        axes[1,1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率平稳性检验图.png", bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n✅ 平稳性图已保存")

    def select_ar_order(self):
        """选择AR-GARCH的最优AR阶数（AIC准则，李东风教授推荐）"""
        best_aic = np.inf
        self.model["best_ar"] = 1
        print(f"\n=== 最优AR阶数选择（AIC准则） ===")
        for ar in range(1, self.config.max_ar_lag + 1):
            try:
                model = arch_model(
                    self.data["dln_rate"],
                    mean='AR', lags=ar,
                    vol='GARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],  # 先默认GARCH(1,1)
                    dist='Normal'
                )
                res = model.fit(disp='off', options={'maxiter': 1000})
                print(f"AR({ar}) - AIC: {res.aic:.4f}")
                if res.aic < best_aic:
                    best_aic = res.aic
                    self.model["best_ar"] = ar
            except:
                print(f"AR({ar}) - 拟合失败")
                continue
        print(f"最优AR阶数：AR({self.model['best_ar']})（AIC最小：{best_aic:.4f}）")

    # ---------------------- 新增：GARCH模型拟合+诊断（按李东风教授流程） ----------------------
    def fit_garch(self):
        """拟合AR-GARCH模型（修复对数似然属性名错误）"""
        # 按李东风教授推荐：先AR(p)建模，再GARCH(1,1)捕捉波动
        self.model["garch_model"] = arch_model(
            self.data["dln_rate"],
            mean='AR', lags=self.model["best_ar"],
            vol='GARCH', p=self.config.garch_order[0], q=self.config.garch_order[1],
            dist='Normal'  # 基础模型用正态分布，后续可扩展t分布
        )
        self.model["garch_result"] = self.model["garch_model"].fit(disp='off', options={'maxiter': 1000})
        
        # 提取GARCH残差和标准化残差（用于诊断）
        self.data["garch_resid"] = self.model["garch_result"].resid
        self.data["garch_std_resid"] = self.data["garch_resid"] / self.model["garch_result"].conditional_volatility
        
        # 输出GARCH模型详细结果（修复系数名格式+对数似然属性名）
        print(f"\n=== AR({self.model['best_ar']})-GARCH{self.config.garch_order}模型结果 ===")
        print(f"1. 均值方程系数（AR项）:")
        # 新版本arch的AR项参数名格式：mean['AR', 1], mean['AR', 2], ...
        for i in range(self.model["best_ar"]):
            param_name = f"mean['AR', {i+1}]"
            if param_name in self.model["garch_result"].params.index:
                coef = self.model["garch_result"].params[param_name]
                p_val = self.model["garch_result"].pvalues[param_name]
                print(f"   AR({i+1}): {coef:.6f}（p={p_val:.4f} {'*' if p_val<0.05 else ''}）")
            else:
                print(f"   AR({i+1}): 系数不存在（模型未估计）")
        print(f"\n2. 波动率方程系数（GARCH项）:")
        # 波动率方程参数名：omega, alpha[1], beta[1]（格式不变）
        print(f"   常数项(omega): {self.model['garch_result'].params['omega']:.6f}（p={self.model['garch_result'].pvalues['omega']:.4f} {'*' if self.model['garch_result'].pvalues['omega']<0.05 else ''}）")
        print(f"   ARCH项(alpha1): {self.model['garch_result'].params['alpha[1]']:.6f}（p={self.model['garch_result'].pvalues['alpha[1]']:.4f} {'*' if self.model['garch_result'].pvalues['alpha[1]']<0.05 else ''}）")
        print(f"   GARCH项(beta1): {self.model['garch_result'].params['beta[1]']:.6f}（p={self.model['garch_result'].pvalues['beta[1]']:.4f} {'*' if self.model['garch_result'].pvalues['beta[1]']<0.05 else ''}）")
        print(f"   波动率持续性(alpha1+beta1): {self.model['garch_result'].params['alpha[1]'] + self.model['garch_result'].params['beta[1]']:.4f}（<1，平稳）")
        print(f"\n3. 模型拟合优度:")
        print(f"   AIC: {self.model['garch_result'].aic:.4f} | BIC: {self.model['garch_result'].bic:.4f}")
        # 修复：ARCHModelResult的对数似然属性名是loglikelihood，而非llf
        print(f"   对数似然: {self.model['garch_result'].loglikelihood:.4f}")

    def garch_diagnosis(self):
        """GARCH模型诊断（使用手动实现的Ljung-Box检验，兼容statsmodels 0.14.5）"""
        print(f"\n=== AR-GARCH模型诊断（李东风教授流程） ===")
        # 1. 残差ACF检验（手动实现Ljung-Box）
        q_stat, acf_p_val = self.ljung_box_test(self.data["garch_std_resid"], lags=12)
        self.model["garch_diag"]["resid_acf_p"] = acf_p_val
        print(f"1. 标准化残差ACF检验（Ljung-Box）:")
        print(f"   Q统计量: {q_stat:.4f} | p值: {acf_p_val:.4f}")
        print(f"   结论: {'残差无自相关（波动已充分捕捉）' if acf_p_val > 0.05 else '残差存在自相关（需调整模型）'}")
        
        # 2. 残差正态性检验（Shapiro-Wilk检验）
        norm_test = stats.shapiro(self.data["garch_std_resid"].dropna())
        self.model["garch_diag"]["resid_norm_p"] = norm_test.pvalue
        print(f"\n2. 标准化残差正态性检验（Shapiro-Wilk）:")
        print(f"   p值: {self.model['garch_diag']['resid_norm_p']:.4f}")
        print(f"   结论: {'残差近似正态分布' if self.model['garch_diag']['resid_norm_p'] > 0.05 else '残差偏离正态分布（可尝试t分布）'}")
    
    def plot_garch_diagnosis(self):
        """绘制GARCH模型诊断图（4个子图：残差时序、残差ACF、残差QQ图、波动率拟合，对齐李东风教授图6.2）"""
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 10
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
        fig.suptitle('AR-GARCH Model Diagnosis (Li Dongfeng FTS Standard)', fontsize=16, fontweight='bold', y=0.95)
        warmup = max(self.model["best_ar"], self.config.garch_order[1])  # 跳过模型预热期
        diag_dates = self.data["dln_dates"][warmup:]
        
        # 子图1：标准化残差时序（检验波动聚类是否消除）
        axes[0,0].plot(diag_dates, self.data["garch_std_resid"][warmup:], color='#2F4F4F', linewidth=1.5)
        axes[0,0].axhline(y=0, color='#DC143C', linestyle='--', linewidth=2)
        axes[0,0].set_title('Standardized Residuals (No Volatility Clustering)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Date', fontweight='bold')
        axes[0,0].set_ylabel('Standardized Residuals', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3, linestyle='--')
        axes[0,0].spines['top'].set_visible(False)
        axes[0,0].spines['right'].set_visible(False)
        
        # 子图2：标准化残差ACF（检验自相关）
        plot_acf(self.data["garch_std_resid"][warmup:], lags=24, ax=axes[0,1], alpha=0.05, color='#2F4F4F', linewidth=2)
        axes[0,1].set_title(f'Residual ACF (p={self.model["garch_diag"]["resid_acf_p"]:.4f})', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Lags', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3, linestyle='--')
        axes[0,1].spines['top'].set_visible(False)
        axes[0,1].spines['right'].set_visible(False)
        
        # 子图3：残差QQ图（检验正态性）
        stats.probplot(self.data["garch_std_resid"][warmup:].dropna(), dist="norm", plot=axes[1,0])
        axes[1,0].set_title(f'Residual QQ Plot (Normality p={self.model["garch_diag"]["resid_norm_p"]:.4f})', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Theoretical Quantiles', fontweight='bold')
        axes[1,0].set_ylabel('Sample Quantiles', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3, linestyle='--')
        axes[1,0].spines['top'].set_visible(False)
        axes[1,0].spines['right'].set_visible(False)
        
        # 子图4：实际波动vsGARCH拟合波动（检验波动率拟合效果）
        actual_vol = self.data["dln_rate"][warmup:].rolling(window=1).std()  # 实际月度波动
        fitted_vol = self.model["garch_result"].conditional_volatility[warmup:]
        axes[1,1].plot(diag_dates, actual_vol, color='#DC143C', linewidth=2, label='Actual Volatility (Monthly)')
        axes[1,1].plot(diag_dates, fitted_vol, color='#2F4F4F', linewidth=2.5, linestyle='--', label='GARCH Fitted Volatility')
        axes[1,1].set_title('Actual vs GARCH Fitted Volatility', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Date', fontweight='bold')
        axes[1,1].set_ylabel('Volatility (Log-Differenced Tariff)', fontweight='bold')
        axes[1,1].legend(loc='upper right', frameon=True, shadow=True)
        axes[1,1].grid(True, alpha=0.3, linestyle='--')
        axes[1,1].spines['top'].set_visible(False)
        axes[1,1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.garch_diag_dir}GARCH模型诊断图.png", bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ GARCH模型诊断图已保存至：{self.config.garch_diag_dir}")

    # ---------------------- 原有预测逻辑优化（新增波动率预测） ----------------------
    def forecast_tariff(self):
        """预测未来关税率+波动率（按李东风教授预测流程，新增波动率输出）"""
        fc = self.model["garch_result"].forecast(horizon=self.config.forecast_horizon, method='analytic')  # 解析法更稳定
        dln_mean = fc.mean.iloc[-1].values
        dln_vol = fc.variance.iloc[-1].values ** 0.5  # 波动率（标准差）
        
        # 对数差分阶段平滑（减少锯齿）
        dln_mean_smoothed = pd.Series(dln_mean).rolling(window=3, min_periods=1).mean().values
        
        # 补全预测边界
        self.forecast["tariff_mean"] = dln_mean_smoothed
        self.forecast["tariff_lower"] = dln_mean_smoothed - 1.96 * dln_vol
        self.forecast["tariff_upper"] = dln_mean_smoothed + 1.96 * dln_vol
        self.forecast["tariff_vol"] = dln_vol  # 新增：预测波动率
        
        # 还原为原始关税率
        last_ln = self.data["ln_rate"].iloc[-1]
        self.forecast["tariff_rate"] = np.exp(np.cumsum(dln_mean_smoothed) + last_ln)
        self.forecast["tariff_rate"] = np.clip(self.forecast["tariff_rate"], 0, 50)  # 0~50%合理范围
        
        # 预测时间索引
        last_date = self.data["dln_dates"].iloc[-1]
        self.forecast["tariff_dates"] = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=self.config.forecast_horizon, freq='M')
        print(f"\n=== 关税+波动率预测完成 ===")
        print(f"预测时间：{self.forecast['tariff_dates'].min().strftime('%Y-%m')} ~ {self.forecast['tariff_dates'].max().strftime('%Y-%m')}")
        print(f"预测波动率范围：{np.min(self.forecast['tariff_vol']):.4f} ~ {np.max(self.forecast['tariff_vol']):.4f}")

    # ---------------------- GDP联动逻辑（保留+适配） ----------------------
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
        self.data["raw_gdp"].columns = ["Date", "Nominal_GDP_Index", "Real_GDP_Index"]
        
        # 3. 解析日期（适配“1992 - Jan”格式）
        print(f"正在解析日期...")
        self.data["raw_gdp"]["Date"] = pd.to_datetime(
            self.data["raw_gdp"]["Date"],
            format='%Y - %b',
            errors='coerce'
        )
        
        # 4. 清理无效数据
        self.data["raw_gdp"] = self.data["raw_gdp"].sort_values("Date").dropna(subset=["Date", "Real_GDP_Index"])
        if len(self.data["raw_gdp"]) == 0:
            raise ValueError("GDP数据解析失败，请检查Date列格式！")
        
        # 5. 提取实际GDP指数并平稳化
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
        
        # 输出关税对GDP的影响系数
        print(f"\n=== VAR-EGARCH联动模型结果 ===")
        print(f"关税对GDP的短期影响系数（滞后1期）：{self.model['var_result'].coefs[0, 1, 0]:.6f}")
        print(f"关税对GDP的长期影响系数（累计）：{np.sum(self.model['var_result'].coefs[:, 1, 0]):.6f}")
        print(f"模型拟合优度：AIC={self.model['var_result'].aic:.4f} | BIC={self.model['var_result'].bic:.4f}")
        
        # 预测GDP（基于关税预测值）
        last_hist_data = var_input[-self.config.var_lag:]
        forecast_tariff_dln = self.forecast["tariff_mean"].reshape(-1, 1)
        
        # 逐期预测GDP的对数差分
        gdp_forecast_dln = []
        current_input = last_hist_data.copy()
        for i in range(self.config.forecast_horizon):
            next_gdp_dln = self.model["var_result"].forecast(current_input, steps=1)[0, 1]
            gdp_forecast_dln.append(next_gdp_dln)
            # 更新输入窗口
            next_input = np.zeros_like(current_input)
            next_input[:-1] = current_input[1:]
            next_input[-1] = [forecast_tariff_dln[i, 0], next_gdp_dln]
            current_input = next_input
        
        # 还原为原始GDP预测值
        last_gdp_ln = np.log(self.data["gdp_monthly"]["Real_GDP_Index"].iloc[-1])
        self.forecast["gdp_forecast"] = np.exp(np.cumsum(gdp_forecast_dln) + last_gdp_ln)
        print(f"GDP预测完成：{len(self.forecast['gdp_forecast'])}个月度观测值")

    # ---------------------- 可视化增强（对齐学术标准） ----------------------
    def plot_tariff_forecast_with_vol(self):
        """绘制关税+波动率预测图（新增波动率区间，对齐李东风教授图6.3）"""
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
        # 历史数据（最近80个月）
        hist_months = min(80, len(self.data["rate_clean"]))
        ax.plot(
            self.data["dates_clean"][-hist_months:], 
            self.data["rate_clean"][-hist_months:], 
            color='#2F4F4F', linewidth=2.5, label=f'Historical Tariff Rate (Last {hist_months} Months)',
            marker='o', markersize=4
        )
        # 预测数据（均值+波动率区间）
        ax.plot(
            self.forecast["tariff_dates"], 
            self.forecast["tariff_rate"], 
            color='#DC143C', linewidth=3, label='Forecasted Tariff Rate (Mean)',
            marker='s', markersize=5
        )
        # 波动率区间（95%置信区间）
        ax.fill_between(
            self.forecast["tariff_dates"],
            np.exp(np.cumsum(self.forecast["tariff_lower"]) + self.data["ln_rate"].iloc[-1]),
            np.exp(np.cumsum(self.forecast["tariff_upper"]) + self.data["ln_rate"].iloc[-1]),
            color='#DC143C', alpha=0.2, label='95% Confidence Interval'
        )
        # 优化图表样式
        all_rates = np.concatenate([self.data["rate_clean"][-hist_months:].values, self.forecast["tariff_rate"]])
        y_min = all_rates.min() * 0.8
        y_max = all_rates.max() * 1.2
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'AR({self.model["best_ar"]})-GARCH{self.config.garch_order} Tariff Forecast ({self.config.forecast_horizon} Months)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tariff Rate (%)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10.5, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税率+波动率预测图.png", bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 关税率+波动率预测图已保存")

    def plot_tariff_gdp_impact(self):
        """绘制关税对GDP的冲击效应+预测图（优化学术样式）"""
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=300)
        
        # 子图1：关税预测+历史对比
        hist_months = min(60, len(self.data["rate_clean"]))
        axes[0].plot(self.data["dates_clean"][-hist_months:], self.data["rate_clean"][-hist_months:], color='#2F4F4F', linewidth=2.5, label='Historical Tariff Rate', marker='o', markersize=4)
        axes[0].plot(self.forecast["tariff_dates"], self.forecast["tariff_rate"], color='#DC143C', linewidth=3, label='Forecasted Tariff Rate', marker='s', markersize=5)
        axes[0].set_title('U.S. Tariff Rate (Historical + Forecast)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date', fontweight='bold')
        axes[0].set_ylabel('Tariff Rate (%)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # 子图2：GDP真实值+预测值
        axes[1].plot(self.data["gdp_monthly"]["Date"][-hist_months:], self.data["gdp_monthly"]["Real_GDP_Index"][-hist_months:], color='#006400', linewidth=2.5, label='Historical Real GDP Index', marker='o', markersize=4)
        axes[1].plot(self.forecast["gdp_dates"], self.forecast["gdp_forecast"], color='#FF8C00', linewidth=3, label='Forecasted Real GDP Index', marker='s', markersize=5)
        axes[1].set_title('U.S. Real GDP Index (Historical + Forecast) - Tariff Impact', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_ylabel('Real GDP Index', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税-GDP联动预测图.png", bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 关税-GDP联动预测图已保存")

    def plot_impulse_response(self):
        """绘制脉冲响应图（优化学术样式）"""
        irf = self.model["var_result"].irf(12)  # 12期脉冲响应
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
        
        # 提取脉冲响应值和标准误
        irf_data = irf.irfs[:, 1, 0]  # 期数→响应变量(GDP)→冲击变量(关税)
        irf_stderr = irf.stderr()
        irf_std = irf_stderr[:, 1, 0]
        
        # 绘制脉冲响应曲线+95%置信区间
        x_horizon = range(len(irf_data))
        ax.plot(x_horizon, irf_data, color='#DC143C', linewidth=3, marker='o', markersize=5)
        ax.axhline(y=0, color='#2F4F4F', linestyle='--', linewidth=2)
        ax.fill_between(
            x_horizon,
            irf_data - 1.96 * irf_std,
            irf_data + 1.96 * irf_std,
            color='#DC143C', alpha=0.2
        )
        
        # 优化x轴标签
        ax.set_xticks(range(1, len(irf_data)))
        ax.set_xticklabels([f'{i}' for i in range(1, len(irf_data))])
        ax.set_title('Impulse Response: Tariff Rate Shock → GDP Growth', fontsize=14, fontweight='bold')
        ax.set_xlabel('Horizon (Months)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Response of Log-Differenced GDP Index', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}关税对GDP脉冲响应图.png", bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 脉冲响应图已保存")

    # ---------------------- 结果保存拓展（新增GARCH诊断报告） ----------------------
    def save_combined_results(self):
        """保存关税+GDP联动预测结果（新增GARCH诊断报告）"""
        # 1. 关税预测结果（含波动率）
        tariff_forecast_df = pd.DataFrame({
            "Forecast_Date": self.forecast["tariff_dates"].strftime('%Y-%m'),
            "Forecasted_Tariff_Rate(%)": self.forecast["tariff_rate"].round(4),
            "Forecasted_Volatility": self.forecast["tariff_vol"].round(6),
            "DLN_Tariff_Mean": self.forecast["tariff_mean"].round(6),
            "DLN_Tariff_Lower(95%)": self.forecast["tariff_lower"].round(6),
            "DLN_Tariff_Upper(95%)": self.forecast["tariff_upper"].round(6)
        })
        tariff_forecast_df.to_excel(f"{self.config.save_dir}关税率+波动率预测结果.xlsx", index=False)
        
        # 2. GDP预测结果
        gdp_forecast_df = pd.DataFrame({
            "Forecast_Date": self.forecast["gdp_dates"].strftime('%Y-%m'),
            "Forecasted_Real_GDP_Index": self.forecast["gdp_forecast"].round(4),
            "Corresponding_Tariff_Rate(%)": self.forecast["tariff_rate"].round(4),
            "Corresponding_Tariff_Volatility": self.forecast["tariff_vol"].round(6)
        })
        gdp_forecast_df.to_excel(f"{self.config.save_dir}GDP预测结果.xlsx", index=False)
        
        # 3. GARCH模型详细报告（按李东风教授格式）
        with open(f"{self.config.garch_diag_dir}AR-GARCH模型诊断报告.txt", 'w', encoding='utf-8') as f:

            f.write("AR-GARCH Model Detailed Report (Li Dongfeng FTS Standard)\n")
            f.write("="*80 + "\n")
            f.write(f"1. 模型设置:\n")
            f.write(f"   - AR阶数: AR({self.model['best_ar']})\n")
            f.write(f"   - GARCH阶数: GARCH{self.config.garch_order}\n")
            f.write(f"   - 数据范围: {self.data['dln_dates'].min().strftime('%Y-%m')} ~ {self.data['dln_dates'].max().strftime('%Y-%m')}\n")
            f.write(f"   - 观测值数量: {len(self.data['dln_rate'])}\n\n")
            

                                    # 在save_combined_results的GARCH报告部分
            f.write("2. 均值方程系数（AR项）:\n")
            for i in range(self.model["best_ar"]):
                param_name = f"mean['AR', {i+1}]"
                if param_name in self.model["garch_result"].params.index:
                    coef = self.model["garch_result"].params[param_name]
                    p_val = self.model["garch_result"].pvalues[param_name]
                    f.write(f"   AR({i+1}): {coef:.6f} (p={p_val:.4f} {'*' if p_val<0.05 else ''})\n")
                else:
                    f.write(f"   AR({i+1}): 系数未估计\n")
            # f.write("2. 均值方程系数（AR项）:\n")
            # for i in range(self.model["best_ar"]):
            #     coef = self.model["garch_result"].params[f'AR.{i+1}']
            #     p_val = self.model["garch_result"].pvalues[f'AR.{i+1}']
            #     f.write(f"   AR({i+1}): {coef:.6f} (p={p_val:.4f} {'*' if p_val<0.05 else ''})\n")
            f.write(f"\n3. 波动率方程系数（GARCH项）:\n")
            f.write(f"   常数项(omega): {self.model['garch_result'].params['omega']:.6f} (p={self.model['garch_result'].pvalues['omega']:.4f} {'*' if self.model['garch_result'].pvalues['omega']<0.05 else ''})\n")
            f.write(f"   ARCH项(alpha1): {self.model['garch_result'].params['alpha[1]']:.6f} (p={self.model['garch_result'].pvalues['alpha[1]']:.4f} {'*' if self.model['garch_result'].pvalues['alpha[1]']<0.05 else ''})\n")
            f.write(f"   GARCH项(beta1): {self.model['garch_result'].params['beta[1]']:.6f} (p={self.model['garch_result'].pvalues['beta[1]']:.4f} {'*' if self.model['garch_result'].pvalues['beta[1]']<0.05 else ''})\n")
            f.write(f"   波动率持续性(alpha1+beta1): {self.model['garch_result'].params['alpha[1]'] + self.model['garch_result'].params['beta[1]']:.4f} (<1，平稳)\n\n")
            
            f.write("4. 模型拟合优度:\n")
            f.write(f"   - AIC: {self.model['garch_result'].aic:.4f}\n")
            f.write(f"   - BIC: {self.model['garch_result'].bic:.4f}\n")
            f.write(f"   - 对数似然: {self.model['garch_result'].loglikelihood:.4f}\n\n")
            
            f.write("5. 模型诊断结果:\n")
            f.write(f"   - 标准化残差ACF检验（Ljung-Box, lags=12）: p={self.model['garch_diag']['resid_acf_p']:.4f} {'→ 残差无自相关' if self.model['garch_diag']['resid_acf_p']>0.05 else '→ 残差存在自相关'}\n")
            f.write(f"   - 标准化残差正态性检验（Shapiro-Wilk）: p={self.model['garch_diag']['resid_norm_p']:.4f} {'→ 近似正态' if self.model['garch_diag']['resid_norm_p']>0.05 else '→ 偏离正态'}\n")
        
        # 4. VAR-EGARCH联动模型结果
        with open(f"{self.config.save_dir}VAR-EGARCH联动模型结果.txt", 'w', encoding='utf-8') as f:
            f.write("VAR-EGARCH Model Summary (Tariff → GDP)\n")
            f.write("="*80 + "\n")
            f.write(f"VAR Lag Order: {self.config.var_lag}\n")
            f.write(f"GARCH Order: {self.config.garch_order}\n\n")
            f.write("Tariff → GDP Impact Coefficients (Lag 1-6):\n")
            for i in range(self.config.var_lag):
                coef = self.model["var_result"].coefs[i, 1, 0]
                f.write(f"  Lag {i+1}: {coef:.6f}\n")
            f.write(f"\nLong-Term Impact Coefficient: {np.sum(self.model['var_result'].coefs[:, 1, 0]):.6f}\n")
            f.write(f"VAR Model Fit Metrics:\n")
            f.write(f"  AIC: {self.model['var_result'].aic:.4f}\n")
            f.write(f"  BIC: {self.model['var_result'].bic:.4f}\n")
            f.write(f"  Log Likelihood: {self.model['var_result'].llf:.4f}\n")
        
        print(f"\n✅ 所有结果已保存至：{self.config.save_dir}")
        print(f"包含：关税率+波动率预测.xlsx、GDP预测.xlsx、GARCH诊断报告.txt、VAR联动结果.txt、4张可视化图")

# ---------------------- 主流程（按李东风教授GARCH流程优化） ----------------------
def main():
    # 初始化配置和分析器
    config = Config()
    analyzer = TariffRateAnalyzer(config)
    
    # 第一步：关税数据预处理+平稳性检验（李东风教授流程第一步）
    analyzer.load_and_preprocess_tariff()
    analyzer.detect_outliers()
    analyzer.plot_stationarity()
    
    # 第二步：AR-GARCH模型拟合+诊断（核心步骤，新增诊断）
    analyzer.select_ar_order()
    analyzer.fit_garch()
    analyzer.garch_diagnosis()  # 新增：模型诊断
    analyzer.plot_garch_diagnosis()  # 新增：诊断可视化
    
    # 第三步：关税+波动率预测（新增波动率输出）
    analyzer.forecast_tariff()
    analyzer.plot_tariff_forecast_with_vol()  # 新增：关税+波动率预测图
    
    # 第四步：GDP联动建模+预测
    analyzer.load_and_preprocess_gdp()
    analyzer.combine_tariff_gdp()
    analyzer.fit_var_egarch()
    analyzer.plot_tariff_gdp_impact()
    analyzer.plot_impulse_response()
    
    # 第五步：结果保存（新增GARCH诊断报告）
    analyzer.save_combined_results()

if __name__ == "__main__":
    main()