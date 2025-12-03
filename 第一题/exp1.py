import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from arch import arch_model  # 需安装arch库：pip install arch

# 1. 数据读取（关税月频数据：2010-2024）
tariff_data = pd.read_excel("关税月频数据.xlsx", index_col="date", parse_dates=True)
tariff_series = tariff_data["tariff_rate"]

# 2. 数据预处理：对数差分+ADF检验（参考论文）
dln_tariff = np.log(tariff_series).diff().dropna()  # 对数差分
adf_result = adfuller(dln_tariff)
print(f"关税ADF检验p值：{adf_result[1]:.4f}")  # p<0.05说明平稳

# 3. AR-GARCH模型拟合与预测（借鉴论文3.3节）
model = arch_model(
    dln_tariff, mean="AR", lags=6, vol="GARCH", p=1, q=1  # AR(6)+GARCH(1,1)
)
model_fit = model.fit(disp="off")

# 4. 预测未来12个月关税波动
forecast = model_fit.forecast(horizon=12)
tariff_pred = forecast.mean.dropna()  # 预测均值
tariff_vol = forecast.variance.dropna()  # 预测波动率

# 5. 绘图：关税历史+预测趋势（论文图4风格）
plt.figure(figsize=(12, 6))
plt.plot(dln_tariff.index, dln_tariff, label="Historical Tariff (Log-Diff)")
plt.plot(tariff_pred.index, tariff_pred, "r-", label="Predicted Tariff (12M)")
plt.fill_between(
    tariff_pred.index,
    tariff_pred.values - np.sqrt(tariff_vol.values),
    tariff_pred.values + np.sqrt(tariff_vol.values),
    alpha=0.3, color="red", label="Volatility Interval"
)
plt.xlabel("Date")
plt.ylabel("Log-Differenced Tariff Rate")
plt.title("Tariff Volatility Prediction (AR-GARCH(6,1,1))")
plt.legend()
plt.savefig("关税预测图.png", dpi=300, bbox_inches="tight")
plt.show()
from midaspy import MIDASReg  # 需安装：pip install midaspy

# 1. 数据准备
# 高频数据：关税月频（2010-2024）；低频数据：GDP季频（2010Q1-2024Q4）
gdp_quarter = pd.read_excel("GDP季频数据.xlsx", index_col="quarter", parse_dates=True)
dln_gdp = np.log(gdp_quarter["gdp"]).diff().dropna()  # GDP对数差分

# 2. 构造MIDAS数据（高频滞后项：前6个月关税）
midas_data = pd.DataFrame({"gdp": dln_gdp})
for k in range(6):  # 滞后0-5个月
    midas_data[f"tariff_lag{k}"] = dln_tariff.shift(k).reindex(dln_gdp.index)

# 3. 拟合MIDAS模型（指数权重）
midas_model = MIDASReg(
    y=midas_data["gdp"],
    X=midas_data[["tariff_lag0", "tariff_lag1", "tariff_lag2", "tariff_lag3", "tariff_lag4", "tariff_lag5"]],
    freq_x=12,  # 高频数据频率（月=12）
    freq_y=4,   # 低频数据频率（季=4）
    weight_func="exp_almon"  # 指数衰减权重
)
midas_results = midas_model.fit()
print(midas_results.summary())

# 4. 绘图：MIDAS拟合效果（论文图4风格）
plt.figure(figsize=(10, 5))
plt.plot(dln_gdp.index, dln_gdp, label="Actual GDP Growth")
plt.plot(dln_gdp.index, midas_results.fittedvalues, "r--", label="MIDAS Fitted")
plt.xlabel("Quarter")
plt.ylabel("Log-Differenced GDP Growth")
plt.title("MIDAS Model Fit (Tariff→GDP)")
plt.legend()
plt.savefig("MIDAS拟合图.png", dpi=300)
plt.show()
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.irf import IRAnalysis

# 1. 数据准备（VAR输入：季度数据，3个变量）
var_data = pd.DataFrame({
    "dln_tariff_q": midas_data["gdp"].index.map(lambda x: dln_tariff.loc[x - pd.DateOffset(months=1)]),  # 季度关税代理变量
    "dln_gdp": dln_gdp,
    "dln_exchange": np.log(pd.read_excel("汇率季频数据.xlsx")["exchange"]).diff().dropna()  # 汇率控制变量
}).dropna()

# 2. VAR模型拟合（参考论文4.3节）
var_model = VAR(var_data)
optimal_lag = var_model.select_order(maxlags=4).selected_orders["aic"]  # 最优滞后阶数
var_results = var_model.fit(optimal_lag)

# 3. 模型稳定性检验（特征根图，论文图7风格）
fig, ax = plt.subplots(figsize=(8, 8))
var_results.plot_roots(ax=ax)
plt.title("VAR Model Stability Test (Inverse Roots)")
plt.savefig("VAR稳定性检验.png", dpi=300)
plt.show()

# 4. 脉冲响应分析（论文图8风格）
irf = IRAnalysis(var_results, nobs=8)  # 8期响应
fig = irf.plot(impulse="dln_tariff_q", response="dln_gdp", alpha=0.05)
plt.title("Impulse Response of GDP to Tariff Shock")
plt.savefig("脉冲响应图.png", dpi=300)
plt.show()

# 5. 方差分解（论文图9风格）
fevd = var_results.fevd(periods=8)
fevd.plot(figsize=(10, 5))
plt.title("Variance Decomposition of GDP")
plt.savefig("方差分解图.png", dpi=300)
plt.show()
from synthetic_control import SyntheticControl  # 需安装：pip install synthetic-control

# 1. 数据准备：处理组（中国贸易密切国，如新加坡）、控制组（无往来国，如若干非洲国家）
sc_data = pd.read_excel("SCM数据.xlsx")  # 列：country, year, gdp, tariff（政策变量）
treatment_country = "Singapore"
control_countries = ["CountryA", "CountryB", "CountryC"]  # 无往来国列表

# 2. 拟合SCM模型（以2018年为政策冲击年）
sc = SyntheticControl(
    data=sc_data,
    outcome_var="gdp",
    treatment_var="tariff",
    treatment_unit=treatment_country,
    control_units=control_countries,
    treatment_year=2018,
    pre_treatment_periods=range(2010, 2018),
    post_treatment_periods=range(2018, 2024)
)
sc.fit()

# 3. 绘图：处理组vs合成对照组GDP对比（论文图12风格）
plt.figure(figsize=(12, 6))
plt.plot(sc.pre_treatment_periods, sc.treatment_pre, label=f"{treatment_country} (Actual)")
plt.plot(sc.pre_treatment_periods, sc.synthetic_pre, "r--", label=f"Synthetic Control")
plt.plot(sc.post_treatment_periods, sc.treatment_post, "b-")
plt.plot(sc.post_treatment_periods, sc.synthetic_post, "r--")
plt.axvline(x=2018, color="black", linestyle=":", label="Tariff Policy Shock")
plt.xlabel("Year")
plt.ylabel("GDP (Log Scale)")
plt.title(f"GDP Comparison: {treatment_country} vs Synthetic Control")
plt.legend()
plt.savefig("SCM对比图.png", dpi=300)
plt.show()