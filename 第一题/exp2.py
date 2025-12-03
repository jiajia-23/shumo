import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


import pandas as pd

# ---------------------- 核心：探查表格真实结构 ----------------------
# 读取你的Excel文件（确保文件名与路径正确）
df_tariff = pd.read_excel('DataWeb-Query-Export (2).xlsx')  # 若路径不对，需补全如“D:/xxx/DataWeb-Query-Export (2).xlsx”

# 1. 打印所有实际列名（关键！看列名到底是什么）
print("=== 你的表格所有实际列名 ===")
for idx, col in enumerate(df_tariff.columns):
    print(f"列{idx+1}：'{col}'")  # 加单引号，方便识别列名是否含空格（如“Dutiable  Value”多空格）

# 2. 打印前2行数据（看各列对应的值，确认“国家/年份/月份/应税价值/计算职责”所在列）
print(f"\n=== 表格前2行数据预览 ===")
print(df_tariff.head(2))

# 3. 打印数据形状（总行数、总列数）
print(f"\n=== 数据基本信息 ===")
print(f"总行数：{len(df_tariff)}，总列数：{len(df_tariff.columns)}")




import pandas as pd

# 读取Excel文件的所有子表名称
excel_file = pd.ExcelFile('DataWeb-Query-Export (2).xlsx')
print("=== 该Excel文件包含的所有子表名称 ===")
for sheet_name in excel_file.sheet_names:
    print(f"子表：{sheet_name}")
# ---------------------- 1. 读取数据并验证核心结构 ----------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------- 1. 读取数据+强制数值类型转换 ----------------------
excel_path = 'DataWeb-Query-Export (3).xlsx'

# 读取子表
df_dutiable = pd.read_excel(excel_path, sheet_name='Dutiable Value')
df_calculated = pd.read_excel(excel_path, sheet_name='Calculated Duties')

# 确认中国名称（已验证正确）
china_name = 'China'

# ---------------------- 2. 合并+筛选中国数据 ----------------------
merge_cols = ['Country', 'Year', 'Month']
df_merged = pd.merge(
    df_dutiable[merge_cols + ['Dutiable Value']],
    df_calculated[merge_cols + ['Calculated Duties']],
    on=merge_cols,
    how='inner'
)

# 筛选中国数据
df_china = df_merged[
    (df_merged['Country'] == china_name) & 
    (df_merged['Year'].notna()) & 
    (df_merged['Month'].notna())
].copy()

print(f"✅ 成功筛选到中国数据：{len(df_china)}行")
min_year = int(df_china['Year'].min())
max_year = int(df_china['Year'].max())
print(f"时间范围：{min_year}年 - {max_year}年")

# ---------------------- 3. 核心修复：强制转为浮点型（解决0.00类型问题） ----------------------
# 3.1 转换年份/月份为整数
df_china['Year'] = df_china['Year'].astype(int)
df_china['Month'] = df_china['Month'].astype(int)

# 3.2 强制将金额列转为float（不管原始类型，直接按数值解析）
# 即使是"0.00"字符串，也会被转为0.0浮点型
df_china['Dutiable Value'] = df_china['Dutiable Value'].astype(float)
df_china['Calculated Duties'] = df_china['Calculated Duties'].astype(float)

# 3.3 查看转换后的数据类型（验证是否成功）
print(f"\n=== 数据类型验证 ===")
print(f"Dutiable Value类型：{df_china['Dutiable Value'].dtype}（应为float64）")
print(f"Calculated Duties类型：{df_china['Calculated Duties'].dtype}（应为float64）")

# 3.4 处理0值（统计0值数量，确保合理）
zero_dutiable = (df_china['Dutiable Value'] == 0.0).sum()
zero_calculated = (df_china['Calculated Duties'] == 0.0).sum()
print(f"\n=== 0值统计 ===")
print(f"Dutiable Value为0的行数：{zero_dutiable}")
print(f"Calculated Duties为0的行数：{zero_calculated}")

# 3.5 合并时间列
df_china['Year_Month'] = df_china.apply(
    lambda x: f"{x['Year']}-{x['Month']:02d}",
    axis=1
)
df_china['Date'] = pd.to_datetime(df_china['Year_Month'], format='%Y-%m')

# 3.6 计算关税率（0值已处理，无报错）
df_china['Tariff_Rate(%)'] = np.where(
    df_china['Dutiable Value'] == 0.0,  # 若应税价值为0，关税率设为0
    0.0,
    (df_china['Calculated Duties'] / df_china['Dutiable Value'] * 100).round(2)
)

# 3.7 去除极端异常值（关税率>50%视为不合理，参考实际政策）
df_clean = df_china[df_china['Tariff_Rate(%)'] <= 50.0].sort_values('Date').reset_index(drop=True)

# ---------------------- 4. 输出数据+绘图 ----------------------
# 保存数据
output_cols = ['Date', 'Year_Month', 'Year', 'Month', 'Dutiable Value', 'Calculated Duties', 'Tariff_Rate(%)']
df_clean[output_cols].to_excel('美国对华关税_清洗后数据.xlsx', index=False)
print(f"\n📊 清洗后数据已保存：美国对华关税_清洗后数据.xlsx")
print(f"清洗后有效行数：{len(df_clean)}行")

# 绘制图表
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (16, 8)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# 左轴：金额（转为百万美元，避免数值过大）
line1 = ax1.plot(
    df_clean['Date'],
    df_clean['Dutiable Value'] / 1e6,
    color='#FF7F0E', linewidth=2.5, label='Dutiable Value (Million USD)'
)
line2 = ax1.plot(
    df_clean['Date'],
    df_clean['Calculated Duties'] / 1e6,
    color='#1F77B4', linewidth=2.5, label='Calculated Duties (Million USD)'
)

# 右轴：关税率
line3 = ax2.plot(
    df_clean['Date'],
    df_clean['Tariff_Rate(%)'],
    color='#D62728', linewidth=2.5, linestyle='--', label='Tariff Rate (%)'
)

# 图表美化
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Amount (Million USD)', fontsize=12, fontweight='bold', color='#2C3E50')
ax2.set_ylabel('Tariff Rate (%)', fontsize=12, fontweight='bold', color='#D62728')
plt.title(f'U.S. Tariff on {china_name} ({min_year}-{max_year})', fontsize=14.5, fontweight='bold')

# X轴刻度（每5年1个，清晰不重叠）
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0)

# 图例
ax1.legend(line1+line2+line3, [l.get_label() for l in line1+line2+line3], loc='upper left', fontsize=10.5)

# 网格
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_axisbelow(True)

# 保存图表
plt.tight_layout()
plt.savefig('美国对华关税_时间变化曲线.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('美国对华关税_时间变化曲线.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print(f"\n✅ 图表已保存：美国对华关税_时间变化曲线.png/pdf")