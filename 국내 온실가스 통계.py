import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'NanumGothic.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

df_온실가스 = pd.read_excel('국내 온실가스 통계.xlsx')

if '연도' not in df_온실가스.columns:
    df_온실가스.rename(columns={'A': '연도'}, inplace=True)

start_year = 2000
end_year = 2020
df_filtered = df_온실가스[(df_온실가스['연도'] >= start_year) & (df_온실가스['연도'] <= end_year)]

plt.plot(df_filtered['연도'], df_filtered['총배출량'], label='총배출량')
plt.plot(df_filtered['연도'], df_filtered['순배출량'], label='순배출량')
plt.xlabel('연도 (2000-2020)')
plt.ylabel('배출량 (백만톤 CO₂eq.)')
plt.title('연도별 총배출량과 순배출량 변화')
plt.legend()
plt.show()