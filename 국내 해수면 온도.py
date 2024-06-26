import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'NanumGothic.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

df_해수면온도 = pd.read_excel('국내 해수면 온도.xlsx')

if '연도' not in df_해수면온도.columns:
    df_해수면온도.rename(columns={'A': '연도'}, inplace=True)

df_numeric = df_해수면온도[pd.to_numeric(df_해수면온도['연도'], errors='coerce').notnull()]
df_numeric['연도'] = df_numeric['연도'].astype(int)

start_year = 2000
end_year = 2021
df_filtered = df_numeric[(df_numeric['연도'] >= start_year) & (df_numeric['연도'] <= end_year)]

df_동해 = df_filtered[['연도', '동해']]
df_남해 = df_filtered[['연도', '남해']]
df_서해 = df_filtered[['연도', '서해']]

plt.plot(df_동해['연도'], df_동해['동해'], label='동해')
plt.plot(df_남해['연도'], df_남해['남해'], label='남해')
plt.plot(df_서해['연도'], df_서해['서해'], label='서해')
plt.xlabel('연도')
plt.ylabel('해수면 온도')
plt.title('연도별 동해, 남해, 서해 해수면 온도 변화')
plt.legend()
plt.show()