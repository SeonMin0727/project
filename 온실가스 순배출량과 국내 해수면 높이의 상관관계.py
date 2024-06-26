import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'NanumGothic.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

df_해수면높이 = pd.read_excel('국내 해수면 높이.xlsx')
df_온실가스= pd.read_excel('국내 온실가스 통계.xlsx')

df = pd.merge(df_해수면높이[['연도', '인천', '군산', '목포', '제주', '완도', 
                        '여수', '부산', '울산', '포항', '속초', '울릉도']],
              df_온실가스[['연도', '순배출량']], on='연도')

correlation = df.corr()['순배출량'][1:-1]

fig, ax = plt.subplots(figsize=(10, 6))
correlation.plot(kind='bar', ax=ax)
ax.set_xlabel('지역')
ax.set_ylabel('상관관계 (0-1)')
ax.set_title('순배출량과 국내 해수면 높이의 상관관계')

plt.xticks(rotation=0)
plt.show()