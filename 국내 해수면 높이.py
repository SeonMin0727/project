import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = 'NanumGothic.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name

df_해수면높이 = pd.read_excel('국내 해수면 높이.xlsx', dtype={'연도': int})

df_해수면높이.plot.box(x='연도', y=['인천', '군산', '목포', '제주', '완도', 
                             '여수', '부산', '울산', '포항', '속초', '울릉도'])

plt.xlabel('연도 (2000-2017)')
plt.ylabel('해수면 높이 (cm)')
plt.title('연도별 국내 해수면 높이 변화')

plt.show()