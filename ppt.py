import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 데이터
labels = ['Drowning', 'Floundering', 'Normal', 'Slipping', 'Swimming']
sizes = [3496, 8309, 37058, 10290, 59290]
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#e41a1c', '#984ea3']  # 각각의 색상

# 주요 데이터 포맷팅
total = sum(sizes)
percentages = [size / total * 100 for size in sizes]

# 원형 그래프
fig, ax = plt.subplots()

wedges, texts, autotexts = ax.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%',
    startangle=140, 
    colors=colors,
    wedgeprops=dict(width=0.3, edgecolor='w')
)

# 가운데 텍스트 추가
centre_circle = plt.Circle((0, 0), 0.5, color='white', fc='white')
fig.gca().add_artist(centre_circle)
ax.text(0, 0, '(Situation %)', ha='center', va='center', fontsize=10, fontweight='bold', color='gray')

# 퍼센트 레이블 조정
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color("black")

# 스타일 설정
plt.setp(texts, size=12, weight="bold")
ax.set_title("Situation Class Distribution", fontsize=16, fontweight='bold', pad=20)

plt.show()