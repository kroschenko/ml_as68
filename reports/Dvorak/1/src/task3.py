import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\питон\project\ml_as68\reports\Dvorak\1\src\winequality-white.csv', sep=';')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Функция категоризации качества вина
def quality_to_category(q):
    if q <= 4:
        return 'плохое'
    elif q <= 6:
        return 'среднее'
    else:
        return 'хорошее'
# Создаём новый категориальный столбец
df['quality_cat'] = df['quality'].apply(quality_to_category)
# Считаем количество вин в каждой категории (фиксируем порядок)
order = ['плохое', 'среднее', 'хорошее']
counts = df['quality_cat'].value_counts().reindex(order)
# Цвета для категорий
colors = ['red', 'orange', 'green']
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.7)
# Подписываем значения над каждым столбцом
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height + 20,
            str(int(height)),
            ha='center',
            fontsize=11)
ax.set_xlabel('Категория качества')
ax.set_ylabel('Количество вин')
ax.set_title('Количество вин каждой категории качества', fontsize=14)
plt.tight_layout()
plt.show()