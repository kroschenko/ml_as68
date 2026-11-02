import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('winequality-white.csv', sep=';')

def quality_cat(q):
    if q <= 4:
        return 'плохое'
    elif q <= 6:
        return 'среднее'
    else:
        return 'хорошее'

df['quality_cat'] = df['quality'].apply(quality_cat)

# считаем количество в каждой категории
counts = df['quality_cat'].value_counts()

print(counts)

# строим диаграмму
plt.figure(figsize=(7, 5))
plt.bar(counts.index, counts.values)
plt.title('Количество вин по категориям качества')
plt.xlabel('Категория качества')
plt.ylabel('Количество вин')
plt.show()