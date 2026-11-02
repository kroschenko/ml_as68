import pandas as pd

df = pd.read_csv('winequality-white.csv', sep=';')

# функция, которая по числу возвращает категорию
def quality_cat(q):
    if q <= 4:
        return 'плохое'
    elif q <= 6:
        return 'среднее'
    else:
        return 'хорошее'

# создаём новый столбец, применив функцию к каждому значению quality
df['quality_cat'] = df['quality'].apply(quality_cat)

print(df[['quality', 'quality_cat']].head(10))
print()
print(df['quality_cat'].value_counts())