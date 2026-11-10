import pandas as pd
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
# Создаём новый категориальный столбец на основе quality
df['quality_cat'] = df['quality'].apply(quality_to_category)
print("\nЭто как выглядит с преобразованной Quality:")
print(df[['quality', 'quality_cat']].head(10))
print("\nКоличество образцов каждой категории качества:")
print(df['quality_cat'].value_counts())