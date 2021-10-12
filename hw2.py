import re
import pandas as pd
import numpy as np


train = pd.read_csv("data/wikipedia_train.csv")
test = pd.read_csv("data/wikipedia_test.csv")


# Общее количество просмотров за период из тренировочных данных
train["Total Views"] = train.iloc[:, 1:429].sum(axis=1)

# Медианное значени просмотров за весь период
train["Median Views"] = train.iloc[:, 1:429].median(axis=1)

# Среднее значение просмотров за весь период.
train["Mean Views"] = train.iloc[:, 1:429].mean(axis=1)

# Медианное значение за последние 30 дней
train["Median 30"] = train.iloc[:, (429 - 30):429].median(axis=1)

# Медианное значение за последние 7 дней
train["Median 7"] = train.iloc[:, (429 - 7):429].median(axis=1)

# Медианное значение за последние 60 дней
train["Median 60"] = train.iloc[:, (429 - 60):429].median(axis=1)

# Подсчет медиан по дням недели
train["median_day_2"] = train.iloc[:, 1:429].iloc[:, 0::7].median(axis=1)
train["median_day_3"] = train.iloc[:, 1:429].iloc[:, 1::7].median(axis=1)
train["median_day_4"] = train.iloc[:, 1:429].iloc[:, 2::7].median(axis=1)
train["median_day_5"] = train.iloc[:, 1:429].iloc[:, 3::7].median(axis=1)
train["median_day_6"] = train.iloc[:, 1:429].iloc[:, 4::7].median(axis=1)
train["median_day_0"] = train.iloc[:, 1:429].iloc[:, 5::7].median(axis=1)
train["median_day_1"] = train.iloc[:, 1:429].iloc[:, 6::7].median(axis=1)


# Количество страниц на русском языке
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[0:2]
    return 'na'


train['Language'] = train['Page'].apply(lambda x: get_language(str(x)))
ru_pages = train.loc[train["Language"] == "ru"]
print("Total ru pages =", len(ru_pages), "\n")


# Самая популярная русскоязычная страница
def most_popular_by_row(df, column_name):
    max_value = df[column_name].max()
    most_popular = df[df[column_name] == max_value]
    print("Most popular ru page by", column_name, "=", most_popular["Page"].values, "with", max_value, "views")


for column in ["Total Views", "Median Views", "Mean Views"]:
    most_popular_by_row(ru_pages, column)
print("\n")


# Метрика для валидации результатов
def pandas_smape(df, colomn):
    smape_column_name = "SMAPE " + colomn
    df.fillna(0, inplace=True)
    df[smape_column_name] = 200 * np.abs(df["Visits"] - df[colomn]) / (df["Visits"] + df[colomn])
    df[smape_column_name].fillna(0, inplace=True)
    return np.mean(df[smape_column_name])


prediction = pd.melt(test, id_vars=["Page"], value_vars=test.columns[1:], value_name="Visits", var_name="date")

# Все константные предсказания независящие от дня из тестовых данных
prediction = prediction.merge(train[["Page", "2016-08-31", "Median Views", "Mean Views", "Median 60", "Median 30", "Median 7"]], how='inner', on="Page")


# start_time = time.time()
# Предсказание на основании медиан построенных отдельно для каждого дня недели
prediction["day"] = pd.to_datetime(prediction["date"], format="%Y-%m-%d").dt.dayofweek


def get_median_for_day_of_week(page, day_of_week):
    return train[train["Page"] == page].iloc[0].loc[f"median_day_{day_of_week}"]


prediction["pred_median_days_of_week"] = prediction.apply(lambda row: get_median_for_day_of_week(row[0], row[9]), axis=1)


# Вывод оценок сделанных предсказаний предсказаний
for pred in ["2016-08-31", "Median Views", "Mean Views", "Median 60", "Median 30", "Median 7", "pred_median_days_of_week"]:
    print("SMAPE for", pred, ":", pandas_smape(prediction, pred))
