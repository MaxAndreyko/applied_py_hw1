import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import streamlit as st

sns.set_theme(font="HelveticaNeueCyr")

url = "https://raw.githubusercontent.com/MaxAndreyko/applied_py_hw1/main/final_bd.csv"
df_source = pd.read_csv(url, sep=";", index_col=0)
df_source.drop("ID_CLIENT", inplace=True, axis=1)

df = df_source.copy()
df["TARGET"] = df_source["TARGET"].replace(0, "Не откликнулись").replace(1, "Откликнулись")
df["GENDER"] = df_source["GENDER"].replace(0, "Женщина").replace(1, "Мужчина")
df["SOCSTATUS_WORK_FL"] = df_source["SOCSTATUS_WORK_FL"].replace(0, "Не работает").replace(1, "Работает")
df["SOCSTATUS_PENS_FL"] = df_source["SOCSTATUS_PENS_FL"].replace(0, "Не пенсионер").replace(1, "Пенсионер")


def write_header(text, lvl=1):
    header_lvl = '#'*lvl
    st.markdown(f"{header_lvl} {text}")


def _single(ax, indent=0.02):
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + (p.get_height() * indent)
        value = '{:.1f}%'.format(p.get_height())
        ax.text(_x, _y, value, ha="center")


@st.cache_data
def plot_count_target(data: pd.DataFrame, target_col="TARGET") -> None:
    fig = plt.figure(figsize=(10, 7))

    target_ratio = data[target_col].value_counts()
    target_ratio = target_ratio / target_ratio.sum() * 100
    target_ratio = target_ratio.to_frame().reset_index().rename({"count": "Percent"}, axis=1)

    plot = sns.barplot(data=target_ratio, x=target_col, y="Percent")
    for idx, ax in np.ndenumerate(plot):
        _single(ax)

    plt.title('Процент неоткликнувшихся и откликнувшихся клиентов')
    st.pyplot(fig)


@st.cache_data
def plot_count_features(data, select_feature, target_col="TARGET") -> None:
    fig = plt.figure(figsize=(10, 7))
    sns.countplot(data=data, x=select_feature, hue=target_col, stat="percent")
    plt.title(f'Распределение признака {select_feature} относительно целевой переменной')
    st.pyplot(fig)


@st.cache_data
def plot_dist_features(data, select_feature, stat="count") -> None:
    fig = plt.figure(figsize=(10, 7))
    sns.histplot(data=data[select_feature], kde=False, label=select_feature, stat=stat)
    plt.title(f'Распределение признака {select_feature}')
    st.pyplot(fig)


@st.cache_data
def plot_boxplot_features(data, select_feature) -> None:
    fig = plt.figure(figsize=(10, 7))
    sns.boxplot(data=data, x=select_feature)
    st.pyplot(fig)


@st.cache_data
def plot_boxplot_target(data, select_feature, target_col="TARGET") -> None:
    fig = plt.figure(figsize=(10, 7))
    sns.boxplot(data=data, x=select_feature, hue=target_col)
    st.pyplot(fig)


@st.cache_data
def plot_corr_heatmap(data) -> None:
    fig = plt.figure(figsize=(10, 7))
    data_corr = data.corr()
    sns.heatmap(data_corr, annot=True, fmt='.2f', cmap="crest")
    plt.title(f'Матрица корреляции признаков')
    st.pyplot(fig)


@st.cache_data
def dispay_describe(data, select_feature):
    st.write(data[select_feature].describe().transpose())


@st.cache_data
def plot_numeric_target(data, select_feature, target_col="TARGET") -> None:
    fig = plt.figure(figsize=(10, 7))
    feature_col = data[select_feature]
    feature_intervals = pd.cut(feature_col, bins=6, precision=0)
    sns.countplot(data=pd.concat([feature_intervals, data[target_col]], axis=1), x=select_feature, hue=target_col, stat="percent")
    plt.title(f'Распределение признака {select_feature} относительно целевой переменной')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # value_counts = feature_intervals.value_counts(sort=False)
    # print(fe)


def main() -> None:
    write_header("Разведочный анализ данных")
    write_header("Распределение классов", lvl=2)
    plot_count_target(df)
    st.markdown(f"Классы сильно несбалансированны")

    write_header("Распределение категориальных признаков", lvl=2)
    cat_features = ["GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
    select_feature = st.selectbox("Выберите категориальный признак", cat_features)
    plot_dist_features(df, select_feature)
    st.markdown("Один из типов внутри категориального признака всегда преобладает минимум в 2 раза")

    write_header("Распределение непрерывных признаков", lvl=2)
    float_features = ["PERSONAL_INCOME", "AGE"]
    select_feature = st.selectbox("Выберите непрерывный признак", float_features)
    plot_dist_features(df, select_feature, stat="frequency")
    plot_boxplot_features(df, select_feature)
    st.markdown("""
        В случае дохода (PERSONAL_INCOME) подавляющее большинство наблюдений лежит в диапазоне ~2000 - 50000.
        Какой-то закономерности в распределении возраста нет. Большинство клиентов это люди от 30 до 50    
        """)

    write_header("Матрица корреляции", lvl=2)
    plot_corr_heatmap(df_source)
    st.markdown("Все признаки очень слабо коррелируют, как между собой (за исключением SOCSTATUS_WORK_FL/SOCSTATUS_PENS_FL) и относительно целевой переменной")

    write_header("Распределение дискретных признаков относительно целевой переменной", lvl=2)
    features = ["DEPENDANTS", "CHILD_TOTAL", "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED", "GENDER", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL"]
    select_feature = st.selectbox("Выберите дискретный признак", features)
    plot_count_features(df, select_feature)
    st.markdown("Нет зависимости отклика от признаков. Большинство людей не откликаются на предложение")

    write_header("Распределение непрерывных признаков относительно целевой переменной", lvl=2)
    select_feature = st.selectbox("Выберите признак", ["PERSONAL_INCOME", "AGE"])
    plot_numeric_target(df, select_feature)
    plot_boxplot_target(df, select_feature)
    st.markdown("""
            Из-за того, что наибольшее число клиентов получают доход в диапазоне 17000 - 50000, то
            также видно, что зависимость отклика от дохода слабая.
            Схожая ситуация и с распределением по возрасту. Но с большой натяжкой можно сказать, что чем моложе
            клиент, тем с большей вероятностью от отреагирует на предложение.
        """)

    write_header("Описательные характеристики", lvl=2)
    select_feature = st.selectbox("Выберите признак", df.columns)
    dispay_describe(df, select_feature)


if __name__ == "__main__":
    main()
