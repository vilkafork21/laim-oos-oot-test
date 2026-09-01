"""
HTML report helper functions for OOS-OOT stability test.

Модуль содержит функции для визуализации результатов тестов
в виде HTML-светофоров и таблиц с критериями. Используется
для генерации отчетов в Jupyter notebooks и веб-интерфейсе.
"""

import numpy as np
import pandas as pd
from IPython.display import display, HTML


# =============================================================================
# СВЕТОФОРЫ (TRAFFIC LIGHTS)
# =============================================================================

def display_semaphore(colour, width=46, height=18, return_html=False):
    """
    Отображение светофора в виде HTML.

    Args:
        colour: Цвет светофора ('green', 'yellow', 'amber', 'red', 'grey')
        width: Ширина изображения
        height: Высота изображения
        return_html: Если True - возвращает HTML строку, иначе - отображает

    Returns:
        HTML строка светофора (если return_html=True)
    """
    red_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red { background-color: #ca1d1d; }
                .amber { background-color: #ffd600; }
                .green { background-color: #04d930; }
                .gray { background-color: #ccc; }
            </style>
        </head>
        <body>
        <div class="traffic-light">
            <div class="light red"></div>
            <div class="light gray"></div>
            <div class="light gray"></div>
        </div>
        </body>
        </html>
    """
    yellow_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red { background-color: #ca1d1d; }
                .amber { background-color: #ffd600; }
                .green { background-color: #04d930; }
                .gray { background-color: #ccc; }
            </style>
        </head>
        <body>
        <div class="traffic-light">
            <div class="light gray"></div>
            <div class="light amber"></div>
            <div class="light gray"></div>
        </div>
        </body>
        </html>
    """
    green_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red { background-color: #ca1d1d; }
                .amber { background-color: #ffd600; }
                .green { background-color: #04d930; }
                .gray { background-color: #ccc; }
            </style>
        </head>
        <body>
        <div class="traffic-light">
            <div class="light gray"></div>
            <div class="light gray"></div>
            <div class="light green"></div>
        </div>
        </body>
        </html>
    """
    grey_light_html = """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <title>Результаты тестов</title>
            <style>
                .traffic-light {
                    display: inline-flex;
                    gap: 5px;
                    padding: 10px;
                    align-items: center;
                    border: 2px solid #ccc;
                    border-radius: 100px;
                }
                .light {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                }
                .red { background-color: #ca1d1d; }
                .amber { background-color: #ffd600; }
                .green { background-color: #04d930; }
                .gray { background-color: #ccc; }
            </style>
        </head>
        <body>
        <div class="traffic-light">
            <div class="light gray"></div>
            <div class="light gray"></div>
            <div class="light gray"></div>
        </div>
        </body>
        </html>
    """

    if colour == 'green':
        if not return_html:
            display(HTML(green_light_html))
        else:
            return green_light_html
    elif colour == 'yellow' or colour == 'amber':
        if not return_html:
            display(HTML(yellow_light_html))
        else:
            return yellow_light_html
    elif colour == 'red':
        if not return_html:
            display(HTML(red_light_html))
        else:
            return red_light_html
    elif colour == 'grey' or colour == 'gray':
        if not return_html:
            display(HTML(grey_light_html))
        else:
            return grey_light_html


# =============================================================================
# ТАБЛИЦЫ С КРИТЕРИЯМИ
# =============================================================================

def show_criteria_semaphore(
    green_criterion,
    yellow_criterion,
    red_criterion,
    styles
):
    """
    Формирование таблицы с критериями выставления светофора.

    Args:
        green_criterion: Критерий на зеленый светофор
        yellow_criterion: Критерий на желтый светофор
        red_criterion: Критерий на красный светофор
        styles: Стили для таблицы pandas

    Returns:
        DataFrame с критериями и светофорами
    """
    criterion = pd.DataFrame(columns=['comment', 'colour'])
    criterion['colour'] = ['green', 'yellow', 'red']

    criterion.loc[criterion.colour == 'green', 'comment'] = green_criterion
    criterion.loc[criterion.colour == 'yellow', 'comment'] = yellow_criterion
    criterion.loc[criterion.colour == 'red', 'comment'] = red_criterion

    criterion.loc[criterion['colour'] == 'green', 'colour_img'] = display_semaphore('green', return_html=True)
    criterion.loc[criterion['colour'] == 'yellow', 'colour_img'] = display_semaphore('yellow', return_html=True)
    criterion.loc[criterion['colour'] == 'red', 'colour_img'] = display_semaphore('red', return_html=True)

    criterion.rename({'colour_img': 'Результат', 'comment': 'Критерий'}, axis=1, inplace=True)
    criterion.drop('colour', axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


def show_criteria_semaphore_without_red(
    green_criterion,
    yellow_criterion,
    styles
):
    """
    Формирование таблицы с критериями (без красного светофора).

    Args:
        green_criterion: Критерий на зеленый светофор
        yellow_criterion: Критерий на желтый светофор
        styles: Стили для таблицы pandas

    Returns:
        DataFrame с критериями и светофорами
    """
    criterion = pd.DataFrame(columns=['comment', 'colour'])
    criterion['colour'] = ['green', 'yellow']

    criterion.loc[criterion.colour == 'green', 'comment'] = green_criterion
    criterion.loc[criterion.colour == 'yellow', 'comment'] = yellow_criterion

    criterion.loc[criterion['colour'] == 'green', 'colour_img'] = display_semaphore('green', return_html=True)
    criterion.loc[criterion['colour'] == 'yellow', 'colour_img'] = display_semaphore('yellow', return_html=True)

    criterion.rename({'colour_img': 'Результат', 'comment': 'Критерий'}, axis=1, inplace=True)
    criterion.drop('colour', axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


def show_criteria_semaphore_without_yellow(
    green_criterion,
    red_criterion,
    styles
):
    """
    Формирование таблицы с критериями (без желтого светофора).

    Args:
        green_criterion: Критерий на зеленый светофор
        red_criterion: Критерий на красный светофор
        styles: Стили для таблицы pandas

    Returns:
        DataFrame с критериями и светофорами
    """
    criterion = pd.DataFrame(columns=['comment', 'colour'])
    criterion['colour'] = ['green', 'red']

    criterion.loc[criterion.colour == 'green', 'comment'] = green_criterion
    criterion.loc[criterion.colour == 'red', 'comment'] = red_criterion

    criterion.loc[criterion['colour'] == 'green', 'colour_img'] = display_semaphore('green', return_html=True)
    criterion.loc[criterion['colour'] == 'red', 'colour_img'] = display_semaphore('red', return_html=True)

    criterion.rename({'colour_img': 'Результат', 'comment': 'Критерий'}, axis=1, inplace=True)
    criterion.drop('colour', axis=1, inplace=True)

    try:
        return criterion.style.hide().set_table_styles(styles)
    except Exception:
        return criterion.style.hide_index().set_table_styles(styles)


# =============================================================================
# ОТОБРАЖЕНИЕ DATAFRAME СО СВЕТОФОРАМИ
# =============================================================================

def show_df(
    df,
    green_threshold={},
    red_threshold={},
    type_threshold='abs',
    precision=2
):
    """
    Отображение DataFrame с добавлением колонок светофоров.

    Алгоритм:
        1. Определяем метрики для выставления светофоров
        2. Определяем тип значений (абсолютные или относительные)
        3. Для каждой метрики ставим светофоры
        4. Добавляем столбец светофоров после столбца метрики

    Args:
        df: DataFrame с данными
        green_threshold: Условие для зеленого светофора
        red_threshold: Условие для красного светофора
        type_threshold: 'abs' или 'relative'
        precision: Точность отображения метрик

    Returns:
        DataFrame с добавленными светофорами
    """
    df = df.copy()
    metrics = [*(set(green_threshold.keys()) & set(red_threshold.keys()))]

    colours_by_metric = []
    if metrics:
        if type_threshold == 'relative':
            compare_base = 100 * (1 - df[metrics] / df[metrics].iloc[0])
        elif type_threshold == 'abs':
            compare_base = df[metrics]

        for metric in metrics:
            green, red = np.array(green_threshold[metric]), np.array(red_threshold[metric])
            assert (len(df) == green.size) | (green.size == 1), 'Некорректные по длине границы светофоров.'
            assert (len(df) == red.size) | (red.size == 1), 'Некорректные по длине границы светофоров.'

            mvals = compare_base[metric]
            if (green <= red).all():
                condlist = np.array([mvals >= red, (mvals > green) & (mvals < red), mvals <= green, pd.isna(mvals)])
                colours = np.select(condlist, ['red', 'yellow', 'green', 'grey'])
            elif (green > red).all():
                condlist = np.array([mvals >= green, (mvals > red) & (mvals < green), mvals <= red, pd.isna(mvals)])
                colours = np.select(condlist, ['green', 'yellow', 'red', 'grey'])
            else:
                raise ValueError('Некорректно выставлены границы светофоров.')

            colours_by_metric.append(colours)
            ncol = int(np.where(df.columns == metric)[0][0])
            if type_threshold == 'relative':
                df.insert(ncol + 1, metric + 'Прирост, п.п.', mvals)
                ncol = ncol + 1
            df.insert(ncol + 1, metric + '<br>semaphore', colours)

        colours_by_metric = np.array(colours_by_metric).T

    for column in df.columns:
        if 'semaphore' in str(column):
            df['Результат'] = np.array([display_semaphore(colour, return_html=True) for colour in df[column].values])
    df.drop(columns=[metrics[0] + '<br>semaphore'], inplace=True)

    return df, colours_by_metric