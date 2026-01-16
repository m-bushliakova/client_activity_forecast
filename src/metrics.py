"""
Модуль с метриками для оценки качества моделей.
Содержит функции для вычисления стандартных метрик регрессии,
метрик по сегментам и бизнес-метрики lift@20%.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from typing import Dict, List, Optional, Union, Any


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Вычисляет полный набор метрик регрессии для оценки качества предсказаний.
    
    Метрики:
    - RMSE, MSE, MAE, MedAE - абсолютные ошибки
    - R² - коэффициент детерминации
    - MAPE - средняя абсолютная процентная ошибка
    - MaxError, Q95Error - максимальная и 95-й перцентиль ошибки
    - Относительные метрики (в процентах от среднего)
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения целевой переменной
    y_pred : array-like
        Предсказанные значения модели
    logger : logging.Logger, optional
        Логгер для вывода отладочной информации
    
    Returns
    -------
    dict
        Словарь с метриками
    """
    metrics = {}
    
    # Преобразуем в numpy массивы для единообразия
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Проверка на пустые данные
    if len(y_true) == 0:
        if logger:
            logger.warning("Пустой массив для расчета метрик")
        return {k: np.nan for k in ['rmse', 'mse', 'mae', 'r2', 'medae', 'mape']}
    
    # Основные метрики
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['medae'] = median_absolute_error(y_true, y_pred)
    
    # MAPE (исключаем нулевые значения)
    mask = y_true != 0
    if mask.sum() > 0:
        metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        metrics['mape'] = np.nan
        if logger:
            logger.warning("Все значения y_true равны нулю, MAPE не может быть вычислен")
    
    # Квантильные метрики
    abs_errors = np.abs(y_true - y_pred)
    metrics['max_error'] = np.max(abs_errors)
    metrics['q95_error'] = np.percentile(abs_errors, 95)
    
    # Относительные метрики (в процентах от среднего)
    mean_true = y_true.mean()
    if mean_true != 0 and not np.isnan(mean_true):
        metrics['rmse_percent'] = metrics['rmse'] / mean_true * 100
        metrics['mae_percent'] = metrics['mae'] / mean_true * 100
    else:
        metrics['rmse_percent'] = np.nan
        metrics['mae_percent'] = np.nan
        if logger:
            logger.warning("Среднее значение y_true равно нулю, относительные метрики не могут быть вычислены")
    
    # Статистика
    metrics['count'] = len(y_true)
    metrics['true_mean'] = mean_true
    metrics['true_std'] = y_true.std()
    metrics['true_min'] = y_true.min()
    metrics['true_max'] = y_true.max()
    
    return metrics


def compute_segment_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    feature: Union[np.ndarray, pd.Series],
    bins: List[float],
    segment_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Вычисляет метрики качества модели для сегментов, сформированных на основе значения признака.
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения целевой переменной
    y_pred : array-like
        Предсказанные значения модели
    feature : array-like
        Значения признака, по которым строится сегментация
    bins : list
        Границы интервалов для сегментации
    segment_names : list, optional
        Названия сегментов. Если None, генерируются как S1, S2, ...
    logger : logging.Logger, optional
        Логгер для вывода отладочной информации
    
    Returns
    -------
    pd.DataFrame
        Таблица с метриками для каждого сегмента
    """
    if segment_names is None:
        segment_names = [f'S{i+1}' for i in range(len(bins)-1)]
    
    # Присваиваем сегменты
    segment_assign = pd.cut(feature, bins=bins, labels=segment_names, include_lowest=True)
    
    records = []
    for segment in segment_names:
        mask = segment_assign == segment
        if mask.sum() == 0:
            if logger:
                logger.debug(f"Сегмент {segment} пуст, пропускаем")
            continue
        
        segment_true = y_true[mask]
        segment_pred = y_pred[mask]
        metrics = compute_metrics(segment_true, segment_pred, logger)
        metrics['segment'] = segment
        metrics['count'] = mask.sum()
        metrics['min_feature'] = feature[mask].min()
        metrics['max_feature'] = feature[mask].max()
        records.append(metrics)
    
    if not records:
        if logger:
            logger.warning("Нет данных ни для одного сегмента")
        return pd.DataFrame()
    
    return pd.DataFrame(records).set_index('segment')


def evaluate_model_with_segments(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    feature: Union[np.ndarray, pd.Series],
    bins: List[float],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Рассчитывает метрики глобально и по сегментам, объединяя результаты в одну таблицу.
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения целевой переменной
    y_pred : array-like
        Предсказанные значения модели
    feature : array-like
        Значения признака для сегментации
    bins : list
        Границы интервалов для сегментации
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    pd.DataFrame
        Объединенная таблица с метриками (первая строка - глобальные)
    """
    # Глобальные метрики
    global_metrics = compute_metrics(y_true, y_pred, logger)
    global_metrics_df = pd.DataFrame(global_metrics, index=['global']).T
    
    # Метрики по сегментам
    segment_metrics_df = compute_segment_metrics(y_true, y_pred, feature, bins, logger=logger)
    
    # Объединяем
    if not segment_metrics_df.empty:
        combined_df = pd.concat([global_metrics_df.T, segment_metrics_df])
    else:
        combined_df = global_metrics_df.T
    
    # Округляем для читаемости
    combined_df = combined_df.round(2)
    combined_df.index.name = 'segment'
    
    if logger:
        logger.info("Метрики рассчитаны")
        logger.info(f"Глобальный R2: {global_metrics['r2']:.3f}")
        logger.info(f"Глобальный RMSE: {global_metrics['rmse']:.2f}")
        logger.info(f"Сегментов с данными: {len(segment_metrics_df)}")
    
    return combined_df


def calculate_lift(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    baseline: Union[np.ndarray, pd.Series],
    top_percent: int = 20,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Рассчитывает бизнес-метрику lift@k - насколько лучше модель выбирает топ-k% клиентов
    по сравнению с baseline (обычно значение за прошлый месяц).
    
    Интерпретация:
    - lift > 0: модель лучше baseline
    - lift = 10%: модель дает на 10% больше выручки при выборе топ-k% клиентов
    
    Parameters
    ----------
    y_true : array-like
        Истинные значения целевой переменной (фактическая выручка)
    y_pred : array-like
        Предсказанные значения модели
    baseline : array-like
        Базовые значения для сравнения (например, lag1_amount_11_sum)
    top_percent : int, default=20
        Процент лучших клиентов для отбора
    logger : logging.Logger, optional
        Логгер для вывода отладочной информации
    
    Returns
    -------
    dict
        Словарь с результатами:
        - lift : float - улучшение модели в процентах
        - model_sum : float - сумма реальных значений для топ-k% по модели
        - baseline_sum : float - сумма реальных значений для топ-k% по baseline
        - ideal_sum : float - сумма реальных значений для идеального выбора
        - count : int - количество отобранных клиентов
    """
    # Конвертируем в numpy массивы
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    baseline = np.array(baseline)
    
    n_total = len(y_true)
    n_top = max(1, int(n_total * top_percent / 100))
    
    if n_top > n_total:
        n_top = n_total
        if logger:
            logger.warning(f"top_percent={top_percent} дает n_top={n_top} > n_total={n_total}, используем все данные")
    
    # По модели
    model_idx = np.argsort(y_pred)[-n_top:]
    model_sum = y_true[model_idx].sum()
    
    # По baseline
    baseline_idx = np.argsort(baseline)[-n_top:]
    baseline_sum = y_true[baseline_idx].sum()
    
    # Идеальный выбор (если бы знали будущее)
    ideal_idx = np.argsort(y_true)[-n_top:]
    ideal_sum = y_true[ideal_idx].sum()
    
    # Расчет lift
    if baseline_sum > 0:
        lift = (model_sum - baseline_sum) / baseline_sum * 100
    else:
        lift = 0.0
        if logger:
            logger.warning("baseline_sum = 0, lift не может быть рассчитан")
    
    return {
        'lift': lift,
        'model_sum': model_sum,
        'baseline_sum': baseline_sum,
        'ideal_sum': ideal_sum,
        'count': n_top,
        'n_total': n_total,
        'top_percent': top_percent
    }


def calculate_segment_metrics_from_steps(
    segment_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Рассчитывает бизнес-метрики для одного сегмента на основе данных по шагам.
    
    Parameters
    ----------
    segment_data : list of dict
        Список словарей с данными по каждому шагу для сегмента.
        Каждый словарь должен содержать:
        - step: номер шага
        - test_month: месяц теста
        - lift: lift за месяц
        - model_sum: сумма по модели за месяц
        - baseline_sum: сумма по baseline за месяц
        - ideal_sum: идеальная сумма за месяц
    
    Returns
    -------
    dict
        Агрегированные метрики для сегмента
    """
    if not segment_data:
        return {}
    
    # Преобразуем в DataFrame для удобства
    df = pd.DataFrame(segment_data)
    
    # Считаем итоги
    total_model = df['model_sum'].sum()
    total_baseline = df['baseline_sum'].sum()
    total_ideal = df['ideal_sum'].sum()
    avg_lift = df['lift'].mean()
    
    # Находим лучший и худший месяц
    best_idx = df['lift'].idxmax()
    worst_idx = df['lift'].idxmin()
    
    best_month = df.loc[best_idx]
    worst_month = df.loc[worst_idx]
    
    # Считаем эффективность относительно идеала
    potential_gain = total_ideal - total_baseline
    actual_gain = total_model - total_baseline
    efficiency = (actual_gain / potential_gain * 100) if potential_gain > 0 else 0
    
    # Стабильность (стандартное отклонение lift'ов)
    lift_std = df['lift'].std() if len(df) > 1 else 0
    
    return {
        'avg_lift': avg_lift,
        'lift_std': lift_std,
        'total_model': total_model,
        'total_baseline': total_baseline,
        'total_ideal': total_ideal,
        'gain_vs_baseline': actual_gain,
        'efficiency_vs_ideal': efficiency,
        'best_month': {
            'month': best_month['test_month'],
            'lift': best_month['lift'],
            'gain': best_month['model_sum'] - best_month['baseline_sum']
        },
        'worst_month': {
            'month': worst_month['test_month'],
            'lift': worst_month['lift'],
            'loss': worst_month['baseline_sum'] - worst_month['model_sum']
        },
        'months_count': len(df),
        'positive_months': (df['lift'] > 0).sum(),
        'negative_months': (df['lift'] < 0).sum()
    }


def extract_segment_data(
    wf_results: pd.DataFrame,
    segment: str
) -> List[Dict[str, Any]]:
    """
    Извлекает данные для конкретного сегмента из результатов walk-forward.
    
    Parameters
    ----------
    wf_results : pd.DataFrame
        Результаты walk-forward валидации
    segment : str
        Название сегмента (например, 'S1')
    
    Returns
    -------
    list of dict
        Данные по шагам для сегмента
    """
    lift_col = f"{segment}_lift"
    model_col = f"{segment}_model"
    baseline_col = f"{segment}_baseline"
    ideal_col = f"{segment}_ideal"
    
    # Проверяем наличие колонок
    if lift_col not in wf_results.columns:
        return []
    
    segment_data = []
    for idx, row in wf_results.iterrows():
        segment_data.append({
            'step': idx + 1,
            'test_month': row['test_month'],
            'lift': row[lift_col],
            'model_sum': row[model_col],
            'baseline_sum': row[baseline_col],
            'ideal_sum': row[ideal_col]
        })
    
    return segment_data


def calculate_all_segments_metrics(
    wf_results: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Рассчитывает бизнес-метрики для всех сегментов.
    
    Parameters
    ----------
    wf_results : pd.DataFrame
        Результаты walk-forward валидации
    
    Returns
    -------
    dict
        Метрики по каждому сегменту
    """
    # Собираем все сегменты из колонок
    segments = set()
    for col in wf_results.columns:
        if col.endswith('_lift'):
            # Предполагаем формат "S1_lift"
            segment = col.split('_')[0]
            if segment.startswith('S'):
                segments.add(segment)
    
    segment_metrics = {}
    
    for segment in sorted(segments):
        segment_data = extract_segment_data(wf_results, segment)
        if segment_data:
            segment_metrics[segment] = calculate_segment_metrics_from_steps(segment_data)
    
    return segment_metrics


def calculate_cv_metrics(
    cv_results: List[Dict[str, float]],
    metric_name: str = 'r2'
) -> Dict[str, float]:
    """
    Рассчитывает агрегированные метрики по результатам кросс-валидации.
    
    Parameters
    ----------
    cv_results : list of dict
        Список результатов по фолдам
    metric_name : str, default='r2'
        Название метрики для агрегации
    
    Returns
    -------
    dict
        Агрегированные метрики (mean, std, min, max)
    """
    values = [res.get(metric_name, np.nan) for res in cv_results]
    values = [v for v in values if not np.isnan(v)]
    
    if not values:
        return {
            f'{metric_name}_mean': np.nan,
            f'{metric_name}_std': np.nan,
            f'{metric_name}_min': np.nan,
            f'{metric_name}_max': np.nan,
            'n_folds': 0
        }
    
    return {
        f'{metric_name}_mean': np.mean(values),
        f'{metric_name}_std': np.std(values),
        f'{metric_name}_min': np.min(values),
        f'{metric_name}_max': np.max(values),
        'n_folds': len(values)
    }