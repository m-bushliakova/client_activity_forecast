"""
Модуль для загрузки, очистки и подготовки данных.
Реализует фильтрацию, сегментацию и разделение данных.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.config import SEGMENT_BINS, SEGMENT_LABELS, DEFAULT_FILTERS, DEFAULT_SPLIT


def load_data(
    version: int = 3,
    remove_inactive: bool = DEFAULT_FILTERS['remove_inactive'],
    remove_lag6_zero: bool = DEFAULT_FILTERS['remove_lag6_zero'],
    min_tx_total: int = DEFAULT_FILTERS['min_tx_total'],
    logger: Optional[logging.Logger] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Загружает данные из parquet файла и применяет базовые фильтры.
    
    Этапы фильтрации:
    1. Удаление клиент-месяцев без активности (lag1-lag3 = 0)
    2. Удаление записей с нулевым lag6_tx_total
    3. Удаление клиентов с малым количеством транзакций
    
    Parameters
    ----------
    version : int, default=3
        Версия датасета (влияет на путь к файлу)
    remove_inactive : bool, default=True
        Удалять ли клиент-месяцы с lag1-lag3 = 0
    remove_lag6_zero : bool, default=True
        Удалять ли клиент-месяцы с lag6_tx_total = 0
    min_tx_total : int, default=25
        Минимальное значение tx_total (клиенты с меньшим количеством удаляются)
    logger : logging.Logger, optional
        Логгер для вывода информации
    verbose : bool, default=True
        Если True и logger передан, выводит информацию о ходе выполнения
    
    Returns
    -------
    pd.DataFrame
        Очищенный и отсортированный датафрейм
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Формируем путь к файлу
    file_path = f"../data/processed/client_month_features_{version}.parquet"
    
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        raise
    
    initial_rows = len(df)
    if verbose:
        logger.info(f"Загружено {initial_rows:,} строк")
    
    # Словарь для хранения статистики удалений
    deletion_stats = {}
    
    # 1) Удаляем клиент-месяцы без активности
    if remove_inactive:
        inactive_mask = (
            (df['lag1_amount_11_sum'] == 0) & 
            (df['lag2_amount_11_sum'] == 0) & 
            (df['lag3_amount_11_sum'] == 0)
        )
        df = df[~inactive_mask].copy()
        deletion_stats['inactive'] = inactive_mask.sum()
        
        if verbose:
            logger.info(f"Удалено {inactive_mask.sum():,} клиент-месяцев с lag1-lag3 = 0")
    
    # 2) Удаляем клиент-месяцы с нулевым lag6
    if remove_lag6_zero:
        lag6_mask = (df['lag6_tx_total'] == 0)
        df = df[~lag6_mask].copy()
        deletion_stats['lag6_zero'] = lag6_mask.sum()
        
        if verbose:
            logger.info(f"Удалено {lag6_mask.sum():,} клиент-месяцев с lag6_tx_total = 0")
    
    # 3) Удаляем клиент-месяцы с малым количеством транзакций
    if min_tx_total > 0:
        low_tx_mask = (df['tx_total'] < min_tx_total)
        df = df[~low_tx_mask].copy()
        deletion_stats['low_tx'] = low_tx_mask.sum()
        
        if verbose:
            logger.info(f"Удалено {low_tx_mask.sum():,} клиент-месяцев с tx_total < {min_tx_total}")
    
    if verbose:
        remaining_pct = (len(df) / initial_rows * 100)
        logger.info(f"Осталось {len(df):,} строк ({remaining_pct:.1f}% от исходных)")
        logger.debug(f"Статистика удалений: {deletion_stats}")
    
    # Сортировка для сохранения временного порядка
    df = df.sort_values(['client_id', 'year_month']).reset_index(drop=True)
    
    return df


def split_data(
    df: pd.DataFrame,
    train_size: float = DEFAULT_SPLIT['train_size'],
    val_size: float = DEFAULT_SPLIT['val_size'],
    test_size: float = DEFAULT_SPLIT['test_size'],
    logger: Optional[logging.Logger] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Разделяет данные на train/val/test по месяцам с сохранением временного порядка.
    
    Важно: разделение происходит строго по временной оси, без перемешивания.
    Это критично для корректной оценки моделей на временных рядах.
    
    Parameters
    ----------
    df : pd.DataFrame
        Входной датафрейм (должен содержать колонку 'year_month')
    train_size, val_size, test_size : float, default=(0.7, 0.15, 0.15)
        Доли для разделения (должны суммироваться в 1)
    logger : logging.Logger, optional
        Логгер для вывода информации
    verbose : bool, default=True
        Если True, выводит информацию о разделении
    
    Returns
    -------
    train : pd.DataFrame
        Обучающая выборка
    val : pd.DataFrame
        Валидационная выборка
    test : pd.DataFrame
        Тестовая выборка
    months : List[str]
        Список всех месяцев в хронологическом порядке
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Проверка корректности долей
    total_size = train_size + val_size + test_size
    if abs(total_size - 1.0) > 1e-6:
        raise ValueError(f"Сумма долей должна быть 1, получено {total_size}")
    
    months = sorted(df['year_month'].unique())
    n_months = len(months)
    
    # Вычисляем границы разделения
    train_end = int(n_months * train_size)
    val_end = int(n_months * (train_size + val_size))
    
    train_months = months[:train_end]
    val_months = months[train_end:val_end]
    test_months = months[val_end:]
    
    # Фильтруем данные по месяцам
    train = df[df['year_month'].isin(train_months)].copy()
    val = df[df['year_month'].isin(val_months)].copy()
    test = df[df['year_month'].isin(test_months)].copy()
    
    if verbose:
        logger.info(f"\nTrain: {len(train_months)} месяцев, {len(train):,} строк, "
                   f"{train['client_id'].nunique():,} уникальных клиентов")
        logger.info(f"Val:   {len(val_months)} месяцев, {len(val):,} строк, "
                   f"{val['client_id'].nunique():,} уникальных клиентов")
        logger.info(f"Test:  {len(test_months)} месяцев, {len(test):,} строк, "
                   f"{test['client_id'].nunique():,} уникальных клиентов")
    
    return train, val, test, months


def assign_segment(
    df: pd.DataFrame,
    feature: str = 'lag1_amount_11_sum',
    bins: List[float] = SEGMENT_BINS,
    labels: List[str] = SEGMENT_LABELS
) -> pd.Series:
    """
    Присваивает сегмент каждому наблюдению на основе значения признака.
    
    Сегментация используется для:
    - Построения отдельных моделей для разных групп клиентов
    - Анализа производительности модели в разных группах
    - Бизнес-метрик по сегментам
    
    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    feature : str, default='lag1_amount_11_sum'
        Название признака для сегментации (обычно сумма за прошлый месяц)
    bins : List[float], default=[0, 1e5, 1e6, 1e7, np.inf]
        Границы сегментов
    labels : List[str], default=['S1', 'S2', 'S3', 'S4']
        Метки сегментов (должны соответствовать количеству интервалов)
    
    Returns
    -------
    pd.Series
        Метки сегментов для каждой строки
    
    Raises
    ------
    ValueError
        Если количество интервалов не соответствует количеству меток
    """
    if len(bins) - 1 != len(labels):
        raise ValueError(
            f"Количество интервалов ({len(bins)-1}) должно соответствовать "
            f"количеству меток ({len(labels)})"
        )
    
    return pd.cut(
        df[feature], 
        bins=bins, 
        labels=labels, 
        include_lowest=True,
        right=False  # интервалы [a, b)
    )


def prepare_data_for_pipeline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    bins: List[float] = SEGMENT_BINS,
    labels: List[str] = SEGMENT_LABELS,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Подготавливает данные для пайплайна:
    - Добавляет сегменты на основе lag1_amount_11_sum
    - Создает логарифмический target для стабилизации дисперсии
    
    Parameters
    ----------
    train, val, test : pd.DataFrame
        Разделенные датафреймы
    bins : List[float], default=SEGMENT_BINS
        Границы сегментов
    labels : List[str], default=SEGMENT_LABELS
        Метки сегментов
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Датафреймы с добавленными колонками 'segment' и 'log_amount'
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Добавляем сегменты во все выборки
    for df_name, df in [("Train", train), ("Val", val), ("Test", test)]:
        df['segment'] = assign_segment(df, bins=bins, labels=labels)
        
        # Логарифмический target для стабилизации дисперсии
        # log1p используется для обработки нулей
        df['log_amount'] = np.log1p(df['amount_11_sum'])
    
    # Логируем распределение по сегментам
    logger.info("\nРаспределение по сегментам:")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        seg_counts = df['segment'].value_counts().sort_index().to_dict()
        logger.info(f"{name}: {seg_counts}")
    
    return train, val, test


def load_and_split_data(
    version: int = 3,
    bins: List[float] = SEGMENT_BINS,
    filter_params: Optional[Dict[str, Any]] = None,
    split_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн загрузки и подготовки данных.
    Объединяет загрузку, фильтрацию, разделение и подготовку.
    
    Parameters
    ----------
    version : int, default=3
        Версия датасета
    bins : List[float], default=SEGMENT_BINS
        Границы сегментов
    filter_params : dict, optional
        Параметры фильтрации (переопределяют DEFAULT_FILTERS)
    split_params : dict, optional
        Параметры разделения (переопределяют DEFAULT_SPLIT)
    verbose : bool, default=True
        Если True, выводит информацию о ходе выполнения
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train, val, test датафреймы с колонками segment и log_amount
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Объединяем параметры с дефолтными
    filter_params = filter_params or {}
    split_params = split_params or {}
    
    filter_config = {**DEFAULT_FILTERS, **filter_params}
    split_config = {**DEFAULT_SPLIT, **split_params}
    
    logger.info("\n" + "="*60)
    logger.info("ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    logger.info("="*60)
    
    # Загрузка с фильтрацией
    df = load_data(
        version=version,
        logger=logger,
        verbose=verbose,
        **filter_config
    )
    
    # Разделение на train/val/test
    train, val, test, months = split_data(
        df,
        logger=logger,
        verbose=verbose,
        **split_config
    )
    
    # Подготовка для пайплайна (сегменты, лог target)
    train, val, test = prepare_data_for_pipeline(
        train, val, test,
        bins=bins,
        logger=logger
    )
    
    logger.info(f"\nTrain: {len(train):,} строк, {train['client_id'].nunique():,} клиентов")
    logger.info(f"Val:   {len(val):,} строк, {val['client_id'].nunique():,} клиентов")
    logger.info(f"Test:  {len(test):,} строк, {test['client_id'].nunique():,} клиентов")
    
    return train, val, test