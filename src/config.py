"""
Конфигурационные параметры для пайплайна.
Централизованное хранение всех констант и настроек.
"""

from typing import Dict, List, Optional
import numpy as np


# Константы для сегментации клиентов
# Сегменты основаны на сумме транзакций за прошлый месяц
SEGMENT_BINS: List[float] = [0, 1e5, 1e6, 1e7, np.inf]
SEGMENT_LABELS: List[str] = ['S1', 'S2', 'S3', 'S4']

# Параметры фильтрации данных по умолчанию
DEFAULT_FILTERS = {
    'remove_inactive': True,        # удалять клиентов без активности
    'remove_lag6_zero': True,       # удалять записи с нулевым lag6
    'min_tx_total': 25,              # минимальное количество транзакций
}

# Параметры разделения данных
DEFAULT_SPLIT = {
    'train_size': 0.7,
    'val_size': 0.15,
    'test_size': 0.15,
}

# Параметры отбора признаков по умолчанию
DEFAULT_FEATURE_SELECTION_PARAMS = {
    'corr_threshold': 0.95,          # порог для удаления коррелирующих признаков
    'stopping_rounds': 20,            # шагов без улучшения для остановки
    'improvement_threshold': 0.0005,  # минимальное улучшение для добавления признака
}

# Параметры оптимизации гиперпараметров по умолчанию
DEFAULT_HYPEROPT_PARAMS = {
    'n_trials': 100,                  # количество попыток оптимизации
    'n_splits': 3,                    # количество фолдов для CV
    'aggregation_strategy': 'frequent',  # стратегия агрегации результатов
    'min_freq_ratio': 0.8,             # минимальная частота для стратегии 'frequent'
}

# Параметры модели LightGBM по умолчанию
DEFAULT_LGBM_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'verbose': -1,
    'learning_rate': 0.1,
}

# Порог для определения работающего сегмента (lift в процентах)
WORKING_SEGMENT_LIFT_THRESHOLD: float = 10.0

# Колонки, которые не должны использоваться как признаки
EXCLUDED_COLUMNS = [
    'client_id', 'year_month', 
    'amount_11_sum', 'log_amount', 
    'amount_11_mean', 'amount_11_median', 
    'amount_11_std', 'amount_11_min', 
    'amount_11_max', 'tx_11_count', 
    'tx_total', 'segment'
]


def get_feature_selection_params(custom_params: Optional[Dict] = None) -> Dict:
    """
    Возвращает параметры отбора признаков с возможностью переопределения.
    
    Parameters
    ----------
    custom_params : dict, optional
        Пользовательские параметры для переопределения
    
    Returns
    -------
    dict
        Объединенные параметры
    """
    params = DEFAULT_FEATURE_SELECTION_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    return params


def get_hyperopt_params(custom_params: Optional[Dict] = None) -> Dict:
    """
    Возвращает параметры оптимизации гиперпараметров с возможностью переопределения.
    
    Parameters
    ----------
    custom_params : dict, optional
        Пользовательские параметры для переопределения
    
    Returns
    -------
    dict
        Объединенные параметры
    """
    params = DEFAULT_HYPEROPT_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    return params