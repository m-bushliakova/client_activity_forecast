"""
Утилиты для создания фолдов временной кросс-валидации.
Временная кросс-валидация учитывает хронологический порядок данных,
что критично для временных рядов.
"""

from typing import List, Dict, Any
import logging


def prepare_cv_folds(train_months: List[str], n_splits: int = 3) -> List[Dict[str, Any]]:
    """
    Создает фолды для временной кросс-валидации с расширяющимся окном.
    
    Особенности:
    - Каждый следующий фолд использует больше данных для обучения
    - Валидация всегда на одном следующем месяце
    - Сохраняется временной порядок (нет look-ahead bias)
    
    Parameters
    ----------
    train_months : List[str]
        Отсортированный список месяцев в обучающей выборке
    n_splits : int, default=3
        Количество фолдов для валидации
    
    Returns
    -------
    List[Dict[str, Any]]
        Список фолдов, каждый словарь содержит:
        - fold: номер фолда (1-based)
        - train_months: месяцы для обучения
        - val_months: месяцы для валидации
        - n_train: количество месяцев в обучении
        - n_val: количество месяцев в валидации
    """
    if not train_months:
        raise ValueError("Список месяцев не может быть пустым")
    
    if n_splits > len(train_months):
        raise ValueError(f"n_splits ({n_splits}) не может быть больше количества месяцев ({len(train_months)})")
    
    folds = []
    
    for fold_idx in range(n_splits):
        # Индекс валидационного месяца (с конца)
        val_idx = -(fold_idx + 1)
        
        train_fold_months = train_months[:val_idx]  # все до val_idx
        val_fold_months = [train_months[val_idx]]   # только val_idx месяц
        
        folds.append({
            'fold': fold_idx + 1,
            'train_months': train_fold_months,
            'val_months': val_fold_months,
            'n_train': len(train_fold_months),
            'n_val': len(val_fold_months)
        })
    
    return folds


def validate_folds_structure(folds: List[Dict[str, Any]], logger: logging.Logger = None) -> bool:
    """
    Проверяет корректность структуры фолдов.
    
    Parameters
    ----------
    folds : List[Dict[str, Any]]
        Список фолдов от prepare_cv_folds
    logger : logging.Logger, optional
        Логгер для вывода ошибок
    
    Returns
    -------
    bool
        True если структура корректна
    """
    required_keys = {'fold', 'train_months', 'val_months', 'n_train', 'n_val'}
    
    for fold in folds:
        if not required_keys.issubset(fold.keys()):
            if logger:
                logger.error(f"Фолд {fold.get('fold', 'unknown')} не содержит все необходимые ключи")
            return False
        
        if fold['n_train'] != len(fold['train_months']):
            if logger:
                logger.error(f"Фолд {fold['fold']}: n_train не совпадает с длиной train_months")
            return False
        
        if fold['n_val'] != len(fold['val_months']):
            if logger:
                logger.error(f"Фолд {fold['fold']}: n_val не совпадает с длиной val_months")
            return False
    
    return True