"""
Модуль для обучения моделей и walk-forward валидации.
Содержит функции для обучения моделей по сегментам и оценки их качества.
"""
import sys
import numpy as np
import pandas as pd
import logging
from lightgbm import LGBMRegressor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.append(str(Path(__file__).parent.parent))

from src.metrics import calculate_lift, calculate_all_segments_metrics
from src.logging_utils import log_business_metrics, log_step_result


def prepare_walk_forward_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_months: List[str],
    step: int
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Подготавливает данные для step-шага walk-forward валидации.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Исходная обучающая выборка
    val_df : pd.DataFrame
        Исходная валидационная выборка
    val_months : List[str]
        Список месяцев валидации в хронологическом порядке
    step : int
        Номер текущего шага (1-based)
    
    Returns
    -------
    current_train : pd.DataFrame
        Данные для обучения (train + предыдущие месяцы val)
    current_test : pd.DataFrame
        Данные для тестирования (текущий месяц val)
    train_months : List[str]
        Список месяцев для обучения
    test_month : str
        Текущий тестовый месяц
    """
    test_month = val_months[step - 1]
    prev_months = val_months[:step - 1]
    
    # Обучающие данные: весь исходный train + предыдущие месяцы val
    current_train = pd.concat([
        train_df,
        val_df[val_df['year_month'].isin(prev_months)]
    ])
    
    # Тестовые данные: текущий месяц val
    current_test = val_df[val_df['year_month'] == test_month].copy()
    
    train_months = list(train_df['year_month'].unique()) + prev_months
    
    return current_train, current_test, train_months, test_month


def train_segment_models_for_month(
    current_train: pd.DataFrame,
    current_test: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    best_params_by_segment: Dict[str, Dict[str, Any]]
) -> np.ndarray:
    """
    Обучает модели для всех сегментов на current_train
    и возвращает предсказания для current_test.
    
    Parameters
    ----------
    current_train : pd.DataFrame
        Данные для обучения (с колонкой 'segment')
    current_test : pd.DataFrame
        Данные для тестирования (с колонкой 'segment')
    features_by_segment : dict
        Признаки по сегментам
    best_params_by_segment : dict
        Лучшие гиперпараметры по сегментам
    
    Returns
    -------
    np.ndarray
        Предсказания для всех строк current_test
    """
    predictions = np.zeros(len(current_test))
    
    for segment, features in features_by_segment.items():
        # Фильтруем данные по сегменту
        train_mask = current_train['segment'] == segment
        test_mask = current_test['segment'] == segment
        
        # Пропускаем сегменты с малым количеством данных
        if train_mask.sum() < 100 or test_mask.sum() < 10:
            continue
        
        # Получаем параметры для сегмента
        segment_params = best_params_by_segment.get(segment, {})
        params = segment_params.get('best_params', {})
        use_log = segment_params.get('use_log', False)
        
        # Подготавливаем данные
        X_train = current_train.loc[train_mask, features].fillna(0)
        y_train = current_train.loc[train_mask, 'amount_11_sum']
        X_test = current_test.loc[test_mask, features].fillna(0)
        
        # Логарифмируем target если нужно
        if use_log:
            y_train = np.log1p(y_train)
        
        # Обучаем модель
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Предсказываем
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log) if use_log else pred_log
        
        predictions[test_mask] = pred
    
    return predictions


def calculate_month_metrics(
    current_test: pd.DataFrame,
    predictions: np.ndarray,
    features_by_segment: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Считает метрики для одного месяца по каждому сегменту отдельно.
    
    Parameters
    ----------
    current_test : pd.DataFrame
        Тестовые данные для текущего месяца
    predictions : np.ndarray
        Предсказания модели
    features_by_segment : dict
        Признаки по сегментам (нужен только для списка сегментов)
    
    Returns
    -------
    dict
        Метрики для каждого сегмента и общие метрики месяца
    """
    segment_results = {}
    total_model = 0
    total_baseline = 0
    total_ideal = 0
    
    for segment in features_by_segment.keys():
        mask = current_test['segment'] == segment
        if mask.sum() < 10:
            continue
        
        y_true = current_test.loc[mask, 'amount_11_sum'].values
        y_pred = predictions[mask]
        baseline = current_test.loc[mask, 'lag1_amount_11_sum'].values
        
        result = calculate_lift(y_true, y_pred, baseline)
        
        segment_results[f"{segment}_lift"] = result['lift']
        segment_results[f"{segment}_model"] = result['model_sum']
        segment_results[f"{segment}_baseline"] = result['baseline_sum']
        segment_results[f"{segment}_ideal"] = result['ideal_sum']
        segment_results[f"{segment}_count"] = mask.sum()
        
        total_model += result['model_sum']
        total_baseline += result['baseline_sum']
        total_ideal += result['ideal_sum']
    
    # Добавляем общие метрики месяца
    month_lift = (total_model - total_baseline) / total_baseline * 100 if total_baseline > 0 else 0
    
    segment_results['model_sum'] = total_model
    segment_results['baseline_sum'] = total_baseline
    segment_results['ideal_sum'] = total_ideal
    segment_results['month_lift'] = month_lift
    
    return segment_results


def log_walk_forward_summary(
    logger: logging.Logger,
    val_months: List[str],
    cumulative: Dict[str, float],
    results_df: pd.DataFrame
) -> None:
    """
    Логирует итоговые результаты walk-forward валидации.
    
    Parameters
    ----------
    logger : logging.Logger
        Логгер для вывода
    val_months : List[str]
        Список месяцев валидации
    cumulative : dict
        Накопленные результаты
    results_df : pd.DataFrame
        Таблица с результатами по шагам
    """
    logger.info("\n" + "="*70)
    logger.info("ИТОГ WALK-FORWARD ВАЛИДАЦИИ")
    logger.info("="*70)
    
    logger.info(f"Суммарно за {len(val_months)} месяцев:")
    logger.info(f"  Модель:   {cumulative['model']:>15,.0f} ₽")
    logger.info(f"  Baseline: {cumulative['baseline']:>15,.0f} ₽")
    logger.info(f"  Идеал:    {cumulative['ideal']:>15,.0f} ₽")
    logger.info(f"  Общий lift: {cumulative['lift']:+.1f}%")
    logger.info(f"  Достигнуто от идеала: {cumulative['ideal_pct']:.1f}%")
    
    # Статистика по месяцам
    if len(results_df) > 1:
        logger.info(f"\nСтабильность:")
        logger.info(f"  Средний lift по месяцам: {results_df['month_lift'].mean():+.1f}%")
        logger.info(f"  Std lift по месяцам: {results_df['month_lift'].std():.1f}%")
        logger.info(f"  Лучший месяц: {results_df.loc[results_df['month_lift'].idxmax(), 'test_month']} "
                   f"({results_df['month_lift'].max():+.1f}%)")
        logger.info(f"  Худший месяц: {results_df.loc[results_df['month_lift'].idxmin(), 'test_month']} "
                   f"({results_df['month_lift'].min():+.1f}%)")


def walk_forward_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    best_params_by_segment: Dict[str, Dict[str, Any]],
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Walk-forward валидация: постепенное добавление месяцев в обучение.
    
    Процесс:
    1. Обучаем на train, проверяем на 1-м месяце val
    2. Обучаем на train + 1-й месяц val, проверяем на 2-м месяце val
    3. И так далее до конца val
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка
    val_df : pd.DataFrame
        Валидационная выборка (несколько месяцев)
    features_by_segment : dict
        Признаки по сегментам
    best_params_by_segment : dict
        Лучшие гиперпараметры по сегментам
    logger : logging.Logger, optional
        Логгер
    
    Returns
    -------
    pd.DataFrame
        Таблица с результатами по каждому шагу
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    val_months = sorted(val_df['year_month'].unique())
    results = []
    
    cumulative = {'model': 0, 'baseline': 0, 'ideal': 0}
    
    logger.info("\n" + "="*70)
    logger.info("WALK-FORWARD ВАЛИДАЦИЯ")
    logger.info("="*70)
    
    for step in range(1, len(val_months) + 1):
        # 1. Подготовка данных
        current_train, current_test, train_months, test_month = prepare_walk_forward_data(
            train_df, val_df, val_months, step
        )
        
        # 2. Обучение и предсказание
        predictions = train_segment_models_for_month(
            current_train, current_test,
            features_by_segment, best_params_by_segment
        )
        
        # 3. Расчет метрик
        month_metrics = calculate_month_metrics(current_test, predictions, features_by_segment)
        
        # 4. Накопление
        cumulative['model'] += month_metrics['model_sum']
        cumulative['baseline'] += month_metrics['baseline_sum']
        cumulative['ideal'] += month_metrics['ideal_sum']
        
        # 5. Сохранение результатов
        results.append({
            'step': step,
            'test_month': test_month,
            'train_months': len(train_months),
            **month_metrics,
            'cumulative_model': cumulative['model'],
            'cumulative_baseline': cumulative['baseline'],
            'cumulative_ideal': cumulative['ideal'],
            'cumulative_lift': (cumulative['model'] - cumulative['baseline']) / cumulative['baseline'] * 100 
                               if cumulative['baseline'] > 0 else 0
        })
        
        # 6. Логирование
        log_step_result(logger, step, test_month, month_metrics)
    
    # 7. Итоговое логирование
    results_df = pd.DataFrame(results)
    
    cumulative['lift'] = (cumulative['model'] - cumulative['baseline']) / cumulative['baseline'] * 100
    cumulative['ideal_pct'] = cumulative['model'] / cumulative['ideal'] * 100 if cumulative['ideal'] > 0 else 0
    
    log_walk_forward_summary(logger, val_months, cumulative, results_df)
    
    return results_df


def train_and_evaluate_on_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    best_params_by_segment: Dict[str, Dict[str, Any]],
    target_col: str = 'amount_11_sum',
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, LGBMRegressor], np.ndarray, Dict[str, Any], float]:
    """
    Обучает модели на train и оценивает на val (однократно, без walk-forward).
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка (с колонкой 'segment')
    val_df : pd.DataFrame
        Валидационная выборка (с колонкой 'segment')
    features_by_segment : dict
        Словарь с признаками для каждого сегмента
    best_params_by_segment : dict
        Словарь с лучшими гиперпараметрами для каждого сегмента
    target_col : str
        Название целевой переменной
    logger : logging.Logger, optional
        Логгер
    
    Returns
    -------
    models : dict
        Обученные модели по сегментам
    predictions : np.array
        Предсказания для val
    segment_metrics : dict
        Метрики по сегментам
    global_lift : float
        Глобальный lift@20%
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    models = {}
    predictions = np.zeros(len(val_df))
    segment_metrics = {}
    
    logger.info("\n" + "="*60)
    logger.info("ОБУЧЕНИЕ МОДЕЛЕЙ НА TRAIN, ПРОВЕРКА НА VAL")
    logger.info("="*60)
    
    # Обучаем по каждому сегменту
    for segment in features_by_segment.keys():
        train_mask = train_df['segment'] == segment
        val_mask = val_df['segment'] == segment
        
        if train_mask.sum() < 100:
            logger.warning(f"  {segment}: мало данных в train ({train_mask.sum()}), пропускаем")
            continue
        
        if val_mask.sum() < 10:
            logger.warning(f"  {segment}: мало данных в val ({val_mask.sum()}), пропускаем")
            continue
        
        features = features_by_segment[segment]
        segment_params = best_params_by_segment.get(segment, {})
        params = segment_params.get('best_params', {})
        use_log = segment_params.get('use_log', False)
        
        logger.info(f"\n--- Сегмент {segment} ---")
        logger.info(f"  Признаков: {len(features)}")
        logger.info(f"  Use log: {use_log}")
        logger.info(f"  Train: {train_mask.sum():,} строк")
        logger.info(f"  Val: {val_mask.sum():,} строк")
        
        # Подготовка данных
        X_train = train_df.loc[train_mask, features].fillna(0)
        y_train = train_df.loc[train_mask, target_col]
        X_val = val_df.loc[val_mask, features].fillna(0)
        y_val = val_df.loc[val_mask, target_col]
        
        # Логарифмирование если нужно
        if use_log:
            y_train = np.log1p(y_train)
        
        # Обучение
        model = LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Предсказание
        pred_log = model.predict(X_val)
        if use_log:
            pred = np.expm1(pred_log)
        else:
            pred = pred_log
        
        predictions[val_mask] = pred
        models[segment] = model
        
        # Метрики для сегмента
        lift_result = calculate_lift(
            y_val.values, pred,
            val_df.loc[val_mask, 'lag1_amount_11_sum'].values
        )
        
        segment_metrics[segment] = {
            'lift': lift_result['lift'],
            'model_sum': lift_result['model_sum'],
            'baseline_sum': lift_result['baseline_sum'],
            'ideal_sum': lift_result['ideal_sum'],
            'count_train': train_mask.sum(),
            'count_val': val_mask.sum()
        }
        
        logger.info(f"  Lift@20% vs lag1: {lift_result['lift']:+.1f}%")
        logger.info(f"  Топ-20% по модели: {lift_result['model_sum']:,.0f} ₽")
        logger.info(f"  Топ-20% по lag1:   {lift_result['baseline_sum']:,.0f} ₽")
    
    # Глобальные метрики
    global_lift_result = calculate_lift(
        val_df['amount_11_sum'].values,
        predictions,
        val_df['lag1_amount_11_sum'].values
    )
    
    logger.info("\n" + "="*60)
    logger.info("ГЛОБАЛЬНЫЕ РЕЗУЛЬТАТЫ НА VAL")
    logger.info("="*60)
    logger.info(f"Глобальный lift@20%: {global_lift_result['lift']:+.1f}%")
    logger.info(f"Топ-20% по модели: {global_lift_result['model_sum']:,.0f} ₽")
    logger.info(f"Топ-20% по lag1:   {global_lift_result['baseline_sum']:,.0f} ₽")
    logger.info(f"Топ-20% идеал:     {global_lift_result['ideal_sum']:,.0f} ₽")
    
    return models, predictions, segment_metrics, global_lift_result['lift']


def run_validation_stage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    opt_results: Dict[str, Dict[str, Any]],
    stage_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Запускает walk-forward валидацию и расчет метрик.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка
    val_df : pd.DataFrame
        Валидационная выборка
    features_by_segment : dict
        Признаки по сегментам
    opt_results : dict
        Результаты оптимизации гиперпараметров
    stage_dir : Path
        Директория для сохранения результатов
    logger : logging.Logger, optional
        Логгер
    
    Returns
    -------
    wf_results : pd.DataFrame
        Результаты walk-forward валидации
    segment_metrics : dict
        Бизнес-метрики по сегментам
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("WALK-FORWARD ВАЛИДАЦИЯ")
    logger.info("="*60)
    
    # Запускаем walk-forward валидацию
    wf_results = walk_forward_validation(
        train_df=train_df,
        val_df=val_df,
        features_by_segment=features_by_segment,
        best_params_by_segment=opt_results,
        logger=logger
    )
    
    # Сохраняем результаты
    output_path = stage_dir / "walk_forward_results.csv"
    wf_results.to_csv(output_path, index=False)
    logger.info(f"Результаты сохранены в {output_path}")
    
    # Расчет бизнес-метрик
    segment_metrics = calculate_all_segments_metrics(wf_results)
    log_business_metrics(logger, segment_metrics)
    
    return wf_results, segment_metrics

