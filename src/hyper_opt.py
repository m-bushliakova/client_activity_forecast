"""
Модуль для оптимизации гиперпараметров моделей с временной кросс-валидацией.
Использует Optuna для поиска оптимальных параметров и агрегирует результаты по фолдам.
"""

import optuna
import sys
import json
import pickle
import warnings
import logging
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.cross_val_utils import prepare_cv_folds
from src.config import get_hyperopt_params

warnings.filterwarnings('ignore')


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Целевая функция для оптимизации гиперпараметров с помощью Optuna.
    
    Parameters
    ----------
    trial : optuna.Trial
        Объект trial для предложения параметров
    X_train, X_val : pd.DataFrame
        Признаки для обучения и валидации
    y_train, y_val : pd.Series
        Целевые переменные
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    float
        R2 на валидационной выборке (максимизируем)
    """
    # Предлагаем гиперпараметры для оптимизации
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1  # Используем все ядра
    }
    
    model = LGBMRegressor(**params)
    
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            # callbacks=[optuna.integration.LightGBMPruningCallback(trial, 'rmse')]
        )
        
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        
        return r2
    
    except Exception as e:
        if logger:
            logger.warning(f"Ошибка при обучении модели: {e}")
        return -np.inf  # Возвращаем худшее значение при ошибке


def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int,
    random_state: int = 42,
    timeout: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> optuna.Study:
    """
    Запускает оптимизацию гиперпараметров с Optuna.
    
    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Признаки для обучения и валидации
    y_train, y_val : pd.Series
        Целевые переменные
    n_trials : int
        Количество trials для оптимизации
    random_state : int, default=42
        Seed для воспроизводимости
    timeout : int, optional
        Максимальное время оптимизации в секундах
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    optuna.Study
        Объект study с результатами оптимизации
    """
    # Создаем study с максимизацией R2
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=20,
            interval_steps=10
        )
    )
    
    # Запускаем оптимизацию
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, logger),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        # callbacks=[optuna.study.MaxTrialsCallback(n_trials)]
    )
    
    if logger:
        logger.info(f"Лучшее значение R2: {study.best_value:.4f}")
        logger.debug(f"Лучшие параметры: {study.best_params}")
    
    return study


def aggregate_hyperopt_results(
    fold_results: List[Dict[str, Any]],
    strategy: str = 'frequent',
    min_freq_ratio: float = 0.5,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Агрегирует результаты оптимизации по фолдам.
    
    Стратегии агрегации:
    - 'common' : только параметры, совпадающие во всех фолдах
    - 'frequent' : наиболее частые значения для каждого параметра
    - 'union' : параметры из первого фолда
    
    Parameters
    ----------
    fold_results : List[Dict[str, Any]]
        Список лучших параметров для каждого фолда
    strategy : str, default='frequent'
        Стратегия агрегации
    min_freq_ratio : float, default=0.5
        Минимальная доля фолдов для стратегии 'frequent'
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, Any]
        Агрегированные гиперпараметры
    
    Raises
    ------
    ValueError
        Если указана неизвестная стратегия
    """
    if not fold_results:
        if logger:
            logger.warning("Нет результатов для агрегации")
        return {}
    
    if strategy == 'common':
        # Только параметры, одинаковые во всех фолдах
        common_params = {}
        first_params = fold_results[0]
        
        for key in first_params.keys():
            values = [res.get(key) for res in fold_results]
            if all(v == values[0] for v in values):
                common_params[key] = values[0]
        
        if logger:
            logger.info(f"Общих параметров: {len(common_params)} из {len(first_params)}")
        
        return common_params
    
    elif strategy == 'frequent':
        # Наиболее частые значения для каждого параметра
        n_folds = len(fold_results)
        threshold = max(1, int(n_folds * min_freq_ratio))
        
        aggregated = {}
        all_keys = fold_results[0].keys()
        
        for key in all_keys:
            # Собираем все значения параметра
            values = [res.get(key) for res in fold_results if key in res]
            
            if len(values) < threshold:
                if logger:
                    logger.debug(f"Параметр {key}: недостаточно значений ({len(values)} < {threshold})")
                continue
            
            # Для категориальных параметров берем самое частое
            if isinstance(values[0], (str, bool)):
                most_common = Counter(values).most_common(1)[0]
                if most_common[1] >= threshold:
                    aggregated[key] = most_common[0]
                else:
                    if logger:
                        logger.debug(f"Параметр {key}: частота {most_common[1]} < {threshold}")
            
            # Для числовых параметров берем среднее
            else:
                # Для целочисленных параметров округляем
                mean_val = np.mean(values)
                if key in ['n_estimators', 'num_leaves', 'max_depth', 'min_child_samples']:
                    aggregated[key] = int(round(mean_val))
                else:
                    aggregated[key] = mean_val
        
        if logger:
            logger.info(f"Агрегировано {len(aggregated)} параметров из {len(all_keys)}")
        
        return aggregated
    
    elif strategy == 'union':
        # Берем параметры из первого фолда
        return fold_results[0]
    
    else:
        raise ValueError(f"Неизвестная стратегия агрегации: {strategy}")


def save_optimization_results(
    results: Dict[str, Dict[str, Any]],
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Сохраняет результаты оптимизации в файлы (pickle и JSON).
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Результаты оптимизации для каждого сегмента
    output_dir : Path, optional
        Директория для сохранения
    filename : str, optional
        Имя файла (без расширения)
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Path
        Путь к сохраненному файлу
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_results_{timestamp}"
    
    # Подготавливаем данные для сохранения
    results_to_save = {}
    for segment, res in results.items():
        results_to_save[segment] = {
            'best_params': res.get('best_params', {}),
            'use_log': res.get('use_log', False),
            'best_value': res.get('best_value', None)
        }
    
    # Сохраняем в pickle
    pickle_path = output_dir / f"{filename}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'results': results_to_save,
            'timestamp': datetime.now().isoformat(),
            'n_segments': len(results),
            'version': '1.0'
        }, f)
    
    # Сохраняем в JSON для удобства
    json_path = output_dir / f"{filename}.json"
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    if logger:
        logger.info(f"Результаты сохранены: {pickle_path}")
    
    return pickle_path


def load_optimization_results(
    filepath: Union[str, Path],
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Загружает результаты оптимизации из pickle файла.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Путь к файлу
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, Any]
        Загруженные результаты
    
    Raises
    ------
    FileNotFoundError
        Если файл не найден
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Файл {filepath} не найден")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if logger:
        logger.info(f"Загружено {data.get('n_segments', 0)} сегментов "
                   f"от {data.get('timestamp', 'unknown')}")
    
    return data.get('results', {})


def optimize_hyperparameters_for_segment_with_cv(
    segment_name: str,
    train_df: pd.DataFrame,
    features: List[str],
    target_col: str,
    n_trials: int = 100,
    n_splits: int = 3,
    use_log_target: bool = True,
    aggregation_strategy: str = 'frequent',
    min_freq_ratio: float = 0.5,
    random_state: int = 42,
    timeout_per_fold: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """
    Подбирает гиперпараметры для модели одного сегмента с временной кросс-валидацией.
    
    Процесс:
    1. Создаем временные фолды (обучение на прошлых месяцах, валидация на следующем)
    2. Для каждого фолда запускаем Optuna
    3. Агрегируем лучшие параметры по всем фолдам
    
    Parameters
    ----------
    segment_name : str
        Название сегмента (для логирования)
    train_df : pd.DataFrame
        Обучающая выборка для сегмента
    features : List[str]
        Список признаков для модели
    target_col : str
        Название целевой переменной
    n_trials : int, default=100
        Общее количество trials (распределяется по фолдам)
    n_splits : int, default=3
        Количество фолдов для временной кросс-валидации
    use_log_target : bool, default=True
        Использовать ли логарифмический target
    aggregation_strategy : str, default='frequent'
        Стратегия агрегации результатов по фолдам
    min_freq_ratio : float, default=0.5
        Минимальная доля фолдов для стратегии 'frequent'
    random_state : int, default=42
        Seed для воспроизводимости
    timeout_per_fold : int, optional
        Максимальное время на фолд в секундах
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    dict or None
        Агрегированные лучшие гиперпараметры или None при ошибке
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Получаем все месяцы в train
    train_months = sorted(train_df['year_month'].unique())
    
    if len(train_months) < n_splits + 1:
        logger.warning(f"{segment_name}: недостаточно месяцев "
                       f"({len(train_months)} < {n_splits + 1})")
        return None
    
    # Создаем фолды
    folds = prepare_cv_folds(train_months, n_splits)
    
    logger.info(f"CV для {segment_name}: {n_splits} фолдов, {n_trials} trials")
    
    fold_results = []
    trials_per_fold = max(5, n_trials // n_splits)  # Минимум 5 trials на фолд
    
    for fold_info in folds:
        fold = fold_info['fold']
        train_months_fold = fold_info['train_months']
        val_months_fold = fold_info['val_months']
        
        # Фильтруем данные по месяцам
        train_mask = train_df['year_month'].isin(train_months_fold)
        val_mask = train_df['year_month'].isin(val_months_fold)
        
        train_fold = train_df[train_mask].copy()
        val_fold = train_df[val_mask].copy()
        
        # Проверяем достаточно ли данных
        if len(train_fold) < 50 or len(val_fold) < 10:
            logger.debug(f"Фолд {fold}: недостаточно данных "
                        f"(train={len(train_fold)}, val={len(val_fold)})")
            continue
        
        # Подготавливаем данные
        X_train = train_fold[features].fillna(0)
        y_train = train_fold[target_col]
        X_val = val_fold[features].fillna(0)
        y_val = val_fold[target_col]
        
        if use_log_target:
            y_train = np.log1p(y_train)
            y_val = np.log1p(y_val)
        
        # Запускаем оптимизацию
        try:
            study = run_optuna_study(
                X_train, y_train, X_val, y_val,
                trials_per_fold, random_state, timeout_per_fold, logger
            )
            
            if study.best_params:
                fold_results.append(study.best_params)
                if logger:
                    logger.debug(f"Фолд {fold}: params={study.best_params}, "
                                f"R2={study.best_value:.4f}")
            
        except Exception as e:
            logger.warning(f"Ошибка в фолде {fold}: {e}")
            continue
    
    if not fold_results:
        logger.warning(f"{segment_name}: нет результатов ни по одному фолду")
        return None
    
    # Агрегируем результаты
    best_params = aggregate_hyperopt_results(
        fold_results,
        strategy=aggregation_strategy,
        min_freq_ratio=min_freq_ratio,
        logger=logger
    )
    
    # Добавляем фиксированные параметры
    best_params['random_state'] = random_state
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    
    logger.info(f"{segment_name}: параметры подобраны ({len(best_params)} параметров)")
    
    return best_params


def optimize_all_segments_with_cv(
    train_df: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    target_col: str = 'amount_11_sum',
    use_log_for_segments: Optional[List[str]] = None,
    n_trials: int = 100,
    n_splits: int = 3,
    aggregation_strategy: str = 'frequent',
    min_freq_ratio: float = 0.5,
    random_state: int = 42,
    timeout_per_fold: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Подбирает гиперпараметры для всех сегментов с временной кросс-валидацией.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка (должна содержать колонку 'segment')
    features_by_segment : dict
        Словарь вида {segment_name: list_of_features}
    target_col : str, default='amount_11_sum'
        Название целевой переменной
    use_log_for_segments : list of str, optional
        Список сегментов для использования логарифмического target
        Если None, то для всех кроме последнего
    n_trials : int, default=100
        Количество trials для каждого сегмента
    n_splits : int, default=3
        Количество фолдов для временной кросс-валидации
    aggregation_strategy : str, default='frequent'
        Стратегия агрегации результатов по фолдам
    min_freq_ratio : float, default=0.5
        Минимальная доля фолдов для стратегии 'frequent'
    random_state : int, default=42
        Seed для воспроизводимости
    timeout_per_fold : int, optional
        Максимальное время на фолд в секундах
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    dict
        Результаты для каждого сегмента в формате:
        {
            'S1': {'best_params': {...}, 'use_log': True, 'best_value': float},
            ...
        }
    """
    # Отключаем вывод Optuna для чистоты логов
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Определяем сегменты для логарифмического target
    all_segments = sorted(train_df['segment'].unique())
    
    if use_log_for_segments is None:
        # По умолчанию все кроме последнего (самого крупного)
        use_log_for_segments = all_segments[:-1]
        logger.info(f"Лог target для сегментов: {use_log_for_segments}")
    
    results = {}
    
    for segment in all_segments:
        logger.info(f"\n{'='*50}")
        logger.info(f"СЕГМЕНТ {segment}")
        logger.info(f"{'='*50}")
        
        # Фильтруем данные по сегменту
        segment_mask = train_df['segment'] == segment
        segment_data = train_df[segment_mask].copy()
        
        if len(segment_data) < 100:
            logger.warning(f"  Пропущен: мало данных в train ({len(segment_data)})")
            continue
        
        # Получаем признаки для сегмента
        features = features_by_segment.get(segment, [])
        
        if not features:
            logger.warning(f"  Нет признаков для сегмента {segment}")
            continue
        
        # Определяем, использовать ли логарифмический target
        use_log = segment in use_log_for_segments
        
        logger.info(f"  Признаков: {len(features)}")
        logger.info(f"  Use log: {use_log}")
        logger.info(f"  Данных: {len(segment_data):,} строк")
        logger.info(f"  Месяцев: {segment_data['year_month'].nunique()}")
        
        # Подбираем гиперпараметры
        best_params = optimize_hyperparameters_for_segment_with_cv(
            segment_name=segment,
            train_df=segment_data,
            features=features,
            target_col=target_col,
            n_trials=n_trials,
            n_splits=n_splits,
            use_log_target=use_log,
            aggregation_strategy=aggregation_strategy,
            min_freq_ratio=min_freq_ratio,
            random_state=random_state,
            timeout_per_fold=timeout_per_fold,
            logger=logger
        )
        
        if best_params:
            results[segment] = {
                'best_params': best_params,
                'use_log': use_log,
                'n_features': len(features),
                'n_samples': len(segment_data)
            }
            logger.info(f"  ✅ Параметры подобраны: {best_params}")
        else:
            logger.warning(f"  ❌ Не удалось подобрать параметры")
    
    return results


def run_hyperopt_stage(
    train_df: pd.DataFrame,
    features_by_segment: Dict[str, List[str]],
    segment_labels: List[str],
    hyperopt_params: Optional[Dict] = None,
    output_dir: Path = Path('pipeline_results/dev'),
    logger: Optional[logging.Logger] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Запускает этап оптимизации гиперпараметров с сохранением результатов.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка
    features_by_segment : dict
        Признаки по сегментам
    segment_labels : List[str]
        Список названий сегментов
    hyperopt_params : dict, optional
        Параметры оптимизации
    output_dir : Path
        Директория для сохранения результатов
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Результаты оптимизации для каждого сегмента
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ С CV")
    logger.info("="*60)
    
    # Получаем параметры
    if hyperopt_params is None:
        hyperopt_params = get_hyperopt_params()
    
    # Запускаем оптимизацию
    opt_results = optimize_all_segments_with_cv(
        train_df=train_df,
        features_by_segment=features_by_segment,
        target_col='amount_11_sum',
        use_log_for_segments=segment_labels[:-1],
        n_trials=hyperopt_params.get('n_trials', 100),
        n_splits=hyperopt_params.get('n_splits', 3),
        aggregation_strategy=hyperopt_params.get('aggregation_strategy', 'frequent'),
        min_freq_ratio=hyperopt_params.get('min_freq_ratio', 0.5),
        logger=logger
    )
    
    # Сохраняем результаты
    save_optimization_results(
        opt_results,
        output_dir=output_dir,
        filename="optimization_results",
        logger=logger
    )
    
    return opt_results