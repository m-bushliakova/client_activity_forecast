"""
Модуль для отбора признаков с временной кросс-валидацией.
Реализует двухэтапный подход:
1. Удаление сильно коррелирующих признаков
2. Forward selection для выбора наиболее важных признаков
"""
import sys
import json
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any


sys.path.append(str(Path(__file__).parent.parent))

from src.cross_val_utils import prepare_cv_folds
from src.config import EXCLUDED_COLUMNS, get_feature_selection_params


def select_features_by_correlation(
    df: pd.DataFrame,
    target: str,
    all_features: List[str],
    corr_threshold: float = 0.95,
    plot: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], Dict[str, float], List[Set[str]]]:
    """
    Первый этап отбора: удаление сильно коррелирующих между собой признаков.
    
    Алгоритм:
    1. Строим граф, где вершины - признаки, ребра - сильная корреляция (> threshold)
    2. Находим связные компоненты (кластеры коррелирующих признаков)
    3. В каждом кластере оставляем признак с максимальной корреляцией с target
    
    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм с данными
    target : str
        Название целевой переменной
    all_features : List[str]
        Список всех потенциальных признаков
    corr_threshold : float, default=0.95
        Порог корреляции для удаления
    plot : bool, default=True
        Строить ли визуализации
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    features_to_keep : List[str]
        Отобранные признаки после корреляционного анализа
    target_corr : Dict[str, float]
        Корреляции признаков с целевой переменной
    clusters : List[Set[str]]
        Кластеры коррелирующих признаков
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("ЭТАП 1: АНАЛИЗ КОРРЕЛЯЦИЙ")
    
    # Считаем корреляции с целевой переменной
    target_corr = {}
    valid_features = []
    
    for feature in all_features:
        if feature not in df.columns:
            logger.debug(f"Признак {feature} отсутствует в данных, пропускаем")
            continue
        
        corr = df[feature].corr(df[target])
        if not np.isnan(corr):
            target_corr[feature] = corr
            valid_features.append(feature)
    
    if not valid_features:
        logger.warning("Нет валидных признаков после проверки корреляций")
        return [], target_corr, []
    
    logger.info(f"Признаков: {len(valid_features)} | "
                f"Корреляции: {min(target_corr.values()):.3f} - {max(target_corr.values()):.3f}")
    
    # Находим пары с высокой корреляцией
    high_corr_pairs = []
    for i in range(len(valid_features)):
        for j in range(i+1, len(valid_features)):
            f1, f2 = valid_features[i], valid_features[j]
            corr = df[f1].corr(df[f2])
            if abs(corr) > corr_threshold:
                high_corr_pairs.append((f1, f2, corr))
    
    # Строим граф корреляций
    correlation_graph = nx.Graph()
    for f1, f2, _ in high_corr_pairs:
        correlation_graph.add_edge(f1, f2)
    
    # Находим связные компоненты (кластеры)
    clusters = list(nx.connected_components(correlation_graph))
    logger.info(f"Кластеров: {len(clusters)} | Пар с высокой корреляцией: {len(high_corr_pairs)}")
    
    # Из каждого кластера оставляем признак с максимальной корреляцией с target
    features_to_keep = []
    features_to_drop = []
    
    for cluster in clusters:
        # Выбираем лучший признак в кластере
        best_feature = max(cluster, key=lambda f: abs(target_corr.get(f, 0)))
        features_to_keep.append(best_feature)
        
        # Остальные признаки кластера помечаем к удалению
        cluster_features = list(cluster)
        cluster_features.remove(best_feature)
        features_to_drop.extend(cluster_features)
        
        logger.debug(f"Кластер: {len(cluster)} признаков, лучший: {best_feature} "
                    f"(corr={target_corr[best_feature]:.3f})")
    
    # Добавляем признаки вне кластеров
    all_clustered = set().union(*clusters)
    lonely_features = [f for f in valid_features if f not in all_clustered]
    features_to_keep.extend(lonely_features)
    
    logger.info(f"После отбора: {len(features_to_keep)} признаков "
                f"(удалено {len(features_to_drop)})")
    
    if plot and len(features_to_keep) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Гистограмма корреляций с target
        corr_values = list(target_corr.values())
        axes[0].hist(corr_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_xlabel('Корреляция с target')
        axes[0].set_ylabel('Количество признаков')
        axes[0].set_title('Распределение корреляций')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Сравнение количества признаков
        axes[1].bar(['Исходные', 'После'], [len(valid_features), len(features_to_keep)], 
                   color=['lightcoral', 'lightgreen'])
        axes[1].set_ylabel('Количество признаков')
        axes[1].set_title('Сокращение признаков')
        
        plt.tight_layout()
        plt.show()
    
    return features_to_keep, target_corr, clusters


def forward_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    candidate_features: List[str],
    stopping_rounds: int = 10,
    improvement_threshold: float = 0.0005,
    plot: bool = True,
    model_params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Второй этап отбора: прямой отбор признаков (forward selection).
    
    Алгоритм:
    1. Начинаем с пустого набора признаков
    2. На каждом шаге добавляем признак, который дает наибольшее улучшение
    3. Останавливаемся, если нет улучшений в течение stopping_rounds
    
    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Обучающая и валидационная выборки
    target : str
        Название целевой переменной
    candidate_features : List[str]
        Список кандидатов для отбора (после корреляционного анализа)
    stopping_rounds : int, default=10
        Количество шагов без улучшения для остановки
    improvement_threshold : float, default=0.0005
        Минимальное улучшение для добавления признака
    plot : bool, default=True
        Строить ли график прогресса
    model_params : dict, optional
        Параметры модели LightGBM
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    selected_features : List[str]
        Финальный список отобранных признаков
    history_df : pd.DataFrame
        История отбора с метриками на каждом шаге
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if model_params is None:
        model_params = {'n_estimators': 100, 'random_state': 42, 'verbose': -1}
    
    logger.info("="*60)
    logger.info("ЭТАП 2: ПРЯМОЙ ОТБОР ПРИЗНАКОВ")
    
    # Сортируем кандидатов по абсолютной корреляции с target
    corr_with_target = {
        f: train_df[f].corr(train_df[target]) 
        for f in candidate_features 
        if f in train_df.columns
    }
    
    sorted_features = sorted(
        candidate_features,
        key=lambda f: abs(corr_with_target.get(f, 0)),
        reverse=True
    )
    
    logger.info(f"Кандидатов: {len(sorted_features)}")
    
    # Подготавливаем данные
    X_train_full = train_df[sorted_features]
    y_train = train_df[target]
    X_val_full = val_df[sorted_features]
    y_val = val_df[target]
    
    selected = []
    best_train_score = -np.inf
    no_improve_counter = 0
    history = []
    
    total_candidates = len(sorted_features)
    
    for step_idx, feature in enumerate(sorted_features, 1):
        # Логируем прогресс
        if step_idx % 10 == 0 or step_idx == total_candidates:
            logger.info(f"Шаг {step_idx}/{total_candidates} | "
                       f"отобрано: {len(selected):2d} | "
                       f"лучший Train R2: {best_train_score:.4f}")
        
        # Тестируем добавление признака
        test_features = selected + [feature]
        
        model = LGBMRegressor(**model_params)
        model.fit(X_train_full[test_features], y_train)
        
        train_pred = model.predict(X_train_full[test_features])
        val_pred = model.predict(X_val_full[test_features])
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        # Проверяем улучшение
        if train_r2 > best_train_score + improvement_threshold:
            selected.append(feature)
            best_train_score = train_r2
            no_improve_counter = 0
            logger.debug(f"  → Добавлен {feature}, улучшение до {train_r2:.4f}")
        else:
            no_improve_counter += 1
        
        history.append({
            'step': step_idx,
            'feature': feature,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'best_train_r2_so_far': best_train_score,
            'n_selected': len(selected),
            'was_selected': feature in selected
        })
        
        # Проверка условия остановки
        if no_improve_counter >= stopping_rounds:
            logger.info(f"Остановка: {stopping_rounds} шагов без улучшений")
            break
    
    history_df = pd.DataFrame(history)
    
    logger.info(f"Отобрано: {len(selected)} | "
                f"Лучший Train R2: {best_train_score:.4f}")
    
    # Финальная оценка на валидации
    if selected:
        final_model = LGBMRegressor(**model_params)
        final_model.fit(train_df[selected], train_df[target])
        final_val_pred = final_model.predict(val_df[selected])
        final_val_r2 = r2_score(val_df[target], final_val_pred)
        logger.info(f"Финальный Val R2: {final_val_r2:.4f}")
    
    if plot and history_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['step'], history_df['train_r2'], 'b-', label='Train R2', alpha=0.7)
        plt.plot(history_df['step'], history_df['val_r2'], 'r--', label='Val R2', alpha=0.7)
        
        # Отмечаем моменты добавления признаков
        added_steps = history_df[history_df['was_selected']]['step']
        added_train_r2 = history_df[history_df['was_selected']]['train_r2']
        plt.scatter(added_steps, added_train_r2, color='green', s=50, 
                   zorder=5, label='Признак добавлен')
        
        plt.xlabel('Шаг')
        plt.ylabel('R2')
        plt.title('Forward Selection Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return selected, history_df


def get_features_for_segment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target: str,
    base_features: List[str],
    corr_threshold: float = 0.95,
    stopping_rounds: int = 10,
    improvement_threshold: float = 0.0005,
    plot: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Полный пайплайн отбора признаков для одного сегмента.
    
    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Обучающая и валидационная выборки для сегмента
    target : str
        Название целевой переменной
    base_features : List[str]
        Базовый список всех признаков
    corr_threshold, stopping_rounds, improvement_threshold : various
        Параметры отбора
    plot : bool, default=True
        Строить ли визуализации
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    final_features : List[str]
        Финальные отобранные признаки
    stage1_features : List[str]
        Признаки после первого этапа
    history_df : pd.DataFrame
        История отбора
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Сегмент | данных: {len(train_df):,} строк")
    
    # Этап 1: удаление коррелирующих признаков
    stage1_features, target_corr, clusters = select_features_by_correlation(
        train_df, target, base_features, corr_threshold, plot=plot, logger=logger
    )
    
    if not stage1_features:
        logger.warning("Нет признаков после корреляционного анализа")
        return [], [], pd.DataFrame()
    
    # Этап 2: прямой отбор
    final_features, history_df = forward_selection(
        train_df, val_df, target, stage1_features,
        stopping_rounds=stopping_rounds,
        improvement_threshold=improvement_threshold,
        plot=plot, logger=logger
    )
    
    reduction_pct = (1 - len(final_features) / len(base_features)) * 100 if base_features else 0
    logger.info(f"Исходных: {len(base_features)} | Финальных: {len(final_features)} | "
                f"Сокращение: {reduction_pct:.1f}%")
    
    return final_features, stage1_features, history_df


def load_features_from_json(
    filename: str = 'features_by_segment.json',
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    Загружает список признаков для каждого сегмента из JSON файла.
    
    Parameters
    ----------
    filename : str, default='features_by_segment.json'
        Путь к JSON файлу
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, List[str]]
        Словарь вида {название_сегмента: список_признаков}
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        with open(filename, 'r') as f:
            features_by_segment = json.load(f)
        
        logger.info("ЗАГРУЖЕННЫЕ ПРИЗНАКИ")
        for seg, feats in features_by_segment.items():
            logger.info(f"{seg}: {len(feats)} признаков")
        
        return features_by_segment
    
    except FileNotFoundError:
        logger.error(f"Файл {filename} не найден")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Ошибка парсинга JSON в файле {filename}")
        return {}


def get_segment_data(
    train: pd.DataFrame,
    segment: str,
    min_rows: int = 100,
    logger: Optional[logging.Logger] = None
) -> Optional[pd.DataFrame]:
    """
    Получает данные для конкретного сегмента с проверкой минимального количества.
    
    Parameters
    ----------
    train : pd.DataFrame
        Обучающая выборка (должна содержать колонку 'segment')
    segment : str
        Название сегмента
    min_rows : int, default=100
        Минимальное количество строк для сегмента
    logger : logging.Logger, optional
        Логгер для вывода предупреждений
    
    Returns
    -------
    Optional[pd.DataFrame]
        Данные сегмента или None, если данных недостаточно
    """
    segment_mask = train['segment'] == segment
    segment_data = train[segment_mask].copy()
    
    if len(segment_data) < min_rows:
        if logger:
            logger.warning(f"{segment}: мало данных ({len(segment_data)}), пропускаем")
        return None
    
    return segment_data


def run_fold_selection(
    segment_data: pd.DataFrame,
    fold_info: Dict[str, Any],
    target: str,
    base_features: List[str],
    feature_selection_params: Dict[str, Any],
    logger: logging.Logger
) -> Optional[Set[str]]:
    """
    Запускает отбор признаков для одного фолда кросс-валидации.
    
    Parameters
    ----------
    segment_data : pd.DataFrame
        Данные сегмента (все месяцы)
    fold_info : Dict
        Информация о фолде от prepare_cv_folds
    target : str
        Название целевой переменной
    base_features : List[str]
        Базовый список признаков
    feature_selection_params : Dict
        Параметры отбора признаков
    logger : logging.Logger
        Логгер для вывода информации
    
    Returns
    -------
    Optional[Set[str]]
        Множество отобранных признаков или None при ошибке
    """
    fold = fold_info['fold']
    train_months = fold_info['train_months']
    val_months = fold_info['val_months']
    
    logger.info(f"Фолд {fold}: Train {len(train_months)} мес, Val {len(val_months)} мес")
    
    # Фильтруем данные по месяцам
    train_mask = segment_data['year_month'].isin(train_months)
    val_mask = segment_data['year_month'].isin(val_months)
    
    train_fold = segment_data[train_mask].copy()
    val_fold = segment_data[val_mask].copy()
    
    # Проверяем достаточно ли данных
    if len(train_fold) < 50 or len(val_fold) < 10:
        logger.warning(f"    Мало данных: train={len(train_fold)}, val={len(val_fold)}")
        return None
    
    # Запускаем отбор признаков
    final_features, _, _ = get_features_for_segment(
        train_df=train_fold,
        val_df=val_fold,
        target=target,
        base_features=base_features,
        plot=False,
        logger=logger,
        **feature_selection_params
    )
    
    return set(final_features)


def aggregate_cv_results(
    all_selected_features: List[Set[str]],
    strategy: str = 'frequent',
    min_freq_ratio: float = 0.5,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    Агрегирует результаты отбора признаков по всем фолдам.
    
    Стратегии агрегации:
    - 'common' : только признаки, отобранные во всех фолдах
    - 'frequent' : признаки, отобранные в >= min_freq_ratio фолдов
    - 'union' : объединение всех признаков из всех фолдов
    
    Parameters
    ----------
    all_selected_features : List[Set[str]]
        Список множеств признаков для каждого фолда
    strategy : str, default='frequent'
        Стратегия агрегации
    min_freq_ratio : float, default=0.5
        Минимальная доля фолдов для стратегии 'frequent'
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    selected_features : List[str]
        Отобранные признаки после агрегации
    feature_counts : Dict[str, int]
        Количество фолдов, в которых отобран каждый признак
    """
    if not all_selected_features:
        return [], {}
    
    n_folds = len(all_selected_features)
    
    # Считаем частоту появления признаков
    feature_counts = {}
    for feat_set in all_selected_features:
        for feat in feat_set:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
    
    # Применяем стратегию
    if strategy == 'common' and n_folds > 1:
        selected = set.intersection(*all_selected_features)
        selected = list(selected)
    elif strategy == 'frequent':
        threshold = max(1, int(n_folds * min_freq_ratio))
        selected = [feat for feat, count in feature_counts.items() if count >= threshold]
    elif strategy == 'union':
        selected = list(set().union(*all_selected_features))
    else:
        selected = list(feature_counts.keys())
    
    # Сортируем по частоте (самые частые первые)
    selected = sorted(selected, key=lambda f: feature_counts[f], reverse=True)
    
    if logger:
        logger.info(f"Отобрано после агрегации: {len(selected)} признаков "
                    f"(из {len(feature_counts)} уникальных)")
    
    return selected, feature_counts


def run_feature_selection_with_cv(
    train_df: pd.DataFrame,
    segment_labels: List[str],
    target_col: str = 'amount_11_sum',
    log_target_col: str = 'log_amount',
    n_splits: int = 3,
    selection_strategy: str = 'frequent',
    min_freq_ratio: float = 0.5,
    feature_selection_params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    Запускает отбор признаков для всех сегментов с временной кросс-валидацией.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка (должна содержать колонки 'segment', 'year_month')
    segment_labels : List[str]
        Список названий сегментов
    target_col : str, default='amount_11_sum'
        Название целевой переменной
    log_target_col : str, default='log_amount'
        Название логарифмической целевой переменной
    n_splits : int, default=3
        Количество фолдов для кросс-валидации
    selection_strategy : str, default='frequent'
        Стратегия агрегации результатов
    min_freq_ratio : float, default=0.5
        Минимальная доля фолдов для стратегии 'frequent'
    feature_selection_params : dict, optional
        Параметры отбора признаков
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, List[str]]
        Словарь вида {название_сегмента: список_признаков}
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Получаем параметры отбора
    if feature_selection_params is None:
        feature_selection_params = get_feature_selection_params()
    
    # Получаем все месяцы в хронологическом порядке
    train_months = sorted(train_df['year_month'].unique())
    logger.info(f"Всего месяцев в train: {len(train_months)}")
    
    # Создаем фолды для временной CV
    folds = prepare_cv_folds(train_months, n_splits)
    
    # Определяем базовые признаки (исключаем служебные колонки)
    base_features = [
        col for col in train_df.columns 
        if col not in EXCLUDED_COLUMNS
    ]
    
    logger.info(f"Всего базовых признаков: {len(base_features)}")
    
    features_by_segment = {}
    
    for segment in segment_labels:
        logger.info(f"\nСЕГМЕНТ {segment}")
        
        # Получаем данные сегмента
        segment_data = get_segment_data(train_df, segment, logger=logger)
        if segment_data is None:
            features_by_segment[segment] = []
            continue
        
        # Определяем целевую переменную
        current_target = log_target_col if segment != segment_labels[-1] else target_col
        logger.info(f"  Целевая переменная: {current_target}")
        
        # Собираем результаты по всем фолдам
        all_selected_features = []
        
        for fold_info in folds:
            result = run_fold_selection(
                segment_data=segment_data,
                fold_info=fold_info,
                target=current_target,
                base_features=base_features,
                feature_selection_params=feature_selection_params,
                logger=logger
            )
            
            if result is not None:
                all_selected_features.append(result)
        
        # Проверяем, есть ли результаты
        if not all_selected_features:
            logger.warning(f"  Нет результатов по фолдам для сегмента {segment}")
            features_by_segment[segment] = []
            continue
        
        # Агрегируем результаты
        selected_features, feature_counts = aggregate_cv_results(
            all_selected_features,
            strategy=selection_strategy,
            min_freq_ratio=min_freq_ratio,
            logger=logger
        )
        
        features_by_segment[segment] = selected_features
        logger.info(f"{segment}: отобрано {len(selected_features)} признаков")
    
    return features_by_segment


def run_feature_selection_stage(
    train_df: pd.DataFrame,
    segment_labels: List[str],
    feature_selection_params: Optional[Dict] = None,
    hyperopt_params: Optional[Dict] = None,
    output_dir: Path = Path('pipeline_results/dev'),
    logger: Optional[logging.Logger] = None
) -> Dict[str, List[str]]:
    """
    Запускает полный этап отбора признаков с сохранением результатов.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка
    segment_labels : List[str]
        Список названий сегментов
    feature_selection_params : dict, optional
        Параметры отбора признаков
    hyperopt_params : dict, optional
        Параметры оптимизации (нужны только для n_splits)
    output_dir : Path
        Директория для сохранения результатов
    logger : logging.Logger, optional
        Логгер для вывода информации
    
    Returns
    -------
    Dict[str, List[str]]
        Словарь с отобранными признаками для каждого сегмента
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("ОТБОР ПРИЗНАКОВ С КРОСС-ВАЛИДАЦИЕЙ")
    logger.info("="*60)
    
    # Получаем параметры
    if feature_selection_params is None:
        feature_selection_params = get_feature_selection_params()
    
    # Количество фолдов из hyperopt_params или по умолчанию
    n_splits = hyperopt_params.get('n_splits', 3) if hyperopt_params else 3
    
    # Запускаем отбор
    features_by_segment = run_feature_selection_with_cv(
        train_df=train_df,
        segment_labels=segment_labels,
        n_splits=n_splits,
        feature_selection_params=feature_selection_params,
        logger=logger
    )
    
    # Сохраняем результаты
    output_path = output_dir / "features_by_segment.json"
    features_serializable = {
        segment: list(features) 
        for segment, features in features_by_segment.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(features_serializable, f, indent=2)
    
    logger.info(f"Результаты сохранены в {output_path}")
    
    return features_by_segment