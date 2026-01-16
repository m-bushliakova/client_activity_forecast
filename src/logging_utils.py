"""
Утилиты для логирования и мониторинга пайплайна.
Обеспечивает единообразное логирование и интеграцию с MLflow.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import mlflow


def setup_logging(
    verbose: int = 1,
    log_file: Optional[Union[str, Path]] = None,
    logger_name: str = 'pipeline'
) -> logging.Logger:
    """
    Настраивает логирование для пайплайна.
    
    Parameters
    ----------
    verbose : int, default=1
        Уровень детализации:
        - 0: только ошибки
        - 1: основные этапы
        - 2: детальная информация (debug)
    log_file : str or Path, optional
        Путь к файлу для сохранения логов
    logger_name : str, default='pipeline'
        Имя логгера
    
    Returns
    -------
    logging.Logger
        Настроенный логгер
    """
    # Создаем или получаем логгер
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем консольный вывод
    console_handler = logging.StreamHandler(sys.stdout)
    
    if verbose == 0:
        console_handler.setLevel(logging.ERROR)
    elif verbose == 1:
        console_handler.setLevel(logging.INFO)
    else:  # verbose >= 2
        console_handler.setLevel(logging.DEBUG)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Настраиваем файловый вывод
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # В файл пишем всё
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Логи также сохраняются в {log_path}")
    
    return logger


def get_pipeline_logger(
    verbose: int = 1,
    log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Возвращает существующий логгер пайплайна или создает новый.
    
    Parameters
    ----------
    verbose : int, default=1
        Уровень детализации
    log_file : str or Path, optional
        Путь к файлу для сохранения логов
    
    Returns
    -------
    logging.Logger
        Логгер пайплайна
    """
    logger = logging.getLogger('pipeline')
    
    # Если логгер уже настроен, возвращаем его
    if logger.handlers:
        return logger
    
    # Иначе создаем новый с указанными параметрами
    return setup_logging(verbose=verbose, log_file=log_file)


def log_business_metrics(
    logger: logging.Logger,
    segment_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Логирует бизнес-метрики в читаемом виде.
    
    Parameters
    ----------
    logger : logging.Logger
        Логгер для вывода
    segment_metrics : dict
        Метрики по сегментам от calculate_all_segments_metrics
    
    Returns
    -------
    dict
        Те же метрики (для возможного дальнейшего использования)
    """
    if not segment_metrics:
        logger.warning("Нет метрик для отображения")
        return segment_metrics
    
    logger.info("\n" + "="*80)
    logger.info("БИЗНЕС-МЕТРИКИ ПО СЕГМЕНТАМ")
    logger.info("="*80)
    
    total_gain = 0
    total_baseline = 0
    working_segments = []
    
    for segment, metrics in sorted(segment_metrics.items()):
        logger.info(f"\n{segment}:")
        logger.info(f"  Средний lift: {metrics['avg_lift']:+.1f}%")
        logger.info(f"  За весь период ({metrics['months_count']} месяцев):")
        logger.info(f"    Модель:    {metrics['total_model']:>15,.0f} ₽")
        logger.info(f"    Baseline:  {metrics['total_baseline']:>15,.0f} ₽")
        logger.info(f"    Идеал:     {metrics['total_ideal']:>15,.0f} ₽")
        logger.info(f"    Прирост:   {metrics['gain_vs_baseline']:>+15,.0f} ₽ "
                   f"({metrics['avg_lift']:+.1f}%)")
        logger.info(f"    Достигнуто от идеала: {metrics['efficiency_vs_ideal']:.1f}%")
        
        # Считаем итоги только для работающих сегментов (с положительным приростом)
        if metrics['gain_vs_baseline'] > 0:
            total_gain += metrics['gain_vs_baseline']
            total_baseline += metrics['total_baseline']
            working_segments.append(segment)
        
        # Лучший и худший месяц
        if 'best_month' in metrics:
            logger.info(f"  Лучший месяц: {metrics['best_month']['month']} "
                       f"(lift {metrics['best_month']['lift']:+.1f}%, "
                       f"+{metrics['best_month']['gain']:,.0f} ₽)")
        
        if 'worst_month' in metrics:
            logger.info(f"  Худший месяц: {metrics['worst_month']['month']} "
                       f"(lift {metrics['worst_month']['lift']:+.1f}%, "
                       f"-{metrics['worst_month']['loss']:,.0f} ₽)")
    
    # Итоги по работающим сегментам
    if total_baseline > 0 and working_segments:
        logger.info("\n" + "="*80)
        logger.info(f"ИТОГО по работающим сегментам ({', '.join(working_segments)}):")
        logger.info(f"  Дополнительная выручка: {total_gain:,.0f} ₽")
        logger.info(f"  Общий lift: {total_gain / total_baseline * 100:+.1f}%")
        logger.info("="*80)
    
    return segment_metrics


def log_artifacts_to_mlflow(
    stage_dir: Path,
    log_file: Path,
    segment_metrics: Dict[str, Dict[str, Any]],
    logger: logging.Logger
) -> None:
    """
    Логирует артефакты в MLflow.
    
    Parameters
    ----------
    stage_dir : Path
        Директория этапа с результатами
    log_file : Path
        Путь к файлу логов
    segment_metrics : dict
        Метрики по сегментам
    logger : logging.Logger
        Логгер для вывода информации
    """
    # Логируем метрики в MLflow
    for segment, metrics in segment_metrics.items():
        mlflow.log_metric(f"{segment}_avg_lift", metrics.get('avg_lift', 0))
        mlflow.log_metric(f"{segment}_gain", metrics.get('gain_vs_baseline', 0))
        mlflow.log_metric(f"{segment}_efficiency", metrics.get('efficiency_vs_ideal', 0))
        mlflow.log_metric(f"{segment}_months", metrics.get('months_count', 0))
    
    # Логируем файлы
    features_file = stage_dir / "features_by_segment.json"
    if features_file.exists():
        mlflow.log_artifact(str(features_file))
    else:
        logger.warning(f"Файл {features_file} не найден")
    
    wf_results_file = stage_dir / "walk_forward_results.csv"
    if wf_results_file.exists():
        mlflow.log_artifact(str(wf_results_file))
    else:
        logger.warning(f"Файл {wf_results_file} не найден")
    
    opt_results_file = stage_dir / "optimization_results.pkl"
    if opt_results_file.exists():
        mlflow.log_artifact(str(opt_results_file))
    
    # Логируем лог-файл
    if log_file.exists() and log_file.stat().st_size > 0:
        mlflow.log_artifact(str(log_file))
    else:
        logger.warning(f"Лог-файл {log_file} не найден или пуст")


def log_section_header(logger: logging.Logger, title: str, width: int = 70) -> None:
    """
    Логирует заголовок секции для лучшей читаемости.
    
    Parameters
    ----------
    logger : logging.Logger
        Логгер для вывода
    title : str
        Заголовок секции
    width : int, default=70
        Ширина линии
    """
    logger.info("\n" + "="*width)
    logger.info(title.upper())
    logger.info("="*width)


def log_step_result(
    logger: logging.Logger,
    step: int,
    test_month: str,
    metrics: Dict[str, float],
    prefix: str = "  "
) -> None:
    """
    Логирует результаты одного шага walk-forward валидации.
    
    Parameters
    ----------
    logger : logging.Logger
        Логгер для вывода
    step : int
        Номер шага
    test_month : str
        Тестовый месяц
    metrics : dict
        Метрики шага
    prefix : str, default="  "
        Префикс для отступов
    """
    logger.info(f"\n{prefix}--- Шаг {step}: Test {test_month} ---")
    
    if 'month_lift' in metrics:
        logger.info(f"{prefix}  Месяц {test_month}: общий lift = {metrics['month_lift']:+.1f}%")
    
    # Логируем по сегментам
    for key, value in metrics.items():
        if key.endswith('_lift'):
            segment = key.split('_')[0]
            logger.info(f"{prefix}    {segment}: lift = {value:+.1f}%")
    
    if 'model_sum' in metrics and 'baseline_sum' in metrics:
        logger.info(f"{prefix}    Модель: {metrics['model_sum']:,.0f} ₽, "
                   f"Baseline: {metrics['baseline_sum']:,.0f} ₽")