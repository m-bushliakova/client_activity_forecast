"""
Главный модуль пайплайна, объединяющий все этапы:
- Загрузка и подготовка данных
- Отбор признаков
- Оптимизация гиперпараметров
- Walk-forward валидация
- Финальное обучение
"""
import sys
import warnings
import mlflow
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

sys.path.append(str(Path(__file__).parent.parent))

from src.hyper_opt import run_hyperopt_stage
from src.model import run_validation_stage
from src.logging_utils import setup_logging, log_artifacts_to_mlflow, log_section_header
from src.data_processing import load_and_split_data
from src.feature_selection import run_feature_selection_stage
from src.config import (
    SEGMENT_BINS, SEGMENT_LABELS, 
    get_feature_selection_params, get_hyperopt_params,
    WORKING_SEGMENT_LIFT_THRESHOLD
)

warnings.filterwarnings('ignore')


def setup_stage_directory(
    output_dir: Union[str, Path],
    stage_name: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[Path, Path, logging.Logger]:
    """
    Создает директорию для этапа и настраивает логгер.
    
    Parameters
    ----------
    output_dir : str or Path
        Базовая директория для результатов
    stage_name : str
        Название этапа (dev, final, etc.)
    logger : logging.Logger, optional
        Существующий логгер
    
    Returns
    -------
    stage_dir : Path
        Путь к директории этапа
    log_file : Path
        Путь к файлу логов
    logger : logging.Logger
        Настроенный логгер
    """
    stage_dir = Path(output_dir) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = stage_dir / "pipeline.log"
    
    if logger is None:
        logger = setup_logging(verbose=1, log_file=str(log_file))
    else:
        # Добавляем файловый handler к существующему логгеру
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        logger.info(f"Логи также сохраняются в {log_file}")
    
    return stage_dir, log_file, logger


def run_pipeline_stage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    segment_labels: List[str],
    stage_name: str = 'dev',
    bins: List[float] = SEGMENT_BINS,
    feature_selection_params: Optional[Dict] = None,
    hyperopt_params: Optional[Dict] = None,
    output_dir: Union[str, Path] = 'pipeline_results',
    skip_feature_selection: bool = False,
    skip_hyperopt: bool = False,
    precomputed_features: Optional[Dict[str, List[str]]] = None,
    precomputed_params: Optional[Dict[str, Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Универсальная функция для запуска любого этапа пайплайна.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Обучающая выборка для этого этапа
    val_df : pd.DataFrame
        Валидационная выборка для этого этапа
    segment_labels : List[str]
        Список названий сегментов
    stage_name : str, default='dev'
        Название этапа
    bins : List[float], default=SEGMENT_BINS
        Границы сегментов
    feature_selection_params : dict, optional
        Параметры отбора признаков
    hyperopt_params : dict, optional
        Параметры оптимизации гиперпараметров
    output_dir : str or Path, default='pipeline_results'
        Базовая директория для сохранения результатов
    skip_feature_selection : bool, default=False
        Пропустить ли отбор признаков
    skip_hyperopt : bool, default=False
        Пропустить ли оптимизацию гиперпараметров
    precomputed_features : dict, optional
        Предварительно отобранные признаки (если skip_feature_selection=True)
    precomputed_params : dict, optional
        Предварительно подобранные гиперпараметры (если skip_hyperopt=True)
    logger : logging.Logger, optional
        Логгер
    
    Returns
    -------
    dict
        Результаты этапа:
        - features_by_segment: отобранные признаки
        - opt_results: результаты оптимизации
        - wf_results: результаты walk-forward
        - segment_metrics: бизнес-метрики
    """
    # 1. Подготовка директории и логгера
    stage_dir, log_file, logger = setup_stage_directory(output_dir, stage_name, logger)

    with mlflow.start_run(run_name=f"{stage_name}_pipeline"):
        # Логируем параметры в MLflow
        mlflow.log_params({
            "stage": stage_name,
            "n_segments": len(segment_labels),
            "feature_selection_params": str(feature_selection_params),
            "hyperopt_params": str(hyperopt_params),
            "skip_feature_selection": skip_feature_selection,
            "skip_hyperopt": skip_hyperopt
        })

        log_section_header(logger, f"ЭТАП: {stage_name.upper()}")
        
        # 2. Отбор признаков
        if skip_feature_selection:
            features_by_segment = precomputed_features
            logger.info("\nИспользуем переданные признаки (отбор пропущен)")
        else:
            features_by_segment = run_feature_selection_stage(
                train_df=train_df,
                segment_labels=segment_labels,
                feature_selection_params=feature_selection_params,
                hyperopt_params=hyperopt_params,
                output_dir=stage_dir,
                logger=logger
            )
        
        # 3. Оптимизация гиперпараметров
        if skip_hyperopt:
            opt_results = precomputed_params
            logger.info("\nИспользуем переданные гиперпараметры (оптимизация пропущена)")
        else:
            opt_results = run_hyperopt_stage(
                train_df=train_df,
                features_by_segment=features_by_segment,
                segment_labels=segment_labels,
                hyperopt_params=hyperopt_params,
                output_dir=stage_dir,
                logger=logger
            )
        
        # 4. Валидация и метрики
        wf_results, segment_metrics = run_validation_stage(
            train_df=train_df,
            val_df=val_df,
            features_by_segment=features_by_segment,
            opt_results=opt_results,
            stage_dir=stage_dir,
            logger=logger
        )
        
        # 5. Логирование в MLflow
        log_artifacts_to_mlflow(stage_dir, log_file, segment_metrics, logger)
        
        return {
            'features_by_segment': features_by_segment,
            'opt_results': opt_results,
            'wf_results': wf_results,
            'segment_metrics': segment_metrics,
            'stage_dir': stage_dir
        }


def run_full_pipeline(
    version: int = 3,
    bins: List[float] = SEGMENT_BINS,
    feature_selection_params: Optional[Dict] = None,
    hyperopt_params: Optional[Dict] = None,
    output_dir: Union[str, Path] = 'pipeline_results',
    verbose: int = 1,
    log_file: Union[str, Path] = 'pipeline.log',
    working_lift_threshold: float = WORKING_SEGMENT_LIFT_THRESHOLD,
    mlflow_tracking_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Запускает полный процесс: разработка + финальное обучение.
    
    Этапы:
    1. Загрузка и подготовка данных
    2. Разработка на train + val (отбор признаков, оптимизация, валидация)
    3. Определение работающих сегментов (lift >= threshold)
    4. Финальное обучение на train+val, тест на test
    
    Parameters
    ----------
    version : int, default=3
        Версия датасета
    bins : List[float], default=SEGMENT_BINS
        Границы сегментов
    feature_selection_params : dict, optional
        Параметры отбора признаков
    hyperopt_params : dict, optional
        Параметры оптимизации гиперпараметров
    output_dir : str or Path, default='pipeline_results'
        Директория для сохранения результатов
    verbose : int, default=1
        Уровень детализации логов
    log_file : str or Path, default='pipeline.log'
        Путь к файлу логов
    working_lift_threshold : float, default=10.0
        Порог lift для определения работающего сегмента
    mlflow_tracking_uri : str, optional
        URI для MLflow tracking server
    
    Returns
    -------
    dict
        Результаты пайплайна:
        - dev: результаты этапа разработки
        - final: результаты финального этапа (если есть работающие сегменты)
        - working_segments: список работающих сегментов
    """
    # Настройка MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Настройка логирования
    logger = setup_logging(verbose, log_file)
    
    # Получаем параметры с дефолтными значениями
    if feature_selection_params is None:
        feature_selection_params = get_feature_selection_params()
    
    if hyperopt_params is None:
        hyperopt_params = get_hyperopt_params()
    
    # 1. Загрузка и подготовка данных
    log_section_header(logger, "ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    
    train, val, test = load_and_split_data(
        version=version,
        bins=bins,
        verbose=verbose > 0,
        logger=logger
    )
    
    segment_labels = SEGMENT_LABELS
    
    # 2. Этап разработки (train + val)
    log_section_header(logger, "ЭТАП РАЗРАБОТКИ (TRAIN + VAL)")
    
    dev_results = run_pipeline_stage(
        train_df=train,
        val_df=val,
        segment_labels=segment_labels,
        stage_name='dev',
        bins=bins,
        feature_selection_params=feature_selection_params,
        hyperopt_params=hyperopt_params,
        output_dir=output_dir,
        skip_feature_selection=False,
        skip_hyperopt=False,
        logger=logger
    )
    
    # 3. Определяем работающие сегменты
    working_segments = []
    for segment, metrics in dev_results['segment_metrics'].items():
        avg_lift = metrics.get('avg_lift', 0)
        if avg_lift >= working_lift_threshold:
            working_segments.append(segment)
            logger.info(f"✅ Сегмент {segment}: средний lift {avg_lift:.1f}% (работает)")
        else:
            logger.info(f"❌ Сегмент {segment}: средний lift {avg_lift:.1f}% (не работает)")

    if working_segments:
        logger.info(f"\n✅ Работающие сегменты: {working_segments}")
        logger.info("   Продолжаем с финальным обучением")

        # 4. Финальный этап (train+val -> test)
        log_section_header(logger, "ФИНАЛЬНЫЙ ЭТАП (TRAIN+VAL -> TEST)")
        
        # Объединяем train и val для финального обучения
        train_val = pd.concat([train, val]).reset_index(drop=True)
        
        # Для финального этапа пропускаем оптимизацию, используем параметры из разработки
        final_results = run_pipeline_stage(
            train_df=train_val,
            val_df=test,
            segment_labels=segment_labels,
            stage_name='final',
            bins=bins,
            feature_selection_params=feature_selection_params,
            hyperopt_params=hyperopt_params,
            output_dir=output_dir,
            skip_feature_selection=False,  # Отбор признаков на полных данных
            skip_hyperopt=True,             # Оптимизацию пропускаем
            precomputed_params=dev_results['opt_results'],
            logger=logger
        )
        
        return {
            'dev': dev_results,
            'final': final_results,
            'working_segments': working_segments
        }
    else:
        logger.warning(f"\n⚠️ Нет работающих сегментов (lift >= {working_lift_threshold}%)")
        logger.warning("   Прерываем процесс, финальное обучение не выполняется")
        return {'dev': dev_results, 'working_segments': []}