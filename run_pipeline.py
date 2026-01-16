#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Точка входа для запуска полного пайплайна.
Поддерживает как командную строку, так и вызов из Jupyter Notebook.

Пример использования в командной строке:
    python run_pipeline.py --version 3 --verbose 1

Пример использования в Jupyter:
    1) запустить сначала ml flow в терминале
    mlflow ui
    2)в блокноте
    from run_pipeline import run_pipeline_from_notebook
    results = run_pipeline_from_notebook(version=3, verbose=1)
"""

import argparse
import mlflow
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))


from src.pipelines import run_full_pipeline
from src.config import SEGMENT_BINS

warnings.filterwarnings('ignore')


def parse_arguments():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description='Запуск пайплайна прогнозирования выручки')
    
    parser.add_argument('--version', type=int, default=3,
                        help='Версия датасета (default: 3)')
    
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Уровень детализации логов (0-2, default: 1)')
    
    parser.add_argument('--output-dir', type=str, default='pipeline_results',
                        help='Директория для сохранения результатов (default: pipeline_results)')
    
    parser.add_argument('--log-file', type=str, default='pipeline.log',
                        help='Файл для сохранения логов (default: pipeline.log)')
    
    parser.add_argument('--mlflow-uri', type=str, default='http://127.0.0.1:5000',
                        help='URI для MLflow tracking server (default: http://127.0.0.1:5000)')
    
    parser.add_argument('--corr-threshold', type=float, default=0.95,
                        help='Порог корреляции для удаления признаков (default: 0.95)')
    
    parser.add_argument('--stopping-rounds', type=int, default=20,
                        help='Количество шагов без улучшения для остановки (default: 20)')
    
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Количество trials для Optuna (default: 100)')
    
    parser.add_argument('--n-splits', type=int, default=3,
                        help='Количество фолдов для CV (default: 3)')
    
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Отключить MLflow (для запуска без сервера)')
    
    return parser.parse_args()


def run_pipeline_from_notebook(
    version: int = 3,
    verbose: int = 1,
    output_dir: Union[str, Path] = 'pipeline_results',
    log_file: str = 'pipeline.log',
    mlflow_uri: Optional[str] = 'http://127.0.0.1:5000',
    corr_threshold: float = 0.95,
    stopping_rounds: int = 20,
    n_trials: int = 100,
    n_splits: int = 3,
    improvement_threshold: float = 0.0005,
    min_freq_ratio: float = 0.8,
    disable_mlflow: bool = False,
    return_results: bool = True,
    show_progress: bool = True
) -> Dict[str, Any]:
    """
    Функция для запуска пайплайна из Jupyter Notebook.
    
    Parameters
    ----------
    version : int, default=3
        Версия датасета
    verbose : int, default=1
        Уровень детализации логов (0-2)
    output_dir : str or Path, default='pipeline_results'
        Директория для сохранения результатов
    log_file : str, default='pipeline.log'
        Файл для сохранения логов
    mlflow_uri : str, optional
        URI для MLflow tracking server
    corr_threshold : float, default=0.95
        Порог корреляции для удаления признаков
    stopping_rounds : int, default=20
        Количество шагов без улучшения для остановки
    n_trials : int, default=100
        Количество trials для Optuna
    n_splits : int, default=3
        Количество фолдов для CV
    improvement_threshold : float, default=0.0005
        Минимальное улучшение для добавления признака
    min_freq_ratio : float, default=0.8
        Минимальная доля фолдов для стратегии 'frequent'
    disable_mlflow : bool, default=False
        Отключить MLflow
    return_results : bool, default=True
        Возвращать ли результаты (если False, только запускает)
    show_progress : bool, default=True
        Показывать ли прогресс-бары
    
    Returns
    -------
    dict
        Результаты пайплайна (если return_results=True)
    """
    # Настройка параметров
    feature_selection_params = {
        'corr_threshold': corr_threshold,
        'stopping_rounds': stopping_rounds,
        'improvement_threshold': improvement_threshold
    }
    
    hyperopt_params = {
        'n_trials': n_trials,
        'n_splits': n_splits,
        'aggregation_strategy': 'frequent',
        'min_freq_ratio': min_freq_ratio
    }
    
    # Настройка MLflow (если не отключен)
    if not disable_mlflow and mlflow_uri:
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("client_pipeline")
            print(f"✅ MLflow настроен: {mlflow_uri}")
        except Exception as e:
            print(f"⚠️  Ошибка настройки MLflow: {e}")
            print("   Продолжаем без MLflow")
            disable_mlflow = True
    else:
        disable_mlflow = True
    
    # Для Jupyter отключаем прогресс-бары Optuna если нужно
    if not show_progress:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Запуск пайплайна
    print("\n" + "="*70)
    print("🚀 ЗАПУСК ПАЙПЛАЙНА ИЗ JUPYTER NOTEBOOK")
    print("="*70)
    print(f"Версия датасета: {version}")
    print(f"Параметры отбора: corr={corr_threshold}, stopping={stopping_rounds}")
    print(f"Параметры Optuna: trials={n_trials}, splits={n_splits}")
    print("="*70 + "\n")
    
    results = run_full_pipeline(
        version=version,
        bins=SEGMENT_BINS,
        feature_selection_params=feature_selection_params,
        hyperopt_params=hyperopt_params,
        output_dir=output_dir,
        verbose=verbose,
        log_file=log_file,
        mlflow_tracking_uri=None if disable_mlflow else mlflow_uri
    )
    
    if return_results:
        return results
    else:
        return {"status": "completed", "output_dir": output_dir}

def quick_test(
    version: int = 3,
    n_trials: int = 20,
    n_splits: int = 2,
    verbose: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Быстрый запуск пайплайна для тестирования (с уменьшенными параметрами).
    
    Parameters
    ----------
    version : int, default=3
        Версия датасета
    n_trials : int, default=20
        Уменьшенное количество trials
    n_splits : int, default=2
        Уменьшенное количество фолдов
    verbose : int, default=2
        Уровень детализации (по умолчанию максимальный для теста)
    **kwargs : dict
        Дополнительные параметры для run_pipeline_from_notebook
    
    Returns
    -------
    dict
        Результаты пайплайна
    """
    print("🧪 ЗАПУСК БЫСТРОГО ТЕСТА (уменьшенные параметры)")
    
    if 'verbose' in kwargs:
        del kwargs['verbose']
    
    return run_pipeline_from_notebook(
        version=version,
        n_trials=n_trials,
        n_splits=n_splits,
        verbose=verbose,
        **kwargs
    )

def load_saved_results(
    output_dir: Union[str, Path] = 'pipeline_results',
    stage: str = 'final'
) -> Dict[str, Any]:
    """
    Загружает сохраненные результаты пайплайна.
    
    Parameters
    ----------
    output_dir : str or Path, default='pipeline_results'
        Базовая директория с результатами
    stage : str, default='final'
        Этап ('dev' или 'final')
    
    Returns
    -------
    dict
        Загруженные результаты
    """
    import pickle
    import json
    import pandas as pd
    
    output_dir = Path(output_dir) / stage
    results = {}
    
    # Загружаем признаки
    features_file = output_dir / "features_by_segment.json"
    if features_file.exists():
        with open(features_file, 'r') as f:
            results['features_by_segment'] = json.load(f)
        print(f"✅ Загружены признаки из {features_file}")
    
    # Загружаем оптимизацию
    opt_file = output_dir / "optimization_results.pkl"
    if opt_file.exists():
        with open(opt_file, 'rb') as f:
            opt_data = pickle.load(f)
            results['opt_results'] = opt_data.get('results', {})
        print(f"✅ Загружены гиперпараметры из {opt_file}")
    
    # Загружаем walk-forward результаты
    wf_file = output_dir / "walk_forward_results.csv"
    if wf_file.exists():
        results['wf_results'] = pd.read_csv(wf_file)
        print(f"✅ Загружены walk-forward результаты из {wf_file}")
    
    return results


def main():
    """Главная функция запуска пайплайна из командной строки."""
    args = parse_arguments()
    
    # Запуск пайплайна
    results = run_pipeline_from_notebook(
        version=args.version,
        verbose=args.verbose,
        output_dir=args.output_dir,
        log_file=args.log_file,
        mlflow_uri=None if args.no_mlflow else args.mlflow_uri,
        corr_threshold=args.corr_threshold,
        stopping_rounds=args.stopping_rounds,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        disable_mlflow=args.no_mlflow,
        return_results=False
    )
    
    print("\n" + "="*70)
    print("✅ ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())