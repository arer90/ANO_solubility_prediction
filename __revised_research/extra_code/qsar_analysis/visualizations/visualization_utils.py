"""
QSAR Visualization Utilities

This module contains utility functions for visualization management and debugging.
"""

import os
import gc
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import config
try:
    from ..config import PLOT_SETTINGS
except ImportError:
    from config import PLOT_SETTINGS


def safe_savefig(path: Path, dpi: int = 300, **kwargs):
    """안전하게 figure 저장"""
    try:
        # 경로 생성
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 디버깅: 전체 경로 출력
        print(f"      Attempting to save to: {path}")
        
        # 기본 설정
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'facecolor': 'white'
        }
        
        # kwargs로 기본값 덮어쓰기 (중복 방지)
        save_kwargs.update(kwargs)
        
        # 파일 저장
        plt.savefig(path, **save_kwargs)
        print(f"      ✓ Saved: {path.name} at {path.parent}")
        return True
    except Exception as e:
        print(f"      ❌ Failed to save {path.name}: {str(e)}")
        print(f"      Full path was: {path}")
        import traceback
        traceback.print_exc()  # 더 자세한 에러 정보
        return False
    finally:
        plt.close()
        gc.collect()


def safe_to_excel(df: pd.DataFrame, path: Path, **kwargs):
    """안전하게 Excel 저장"""
    try:
        # 경로 생성
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        df.to_excel(path, **kwargs)
        print(f"      ✓ Saved: {path.name}")
        return True
    except Exception as e:
        print(f"      ❌ Failed to save {path.name}: {str(e)}")
        return False


def safe_excel_writer(path: Path, engine='openpyxl'):
    """안전한 ExcelWriter context manager"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return pd.ExcelWriter(path, engine=engine)
    except Exception as e:
        print(f"      ❌ Failed to create ExcelWriter for {path.name}: {str(e)}")
        return None


def check_and_create_paths(output_dir: Path):
    """모든 필요한 경로를 확인하고 생성"""
    paths_to_create = [
        output_dir / 'meta',
        output_dir / 'statistics',
        output_dir / 'statistics' / 'individual_plots',
        output_dir / 'summary',
        output_dir / 'summary' / 'individual_plots',
        output_dir / 'ad_analysis',
        output_dir / 'ad_analysis' / 'ad_performance_analysis',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'random_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'scaffold_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'cluster_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'time_split',
        output_dir / 'ad_analysis' / 'ad_performance_analysis' / 'test_only'
    ]
    
    print("\nCreating/verifying output directories:")
    for path in paths_to_create:
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {path.relative_to(output_dir)}")
    
    return True


def list_output_files(output_dir: Path, extensions: List[str] = ['.png', '.xlsx', '.txt']):
    """출력 디렉토리의 모든 파일 리스트"""
    print(f"\nListing all files in {output_dir}:")
    
    all_files = []
    for ext in extensions:
        files = list(output_dir.rglob(f'*{ext}'))
        all_files.extend(files)
    
    # 디렉토리별로 그룹화
    files_by_dir = {}
    for file in all_files:
        try:
            rel_dir = file.parent.relative_to(output_dir)
        except ValueError:
            # If file is not relative to output_dir, use absolute path
            rel_dir = file.parent
        
        if rel_dir not in files_by_dir:
            files_by_dir[rel_dir] = []
        files_by_dir[rel_dir].append(file.name)
    
    # 출력
    for dir_path, files in sorted(files_by_dir.items()):
        print(f"\n  {dir_path}/")
        for file in sorted(files):
            print(f"    - {file}")
    
    print(f"\nTotal files found: {len(all_files)}")
    return all_files


def verify_output_structure(output_dir: Path):
    """출력 디렉토리 구조 검증"""
    expected_dirs = {
        'meta': ['individual_plots'],
        'statistics': ['individual_plots'],
        'summary': ['individual_plots'],
        'ad_analysis': ['ad_performance_analysis'],
        'ad_analysis/ad_performance_analysis': [
            'random_split', 'scaffold_split', 'cluster_split', 
            'time_split', 'test_only'
        ]
    }
    
    print("\nVerifying output directory structure:")
    all_good = True
    
    for parent, subdirs in expected_dirs.items():
        parent_path = output_dir / parent
        if parent_path.exists():
            print(f"  ✓ {parent}/")
            for subdir in subdirs:
                subdir_path = parent_path / subdir
                if subdir_path.exists():
                    print(f"    ✓ {subdir}/")
                else:
                    print(f"    ❌ {subdir}/ (missing)")
                    all_good = False
        else:
            print(f"  ❌ {parent}/ (missing)")
            all_good = False
    
    return all_good


def create_visualizations_with_debugging(visualization_manager, datasets, splits, 
                                       features, ad_analysis, statistical_results, 
                                       performance_results=None):
    """디버깅 정보와 함께 시각화 생성"""
    print("\n" + "="*60)
    print("STARTING VISUALIZATION CREATION WITH DEBUGGING")
    print("="*60)
    
    # 시작 전 경로 확인
    output_dir = visualization_manager.output_dir
    print(f"\nOutput directory: {output_dir}")
    print(f"Exists: {output_dir.exists()}")
    
    # 입력 데이터 확인
    print("\nInput data check:")
    print(f"  Datasets: {len(datasets) if datasets else 0}")
    print(f"  Splits: {len(splits) if splits else 0}")
    print(f"  Features: {len(features) if features else 0}")
    print(f"  AD Analysis: {len(ad_analysis) if ad_analysis else 0}")
    print(f"  Statistical Results: {len(statistical_results) if statistical_results else 0}")
    print(f"  Performance Results: {len(performance_results) if performance_results else 0}")
    
    # 시각화 생성
    try:
        visualization_manager.create_all_visualizations(
            datasets, splits, features, ad_analysis, 
            statistical_results, performance_results
        )
        success = True
    except Exception as e:
        print(f"\n❌ ERROR during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    # 결과 확인
    print("\n" + "="*60)
    print("VISUALIZATION RESULTS")
    print("="*60)
    
    # 생성된 파일 리스트
    files = list_output_files(output_dir)
    
    # 각 주요 디렉토리 확인
    for subdir in ['meta', 'statistics', 'summary', 'ad_analysis']:
        subpath = output_dir / subdir
        if subpath.exists():
            file_count = len(list(subpath.rglob('*.*')))
            print(f"\n{subdir}: {file_count} files")
            
            # 파일 타입별 개수
            png_count = len(list(subpath.rglob('*.png')))
            xlsx_count = len(list(subpath.rglob('*.xlsx')))
            txt_count = len(list(subpath.rglob('*.txt')))
            
            if png_count > 0:
                print(f"  - PNG files: {png_count}")
            if xlsx_count > 0:
                print(f"  - Excel files: {xlsx_count}")
            if txt_count > 0:
                print(f"  - Text files: {txt_count}")
        else:
            print(f"\n{subdir}: DIRECTORY NOT FOUND")
    
    # 구조 검증
    print("\n" + "-"*60)
    structure_ok = verify_output_structure(output_dir)
    
    if success and structure_ok:
        print("\n✅ Visualization completed successfully!")
    else:
        print("\n⚠️ Visualization completed with issues.")
    
    return files, success


# Export functions
__all__ = [
    'safe_savefig',
    'safe_to_excel',
    'safe_excel_writer',
    'check_and_create_paths',
    'list_output_files',
    'verify_output_structure',
    'create_visualizations_with_debugging'
]