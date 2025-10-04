"""
QSAR Metrics Module - 2025 Update
Focuses on RMSE differences instead of ratios for inner/outer sets
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class QSARMetrics:
    """Calculate and evaluate QSAR model metrics with focus on differences"""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_rmse_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              set_type: str = 'test') -> Dict:
        """Calculate RMSE and related metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        return {
            'set_type': set_type,
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mse': float(mse),
            'residuals': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            },
            'n_samples': len(y_true)
        }
    
    def evaluate_inner_outer_difference(self, inner_metrics: Dict, 
                                      outer_metrics: Dict) -> Dict:
        """
        Evaluate the difference between inner and outer set performance
        2025 Update: Focus on differences instead of ratios
        """
        inner_rmse = inner_metrics['rmse']
        outer_rmse = outer_metrics['rmse']
        
        # Calculate difference (outer - inner)
        rmse_difference = outer_rmse - inner_rmse
        mae_difference = outer_metrics['mae'] - inner_metrics['mae']
        r2_difference = outer_metrics['r2'] - inner_metrics['r2']
        
        # Calculate relative difference (percentage)
        rmse_relative_diff = (rmse_difference / inner_rmse * 100) if inner_rmse > 0 else 0
        
        # Evaluate quality based on difference
        quality = self._evaluate_difference_quality(rmse_difference, rmse_relative_diff)
        
        evaluation = {
            'inner_rmse': float(inner_rmse),
            'outer_rmse': float(outer_rmse),
            'rmse_difference': float(rmse_difference),
            'rmse_relative_difference': float(rmse_relative_diff),
            'mae_difference': float(mae_difference),
            'r2_difference': float(r2_difference),
            'quality': quality,
            'evaluation': self._get_difference_evaluation(rmse_difference, rmse_relative_diff)
        }
        
        # Add warnings if difference is significant
        performance_warnings = []
        if rmse_difference > 0.5:  # More than 0.5 log units difference
            performance_warnings.append("Large RMSE difference (>0.5 log units) - model may be overfitting")
        if rmse_relative_diff > 30:  # More than 30% relative difference
            performance_warnings.append("High relative difference (>30%) - poor generalization")
        if r2_difference < -0.2:  # R² drops by more than 0.2
            performance_warnings.append("Significant R² drop - model performance degradation")
        
        evaluation['warnings'] = performance_warnings
        
        return evaluation
    
    def _evaluate_difference_quality(self, rmse_diff: float, relative_diff: float) -> str:
        """Evaluate the quality based on RMSE difference"""
        if rmse_diff < 0:
            return "Excellent (outer better than inner)"
        elif rmse_diff < 0.1 and relative_diff < 10:
            return "Excellent"
        elif rmse_diff < 0.2 and relative_diff < 20:
            return "Good"
        elif rmse_diff < 0.3 and relative_diff < 30:
            return "Acceptable"
        elif rmse_diff < 0.5 and relative_diff < 50:
            return "Poor"
        else:
            return "Very Poor"
    
    def _get_difference_evaluation(self, rmse_diff: float, relative_diff: float) -> str:
        """Get detailed evaluation of the difference"""
        if rmse_diff < 0:
            return f"Outer set performs better than inner set by {abs(rmse_diff):.3f} log units"
        elif rmse_diff < 0.1:
            return f"Minimal difference ({rmse_diff:.3f} log units, {relative_diff:.1f}%)"
        elif rmse_diff < 0.3:
            return f"Moderate difference ({rmse_diff:.3f} log units, {relative_diff:.1f}%)"
        else:
            return f"Large difference ({rmse_diff:.3f} log units, {relative_diff:.1f}%)"
    
    def calculate_split_metrics(self, predictions_by_split: Dict) -> Dict:
        """Calculate metrics for each split type"""
        split_metrics = {}
        
        for split_name, split_data in predictions_by_split.items():
            if 'y_true' in split_data and 'y_pred' in split_data:
                metrics = self.calculate_rmse_metrics(
                    split_data['y_true'], 
                    split_data['y_pred'],
                    set_type=f"{split_name}_test"
                )
                
                # If inner set predictions available
                if 'y_true_inner' in split_data and 'y_pred_inner' in split_data:
                    inner_metrics = self.calculate_rmse_metrics(
                        split_data['y_true_inner'],
                        split_data['y_pred_inner'],
                        set_type=f"{split_name}_inner"
                    )
                    
                    # Calculate difference
                    difference_eval = self.evaluate_inner_outer_difference(
                        inner_metrics, metrics
                    )
                    
                    split_metrics[split_name] = {
                        'inner': inner_metrics,
                        'outer': metrics,
                        'difference_evaluation': difference_eval
                    }
                else:
                    split_metrics[split_name] = {
                        'outer': metrics
                    }
        
        return split_metrics
    
    def generate_metrics_report(self, split_metrics: Dict) -> str:
        """Generate a text report of metrics focusing on differences"""
        report = []
        report.append("="*60)
        report.append("QSAR METRICS REPORT - RMSE DIFFERENCE ANALYSIS")
        report.append("="*60)
        report.append("")
        
        for split_name, metrics in split_metrics.items():
            report.append(f"\n{split_name.upper()} SPLIT:")
            report.append("-"*40)
            
            if 'outer' in metrics:
                outer = metrics['outer']
                report.append(f"Outer Set (n={outer['n_samples']}):")
                report.append(f"  RMSE: {outer['rmse']:.3f}")
                report.append(f"  MAE:  {outer['mae']:.3f}")
                report.append(f"  R²:   {outer['r2']:.3f}")
            
            if 'inner' in metrics:
                inner = metrics['inner']
                report.append(f"\nInner Set (n={inner['n_samples']}):")
                report.append(f"  RMSE: {inner['rmse']:.3f}")
                report.append(f"  MAE:  {inner['mae']:.3f}")
                report.append(f"  R²:   {inner['r2']:.3f}")
            
            if 'difference_evaluation' in metrics:
                diff = metrics['difference_evaluation']
                report.append(f"\nDifference Analysis:")
                report.append(f"  RMSE Difference: {diff['rmse_difference']:.3f} ({diff['rmse_relative_difference']:.1f}%)")
                report.append(f"  Quality: {diff['quality']}")
                report.append(f"  {diff['evaluation']}")
                
                if diff['warnings']:
                    report.append("\n  [WARNING]  Warnings:")
                    for warning in diff['warnings']:
                        report.append(f"    - {warning}")
        
        report.append("\n" + "="*60)
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate inner set (training) predictions - better performance
    y_true_inner = np.random.randn(n_samples)
    y_pred_inner = y_true_inner + np.random.randn(n_samples) * 0.3
    
    # Simulate outer set (test) predictions - worse performance
    y_true_outer = np.random.randn(n_samples)
    y_pred_outer = y_true_outer + np.random.randn(n_samples) * 0.5
    
    # Calculate metrics
    metrics_calc = QSARMetrics()
    
    inner_metrics = metrics_calc.calculate_rmse_metrics(y_true_inner, y_pred_inner, 'inner')
    outer_metrics = metrics_calc.calculate_rmse_metrics(y_true_outer, y_pred_outer, 'outer')
    
    # Evaluate difference
    diff_eval = metrics_calc.evaluate_inner_outer_difference(inner_metrics, outer_metrics)
    
    print("Inner RMSE:", inner_metrics['rmse'])
    print("Outer RMSE:", outer_metrics['rmse'])
    print("RMSE Difference:", diff_eval['rmse_difference'])
    print("Quality:", diff_eval['quality'])
    print("Evaluation:", diff_eval['evaluation'])