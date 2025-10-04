"""
Advanced AD Visualization Module
3D plots, decision trees, and interactive visualizations
"""

import numpy as np
import matplotlib
matplotlib.use(\'Agg\')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import warnings

warnings.filterwarnings('ignore')


class AdvancedADVisualizer:
    """
    Advanced visualizations for AD analysis with 3D plots
    """
    
    def __init__(self, output_dir: str = "result/advanced_ad_plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
    
    def create_3d_feature_space(self, name: str, ad_data: Dict, features: Dict, save_path: str):
        """Create 3D feature space visualization"""
        try:
            # Get feature data
            if name not in features or 'features' not in features[name]:
                print(f"    [3D] No feature data for {name}")
                return
            
            X = features[name]['features']
            if X.shape[0] < 3:
                print(f"    [3D] Insufficient data points for 3D plot")
                return
            
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=min(3, X.shape[1]))
            X_3d = pca.fit_transform(X)
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], 
                                X_3d[:, 2] if X_3d.shape[1] > 2 else np.zeros(X_3d.shape[0]),
                                c=np.arange(X_3d.shape[0]), cmap='viridis', 
                                alpha=0.6, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            if X_3d.shape[1] > 2:
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title(f'3D Feature Space - {name}')
            
            plt.colorbar(scatter, label='Sample Index')
            plt.savefig(os.path.join(save_path, '3d_feature_space.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] 3D feature space saved")
        except Exception as e:
            print(f"    [WARNING] 3D feature space failed: {str(e)[:100]}")
    
    def create_3d_ad_boundaries(self, name: str, ad_data: Dict, features: Dict, save_path: str):
        """Create 3D AD boundaries visualization"""
        try:
            # Get first split data
            first_split = next(iter(ad_data.keys())) if ad_data else None
            if not first_split or 'ad_results' not in ad_data[first_split]:
                print(f"    [3D] No AD results for boundaries")
                return
            
            # Get features
            if name not in features or 'features' not in features[name]:
                return
            
            X = features[name]['features']
            if X.shape[0] < 3:
                return
            
            # Use PCA
            pca = PCA(n_components=min(3, X.shape[1]))
            X_3d = pca.fit_transform(X)
            
            # Get AD predictions from first method
            ad_results = ad_data[first_split]['ad_results']
            in_ad = None
            
            # Get test indices to properly map AD results
            test_indices = ad_data[first_split].get('test_indices', [])
            
            for method, result in ad_results.items():
                if result and 'in_ad' in result:
                    in_ad_test = result['in_ad']  # This is only for test samples
                    
                    # Create full array with all samples marked as True (in AD) by default
                    in_ad = np.ones(X_3d.shape[0], dtype=bool)
                    
                    # If we have test indices, map the test AD results properly
                    if len(test_indices) > 0:
                        # Mark test samples based on their AD status
                        for i, test_idx in enumerate(test_indices):
                            if i < len(in_ad_test) and test_idx < len(in_ad):
                                in_ad[test_idx] = in_ad_test[i]
                    else:
                        # If no test indices, assume the in_ad is for all data
                        if len(in_ad_test) <= X_3d.shape[0]:
                            in_ad[:len(in_ad_test)] = in_ad_test
                    break
            
            if in_ad is None:
                return
            
            # Create plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points colored by AD status
            colors = ['green' if x else 'red' for x in in_ad]
            ax.scatter(X_3d[:, 0], X_3d[:, 1],
                      X_3d[:, 2] if X_3d.shape[1] > 2 else np.zeros(X_3d.shape[0]),
                      c=colors, alpha=0.6, s=50)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            if X_3d.shape[1] > 2:
                ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
            ax.set_title(f'3D AD Boundaries - {name}')
            
            # Add legend
            green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Inside AD')
            red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Outside AD')
            ax.legend(handles=[green_patch, red_patch])
            
            plt.savefig(os.path.join(save_path, '3d_ad_boundaries.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] 3D AD boundaries saved")
        except Exception as e:
            print(f"    [WARNING] 3D AD boundaries failed: {str(e)[:100]}")
    
    def create_interactive_3d_plot(self, name: str, ad_data: Dict, features: Dict, save_path: str):
        """Create interactive 3D plot (saved as static image)"""
        try:
            # Get features
            if name not in features or 'descriptors' not in features[name]:
                print(f"    [3D] No descriptor data for interactive plot")
                return
            
            descriptors = features[name]['descriptors']
            if descriptors.shape[0] < 3 or descriptors.shape[1] < 3:
                return
            
            # Use first 3 descriptors
            X_3d = descriptors[:, :3]
            
            # Create multi-view plot
            fig = plt.figure(figsize=(18, 6))
            
            # View 1: XY plane
            ax1 = fig.add_subplot(131)
            ax1.scatter(X_3d[:, 0], X_3d[:, 1], alpha=0.6, c='blue', s=30)
            ax1.set_xlabel('Descriptor 1')
            ax1.set_ylabel('Descriptor 2')
            ax1.set_title('XY Plane View')
            ax1.grid(True, alpha=0.3)
            
            # View 2: XZ plane
            ax2 = fig.add_subplot(132)
            ax2.scatter(X_3d[:, 0], X_3d[:, 2], alpha=0.6, c='green', s=30)
            ax2.set_xlabel('Descriptor 1')
            ax2.set_ylabel('Descriptor 3')
            ax2.set_title('XZ Plane View')
            ax2.grid(True, alpha=0.3)
            
            # View 3: YZ plane
            ax3 = fig.add_subplot(133)
            ax3.scatter(X_3d[:, 1], X_3d[:, 2], alpha=0.6, c='red', s=30)
            ax3.set_xlabel('Descriptor 2')
            ax3.set_ylabel('Descriptor 3')
            ax3.set_title('YZ Plane View')
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'3D Descriptor Space Views - {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path, '3d_interactive_views.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    [OK] 3D interactive views saved")
        except Exception as e:
            print(f"    [WARNING] 3D interactive plot failed: {str(e)[:100]}")
    
    def create_3d_ad_boundary_plot(self, X_train: np.ndarray, X_test: np.ndarray,
                                   ad_results: Dict, dataset_name: str = "Dataset") -> None:
        """
        Create 3D visualization of AD boundaries
        """
        # Reduce to 3D using PCA
        pca = PCA(n_components=3, random_state=42)
        X_train_3d = pca.fit_transform(X_train)
        X_test_3d = pca.transform(X_test)
        
        # Get AD predictions (use first available method)
        in_ad = None
        for method, result in ad_results.items():
            if result and 'in_ad' in result:
                in_ad = result['in_ad']
                break
        
        if in_ad is None:
            print("No AD results available for 3D plot")
            return
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 12))
        
        # Subplot 1: Training vs Test distribution
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2],
                   c='blue', marker='o', alpha=0.3, s=20, label='Training')
        ax1.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2],
                   c='red', marker='^', alpha=0.5, s=30, label='Test')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax1.set_title('Training vs Test Distribution')
        ax1.legend()
        
        # Subplot 2: AD membership
        ax2 = fig.add_subplot(222, projection='3d')
        colors = ['green' if x else 'red' for x in in_ad]
        ax2.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2],
                   c=colors, marker='o', alpha=0.6, s=50)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax2.set_title(f'AD Membership (Coverage: {np.mean(in_ad):.1%})')
        
        # Add convex hull for training data
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(X_train_3d)
            for simplex in hull.simplices:
                triangle = X_train_3d[simplex]
                ax2.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                               alpha=0.1, color='blue')
        except:
            pass  # Skip if convex hull fails
        
        # Subplot 3: Density visualization
        ax3 = fig.add_subplot(223, projection='3d')
        
        # Calculate density using KDE
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(X_train_3d.T)
            density_train = kde(X_train_3d.T)
            density_test = kde(X_test_3d.T)
            
            # Plot with density coloring
            scatter = ax3.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2],
                                c=density_test, cmap='viridis', s=50, alpha=0.7)
            plt.colorbar(scatter, ax=ax3, label='Density')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.set_zlabel('PC3')
            ax3.set_title('Test Set Density in Training Space')
        except:
            ax3.text(0.5, 0.5, 0.5, 'Density calculation failed', 
                    horizontalalignment='center')
        
        # Subplot 4: Distance from centroid
        ax4 = fig.add_subplot(224, projection='3d')
        
        # Calculate distances from training centroid
        centroid = np.mean(X_train_3d, axis=0)
        distances = np.linalg.norm(X_test_3d - centroid, axis=1)
        
        scatter2 = ax4.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2],
                             c=distances, cmap='coolwarm', s=50, alpha=0.7)
        
        # Add centroid
        ax4.scatter(centroid[0], centroid[1], centroid[2],
                   c='black', marker='*', s=500, label='Centroid')
        
        plt.colorbar(scatter2, ax=ax4, label='Distance from Centroid')
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC2')
        ax4.set_zlabel('PC3')
        ax4.set_title('Distance from Training Centroid')
        ax4.legend()
        
        plt.suptitle(f'{dataset_name}: 3D AD Boundary Visualization', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f'{dataset_name}_3d_ad_boundary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      [OK] 3D AD boundary plot saved: {output_path.name}")
    
    def create_ad_decision_tree(self, X_train: np.ndarray, X_test: np.ndarray,
                               ad_results: Dict, dataset_name: str = "Dataset") -> None:
        """
        Create decision tree visualization for AD boundaries
        """
        # Get AD predictions
        in_ad = None
        for method, result in ad_results.items():
            if result and 'in_ad' in result:
                in_ad = result['in_ad']
                break
        
        if in_ad is None:
            print("No AD results available for decision tree")
            return
        
        # Combine train (all in AD) and test data
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([np.ones(len(X_train)), np.array(in_ad).astype(int)])
        
        # Limit features for visualization (use PCA if too many)
        if X_combined.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            X_combined = pca.fit_transform(X_combined)
            feature_names = [f'PC{i+1}' for i in range(10)]
        else:
            feature_names = [f'Feature_{i+1}' for i in range(X_combined.shape[1])]
        
        # Train decision tree
        dt = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        dt.fit(X_combined, y_combined)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Subplot 1: Decision tree
        ax1 = axes[0, 0]
        plot_tree(dt, feature_names=feature_names, class_names=['Outside AD', 'Inside AD'],
                 filled=True, rounded=True, fontsize=8, ax=ax1)
        ax1.set_title('AD Decision Tree', fontsize=14)
        
        # Subplot 2: Feature importance
        ax2 = axes[0, 1]
        importances = dt.feature_importances_
        indices = np.argsort(importances)[::-1][:5]  # Top 5
        
        ax2.barh(range(len(indices)), importances[indices], color='steelblue')
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels([feature_names[i] for i in indices])
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 5 Important Features for AD')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Decision path visualization
        ax3 = axes[1, 0]
        
        # Get decision paths for a sample of test points
        n_samples_viz = min(100, len(X_test))
        sample_indices = np.random.choice(len(X_test), n_samples_viz, replace=False)
        
        decision_paths = dt.decision_path(X_test[sample_indices])
        
        # Visualize paths as heatmap
        path_matrix = decision_paths.toarray()
        im = ax3.imshow(path_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Tree Node')
        ax3.set_title('Decision Paths Through Tree')
        plt.colorbar(im, ax=ax3, label='Path (1=visited, 0=not visited)')
        
        # Subplot 4: AD boundary regions
        ax4 = axes[1, 1]
        
        # Use first two principal components for 2D visualization
        if X_train.shape[1] > 2:
            pca2 = PCA(n_components=2, random_state=42)
            X_train_2d = pca2.fit_transform(X_train)
            X_test_2d = pca2.transform(X_test)
        else:
            X_train_2d = X_train[:, :2]
            X_test_2d = X_test[:, :2]
        
        # Create mesh for decision boundary
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        # Plot decision regions
        if X_train.shape[1] > 2:
            # Need to transform mesh points to original space for prediction
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            # Pad with zeros for missing components
            mesh_points_padded = np.hstack([mesh_points, 
                                           np.zeros((len(mesh_points), 8))])
            Z = dt.predict(mesh_points_padded)
        else:
            Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        ax4.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
        ax4.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c='blue', 
                   marker='o', s=20, alpha=0.5, label='Training')
        
        # Color test points by AD membership
        colors = ['green' if x else 'red' for x in in_ad]
        ax4.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=colors,
                   marker='^', s=40, alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('PC1' if X_train.shape[1] > 2 else 'Feature 1')
        ax4.set_ylabel('PC2' if X_train.shape[1] > 2 else 'Feature 2')
        ax4.set_title('AD Decision Boundaries (2D Projection)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset_name}: AD Decision Tree Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f'{dataset_name}_ad_decision_tree.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      [OK] AD decision tree saved: {output_path.name}")
    
    def create_interactive_ad_dashboard(self, X_train: np.ndarray, X_test: np.ndarray,
                                       ad_results: Dict, dataset_name: str = "Dataset") -> None:
        """
        Create interactive Plotly dashboard for AD exploration
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("      [WARNING] Plotly not available, skipping interactive dashboard")
            return
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=3, random_state=42)
        X_train_3d = pca.fit_transform(X_train)
        X_test_3d = pca.transform(X_test)
        
        # Get AD predictions from multiple methods
        ad_methods = {}
        for method, result in ad_results.items():
            if result and 'in_ad' in result:
                ad_methods[method] = result['in_ad']
        
        if not ad_methods:
            print("No AD results available for dashboard")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('3D AD Visualization', 'Method Agreement', 
                          'Coverage by Method', 'Distance Distribution'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                  [{'type': 'bar'}, {'type': 'histogram'}]],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. 3D scatter plot
        # Use first method for coloring
        first_method = list(ad_methods.keys())[0]
        colors = ['green' if x else 'red' for x in ad_methods[first_method]]
        
        fig.add_trace(
            go.Scatter3d(
                x=X_test_3d[:, 0],
                y=X_test_3d[:, 1],
                z=X_test_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=ad_methods[first_method],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="In AD", x=0.45, len=0.4)
                ),
                text=[f'Sample {i}<br>In AD: {v}' for i, v in enumerate(ad_methods[first_method])],
                hovertemplate='%{text}<extra></extra>',
                name='Test Samples'
            ),
            row=1, col=1
        )
        
        # Add training data outline
        fig.add_trace(
            go.Scatter3d(
                x=X_train_3d[:, 0],
                y=X_train_3d[:, 1],
                z=X_train_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightblue',
                    opacity=0.3
                ),
                name='Training',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 2. Method agreement heatmap (using 2D scatter as proxy)
        if len(ad_methods) > 1:
            agreement = np.mean(list(ad_methods.values()), axis=0)
            
            # Use t-SNE for 2D projection
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_test)-1))
            X_test_2d = tsne.fit_transform(X_test)
            
            fig.add_trace(
                go.Scatter(
                    x=X_test_2d[:, 0],
                    y=X_test_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=agreement,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Agreement", x=1.0, len=0.4)
                    ),
                    text=[f'Agreement: {a:.2f}' for a in agreement],
                    hovertemplate='%{text}<extra></extra>',
                    name='Method Agreement'
                ),
                row=1, col=2
            )
        
        # 3. Coverage by method
        coverages = {method: np.mean(in_ad) for method, in_ad in ad_methods.items()}
        
        fig.add_trace(
            go.Bar(
                x=list(coverages.keys()),
                y=list(coverages.values()),
                text=[f'{v:.1%}' for v in coverages.values()],
                textposition='outside',
                marker_color=['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' 
                            for v in coverages.values()],
                name='Coverage'
            ),
            row=2, col=1
        )
        
        # 4. Distance distribution
        # Calculate distances from centroid
        centroid = np.mean(X_train_3d, axis=0)
        distances_train = np.linalg.norm(X_train_3d - centroid, axis=1)
        distances_test = np.linalg.norm(X_test_3d - centroid, axis=1)
        
        fig.add_trace(
            go.Histogram(
                x=distances_train,
                name='Training',
                opacity=0.5,
                marker_color='blue',
                nbinsx=30
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=distances_test,
                name='Test',
                opacity=0.5,
                marker_color='red',
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{dataset_name}: Interactive AD Dashboard',
            showlegend=True,
            height=800,
            width=1200,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Coverage", row=2, col=1)
        fig.update_xaxes(title_text="Distance from Centroid", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        # Save as HTML
        output_path = self.output_dir / f'{dataset_name}_interactive_dashboard.html'
        fig.write_html(str(output_path))
        
        print(f"      [OK] Interactive dashboard saved: {output_path.name}")
    
    def create_comprehensive_ad_report(self, X_train: np.ndarray, X_test: np.ndarray,
                                      ad_results: Dict, dataset_name: str = "Dataset") -> None:
        """
        Create comprehensive AD analysis report with all visualizations
        """
        print(f"\n    Creating comprehensive AD visualizations for {dataset_name}...")
        
        # Create all visualizations
        self.create_3d_ad_boundary_plot(X_train, X_test, ad_results, dataset_name)
        self.create_ad_decision_tree(X_train, X_test, ad_results, dataset_name)
        self.create_interactive_ad_dashboard(X_train, X_test, ad_results, dataset_name)
        
        # Create summary figure
        self._create_ad_summary_figure(X_train, X_test, ad_results, dataset_name)
        
        print(f"    [OK] All AD visualizations completed for {dataset_name}")
    
    def _create_ad_summary_figure(self, X_train: np.ndarray, X_test: np.ndarray,
                                 ad_results: Dict, dataset_name: str) -> None:
        """
        Create summary figure combining key AD metrics
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Coverage comparison
        ax1 = axes[0, 0]
        coverages = {}
        for method, result in ad_results.items():
            if result and 'coverage' in result:
                coverages[method] = result['coverage']
        
        if coverages:
            methods = list(coverages.keys())
            values = list(coverages.values())
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
            
            bars = ax1.bar(range(len(methods)), values, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_ylabel('Coverage')
            ax1.set_title('AD Coverage by Method')
            ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
            ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.1%}', ha='center', va='bottom')
        
        # 2. Feature distributions
        ax2 = axes[0, 1]
        n_features = min(5, X_train.shape[1])
        for i in range(n_features):
            ax2.hist(X_train[:, i], alpha=0.3, bins=20, density=True, label=f'F{i+1} Train')
            ax2.hist(X_test[:, i], alpha=0.3, bins=20, density=True, 
                    linestyle='--', histtype='step', linewidth=2)
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Feature Distributions (Top 5)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample complexity
        ax3 = axes[0, 2]
        train_sizes = [100, 500, 1000, len(X_train)]
        train_sizes = [s for s in train_sizes if s <= len(X_train)]
        
        complexity_scores = []
        for size in train_sizes:
            if size > 0:
                subset = X_train[:size]
                # Simple complexity measure: log(n) * sqrt(d)
                complexity = np.log(size) * np.sqrt(X_train.shape[1])
                complexity_scores.append(complexity)
        
        ax3.plot(train_sizes, complexity_scores, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('Complexity Score')
        ax3.set_title('AD Complexity vs Training Size')
        ax3.grid(True, alpha=0.3)
        
        # 4. Method agreement matrix
        ax4 = axes[1, 0]
        method_names = []
        ad_matrix = []
        
        for method, result in ad_results.items():
            if result and 'in_ad' in result:
                method_names.append(method)
                ad_matrix.append(result['in_ad'])
        
        if len(ad_matrix) > 1:
            ad_matrix = np.array(ad_matrix)
            correlation = np.corrcoef(ad_matrix)
            
            im = ax4.imshow(correlation, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax4.set_xticks(range(len(method_names)))
            ax4.set_yticks(range(len(method_names)))
            ax4.set_xticklabels(method_names, rotation=45, ha='right')
            ax4.set_yticklabels(method_names)
            ax4.set_title('Method Agreement Matrix')
            
            # Add correlation values
            for i in range(len(method_names)):
                for j in range(len(method_names)):
                    ax4.text(j, i, f'{correlation[i, j]:.2f}',
                           ha='center', va='center', color='black' if correlation[i, j] > 0.5 else 'white')
            
            plt.colorbar(im, ax=ax4, label='Correlation')
        else:
            ax4.text(0.5, 0.5, 'Not enough methods for correlation', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Method Agreement Matrix')
        
        # 5. Confidence distribution
        ax5 = axes[1, 1]
        if 'ensemble' in ad_results and 'confidence' in ad_results['ensemble']:
            confidence = ad_results['ensemble']['confidence']
        elif 'reliability_index' in ad_results and 'confidence_scores' in ad_results['reliability_index']:
            confidence = ad_results['reliability_index']['confidence_scores']
        else:
            # Calculate simple confidence as method agreement
            if len(ad_matrix) > 1:
                confidence = np.mean(ad_matrix, axis=0)
            else:
                confidence = None
        
        if confidence is not None:
            ax5.hist(confidence, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax5.axvline(np.mean(confidence), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidence):.2f}')
            ax5.set_xlabel('Confidence Score')
            ax5.set_ylabel('Frequency')
            ax5.set_title('AD Confidence Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        summary_text = f"Dataset: {dataset_name}\n\n"
        summary_text += f"Training samples: {len(X_train)}\n"
        summary_text += f"Test samples: {len(X_test)}\n"
        summary_text += f"Features: {X_train.shape[1]}\n\n"
        
        summary_text += "Coverage Summary:\n"
        for method, coverage in list(coverages.items())[:5]:
            summary_text += f"  {method}: {coverage:.1%}\n"
        
        if len(coverages) > 0:
            summary_text += f"\nAverage coverage: {np.mean(list(coverages.values())):.1%}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))
        ax6.axis('off')
        ax6.set_title('Summary Statistics')
        
        plt.suptitle(f'{dataset_name}: AD Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f'{dataset_name}_ad_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      [OK] AD summary figure saved: {output_path.name}")