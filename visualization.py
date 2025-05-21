import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import os
import scipy.sparse as sp
from scipy.stats import kendalltau
import glob
from sklearn.manifold import TSNE

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

class GNNRankVisualizer:
    """Visualization utilities for GNNRank models and results."""
    
    def __init__(self, dataset_name="complicated_test"):
        """
        Initialize the visualizer with a dataset name.
        
        Args:
            dataset_name (str): Name of the dataset to visualize.
        """
        self.dataset_name = dataset_name
        self.data_dir = os.path.join('data', dataset_name)
        self.result_dir = os.path.join('result_arrays', dataset_name)
        
        # Add existence checks
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        if not os.path.exists(self.result_dir):
            raise FileNotFoundError(f"Result directory {self.result_dir} not found")
        
        # Load adjacency matrix if available
        adj_path = os.path.join(self.data_dir, 'adj.npz')
        if os.path.exists(adj_path):
            self.adj_matrix = sp.load_npz(adj_path).toarray()
            print(f"Loaded adjacency matrix with shape {self.adj_matrix.shape}")
        else:
            self.adj_matrix = None
            print(f"Warning: Adjacency matrix not found at {adj_path}")
            
        # Load ground truth labels if available
        labels_path = os.path.join(self.data_dir, 'labels.npy')
        if os.path.exists(labels_path):
            self.true_labels = np.load(labels_path)
            print(f"Loaded ground truth labels for {len(self.true_labels)} nodes")
        else:
            self.true_labels = None
            print(f"Warning: Ground truth labels not found at {labels_path}")
    
    def load_predictions(self, method_name, trial_idx=0):
        """
        Load model predictions for a specific method and trial.
        
        Args:
            method_name (str): Name of the method (e.g., 'DIGRAC_dist').
            trial_idx (int): Index of the trial to load.
            
        Returns:
            np.ndarray: Predicted rankings or None if not found.
        """
        # Try to find prediction files in logs directory
        # Path: logs/[dataset_name]/[SeedX]/[Timestamp]/[method_name]_pred[trial_idx].npy
        seed_dirs = glob.glob(os.path.join("logs", self.dataset_name, "Seed*")) # Get all SeedX directories
        
        for seed_dir in seed_dirs:
            timestamp_dirs = glob.glob(os.path.join(seed_dir, "*")) # Get all Timestamp subdirectories
            for ts_dir in timestamp_dirs:
                pred_file = os.path.join(ts_dir, f"{method_name}_pred{trial_idx}.npy")
                if os.path.exists(pred_file):
                    pred_ranks = np.load(pred_file)
                    print(f"Loaded predictions from {pred_file}")
                    return pred_ranks
                
        print(f"Warning: Could not find predictions for {method_name}, trial {trial_idx} in logs")
        return None
            
    def _find_metric_file(self, metric_type, method):
        """
        Find the first .npy file for a given metric type and method.
        Args:
            metric_type (str): 'kendalltau' or 'upset'
            method (str): method subfolder name (e.g., 'DIGRAC')
        Returns:
            str or None: Path to the .npy file, or None if not found.
        """
        metric_dir = os.path.join('result_arrays', self.dataset_name, metric_type, method)
        if not os.path.isdir(metric_dir):
            print(f"Warning: Directory {metric_dir} does not exist.")
            return None
        npy_files = [f for f in os.listdir(metric_dir) if f.endswith('.npy')]
        if not npy_files:
            print(f"Warning: No .npy files found in {metric_dir}")
            return None
        return os.path.join(metric_dir, npy_files[0])

    def compare_methods_bar_chart(self, methods, metric='kendall_tau'):
        """
        Create a bar chart comparing different methods based on a metric.
        Args:
            methods (list): List of method subfolder names to compare (e.g., ['DIGRAC']).
            metric (str): Metric to compare ('kendall_tau', 'upset').
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        metrics_data = []
        if not os.path.exists(self.result_dir):
            print(f"Error: Result directory {self.result_dir} not found. Run training first.")
            return None
        # Map metric to subfolder
        metric_type = 'kendalltau' if 'kendall' in metric else 'upset'
        for method in methods:
            metric_file = self._find_metric_file(metric_type, method)
            if not metric_file:
                continue
            arr = np.load(metric_file)
            # For kendalltau: arr shape (5, 2, 3, 2) [variant, trial, split, (tau, p)]
            # For upset: arr shape (5, 2, 3) [variant, trial, split]
            # We'll take the mean over all axes for a summary, or you can customize
            if metric_type == 'kendalltau':
                # Take only tau values (last dim index 0)
                tau_vals = arr[..., 0]
                mean_val = np.nanmean(tau_vals)
                std_val = np.nanstd(tau_vals)
                metrics_data.append({'method': method, 'mean': mean_val, 'std': std_val, 'values': tau_vals.flatten()})
            else:
                mean_val = np.nanmean(arr)
                std_val = np.nanstd(arr)
                metrics_data.append({'method': method, 'mean': mean_val, 'std': std_val, 'values': arr.flatten()})
        if not metrics_data:
            print(f"Error: No {metric} data found for any methods")
            return None
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        methods = [d['method'] for d in metrics_data]
        means = [d['mean'] for d in metrics_data]
        stds = [d['std'] for d in metrics_data]
        bars = ax.bar(methods, means, yerr=stds, capsize=10, alpha=0.7)
        metric_label = metric.replace('_', ' ').title()
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} Comparison for {self.dataset_name}")
        plt.xticks(rotation=45, ha='right')
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        return fig
        
    def plot_rank_comparison(self, method_name, trial_idx=0, top_n=None):
        """
        Plot a comparison between predicted and true rankings.
        
        Args:
            method_name (str): Name of the method to visualize.
            trial_idx (int): Trial index to use.
            top_n (int, optional): Show only the top N nodes. If None, show all.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.true_labels is None:
            print("Error: Ground truth labels not available for this dataset")
            return None
            
        pred_ranks = self.load_predictions(method_name, trial_idx)
        if pred_ranks is None:
            return None
            
        # Create a DataFrame with both true and predicted ranks
        df = pd.DataFrame({
            'Node': np.arange(len(self.true_labels)),
            'True Rank': self.true_labels,
            'Predicted Rank': pred_ranks
        })
        
        # Sort by true rank
        df = df.sort_values('True Rank')
        
        # Limit to top_n if specified
        if top_n is not None and top_n < len(df):
            df = df.head(top_n)
            
        # Calculate Kendall's tau for this specific prediction
        tau, p_value = kendalltau(df['True Rank'], df['Predicted Rank'])
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            df['True Rank'], 
            df['Predicted Rank'],
            alpha=0.7,
            s=80
        )
        
        # Add node IDs as labels if there aren't too many
        if len(df) <= 50:
            for i, row in df.iterrows():
                ax.annotate(
                    str(row['Node']),
                    (row['True Rank'], row['Predicted Rank']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        # Add a diagonal line (perfect prediction)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        # Add labels and title
        ax.set_xlabel('True Rank')
        ax.set_ylabel('Predicted Rank')
        ax.set_title(f'Rank Comparison for {method_name} (Kendall\'s τ: {tau:.3f})')
        
        plt.tight_layout()
        return fig
    
    def visualize_network(self, method_name=None, trial_idx=0, highlight_upsets=True,
                         layout='spring', node_size=300):
        """
        Visualize the network with optional highlighting of prediction upsets.
        
        Args:
            method_name (str, optional): Method name to load predictions from.
            trial_idx (int): Trial index to use.
            highlight_upsets (bool): Whether to highlight edges that represent upsets.
            layout (str): NetworkX layout algorithm to use.
            node_size (int): Size of nodes in the visualization.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        if self.adj_matrix is None:
            print("Error: Adjacency matrix not available")
            return None
            
        # Create directed graph from adjacency matrix
        G = nx.DiGraph()
        n_nodes = self.adj_matrix.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        # Add edges
        for i in range(n_nodes):
            for j in range(n_nodes):
                if self.adj_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=self.adj_matrix[i, j])
        
        # Prepare node colors based on true or predicted ranks
        if method_name is not None:
            pred_ranks = self.load_predictions(method_name, trial_idx)
            if pred_ranks is not None:
                # Normalize ranks to [0, 1] for coloring
                node_ranks = pred_ranks / np.max(pred_ranks)
                node_colors = plt.cm.viridis(node_ranks)
                title = f"Network for {self.dataset_name} with {method_name} Rankings"
            else:
                node_colors = 'skyblue'
                title = f"Network for {self.dataset_name}"
        elif self.true_labels is not None:
            # Use true labels for coloring
            node_ranks = self.true_labels / np.max(self.true_labels)
            node_colors = plt.cm.viridis(node_ranks)
            title = f"Network for {self.dataset_name} with Ground Truth Rankings"
        else:
            node_colors = 'skyblue'
            title = f"Network for {self.dataset_name}"
            
        # Determine edge colors to highlight upsets
        edge_colors = []
        edge_widths = []
        
        if highlight_upsets and method_name is not None and pred_ranks is not None:
            for u, v in G.edges():
                # If u->v but pred_rank[u] > pred_rank[v], it's an upset
                if pred_ranks[u] > pred_ranks[v]:
                    edge_colors.append('red')
                    edge_widths.append(2.0)
                else:
                    edge_colors.append('black')
                    edge_widths.append(0.5)
        else:
            edge_colors = 'black'
            edge_widths = 0.5
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
            
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                              alpha=0.6, arrowsize=15, ax=ax)
        
        # Add node labels if there aren't too many
        if n_nodes <= 50:
            nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
            
        # Add title and remove axis
        plt.title(title)
        plt.axis('off')
        
        # Add a colorbar legend if using ranks for coloring
        if isinstance(node_colors, np.ndarray) and node_colors.ndim > 1:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label('Rank (lower is better)')
            
        # Add a legend for upset edges
        if highlight_upsets and method_name is not None and pred_ranks is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', lw=2, label='Upset'),
                Line2D([0], [0], color='black', lw=0.5, label='Expected')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
        plt.tight_layout()
        return fig
    
    def plot_embeddings_2d(self, method_name, trial_idx=0):
        """
        Plot the node embeddings in 2D using t-SNE.
        Note: This requires embeddings to be saved during training.
        
        Args:
            method_name (str): Method name to load embeddings from.
            trial_idx (int): Trial index to use.
            
        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Try to find embeddings in logs directory
        embedding_file = None
        log_dirs = glob.glob(f"logs/{self.dataset_name}/*")
        
        for log_dir in log_dirs:
            emb_file = os.path.join(log_dir, f"{method_name}_embeddings{trial_idx}.npy")
            if os.path.exists(emb_file):
                embedding_file = emb_file
                break
                
        if embedding_file is None:
            print(f"Warning: No embeddings found for {method_name}, trial {trial_idx}")
            return None
            
        # Load embeddings
        embeddings = np.load(embedding_file)
        print(f"Loaded embeddings with shape {embeddings.shape} from {embedding_file}")
        
        # Use t-SNE to project embeddings to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create dataframe
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'node': np.arange(len(embeddings))
        })
        
        # Add rank information if available
        if self.true_labels is not None:
            df['true_rank'] = self.true_labels
            color_col = 'true_rank'
            color_label = 'True Rank'
        else:
            pred_ranks = self.load_predictions(method_name, trial_idx)
            if pred_ranks is not None:
                df['pred_rank'] = pred_ranks
                color_col = 'pred_rank'
                color_label = 'Predicted Rank'
            else:
                color_col = None
                color_label = None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_col is not None:
            scatter = ax.scatter(
                df['x'], df['y'],
                c=df[color_col], cmap='viridis',
                alpha=0.8, s=100
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_label)
        else:
            ax.scatter(df['x'], df['y'], alpha=0.8, s=100)
            
        # Add node IDs as labels if there aren't too many
        if len(df) <= 50:
            for i, row in df.iterrows():
                ax.annotate(
                    str(row['node']),
                    (row['x'], row['y']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        ax.set_title(f'2D t-SNE Projection of Node Embeddings for {method_name}')
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize visualizer with dataset name
    viz = GNNRankVisualizer("complicated_test")
    
    # Compare different methods
    methods = ["DIGRAC"]
    
    # Create performance comparison plot
    fig1 = viz.compare_methods_bar_chart(methods, metric='kendall_tau')
    if fig1:
        fig1.savefig('kendall_tau_comparison.png')
        print("Saved kendall_tau_comparison.png")
    
    # Create rank comparison plot for a method
    fig2 = viz.plot_rank_comparison('DIGRAC')
    if fig2:
        fig2.savefig('rank_comparison_DIGRAC.png')
        print("Saved rank_comparison_DIGRAC.png")
    
    # Visualize the network with predicted rankings
    fig3 = viz.visualize_network(method_name='DIGRAC', highlight_upsets=True)
    if fig3:
        fig3.savefig('network_visualization.png')
        print("Saved network_visualization.png")
    
    plt.show()
