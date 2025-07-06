import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import logging
import os
from typing import Dict, List, Tuple
import warnings
import json
import random
warnings.filterwarnings('ignore')

# Import our modules
from emotion_detector import AdvancedEmotionDetector
from network_builder import RedditNetworkBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastRedditAnalysisPipeline:
    """Fast analysis pipeline with sampling for large networks"""
    
    def __init__(self):
        self.emotion_detector = AdvancedEmotionDetector()
        self.network_builder = RedditNetworkBuilder()
        self.comments_df = None
        self.posts_df = None
        self.network = None
        self.analysis_results = {}
        
    def load_latest_data(self):
        """Load the most recent data files"""
        data_dir = "data/raw"
        
        # Find the most recent files
        comment_files = [f for f in os.listdir(data_dir) if f.startswith('reddit_comments_fast_')]
        post_files = [f for f in os.listdir(data_dir) if f.startswith('reddit_posts_fast_')]
        
        if not comment_files or not post_files:
            raise FileNotFoundError("No data files found in data/raw/")
        
        # Get the most recent files (highest timestamp)
        latest_comments = max(comment_files)
        latest_posts = max(post_files)
        
        comments_file = os.path.join(data_dir, latest_comments)
        posts_file = os.path.join(data_dir, latest_posts)
        
        logger.info(f"Loading data from {comments_file} and {posts_file}")
        
        self.comments_df = pd.read_csv(comments_file)
        self.posts_df = pd.read_csv(posts_file)
        
        logger.info(f"Loaded {len(self.comments_df)} comments and {len(self.posts_df)} posts")
        
    def preprocess_data(self):
        """Preprocess the data for analysis"""
        logger.info("Preprocessing data")
        
        # Clean comments data
        self.comments_df = self.comments_df.copy()
        
        # Remove deleted/removed comments
        self.comments_df = self.comments_df[
            (self.comments_df['body'].notna()) & 
            (self.comments_df['body'] != '[deleted]') & 
            (self.comments_df['body'] != '[removed]')
        ]
        
        # Convert timestamps
        self.comments_df['created_utc'] = pd.to_datetime(self.comments_df['created_utc'], unit='s')
        self.posts_df['created_utc'] = pd.to_datetime(self.posts_df['created_utc'], unit='s')
        
        # Add word count if not present
        if 'word_count' not in self.comments_df.columns:
            self.comments_df['word_count'] = self.comments_df['body'].str.split().str.len()
        
        logger.info(f"Preprocessed {len(self.comments_df)} comments")
        
    def detect_emotions(self, sample_size: int = 5000):
        """Detect emotions in comments with multi-emotion analysis (using sample for speed)"""
        logger.info(f"Detecting emotions in {sample_size} comments with multi-emotion analysis")
        
        # Sample comments for emotion detection
        if len(self.comments_df) > sample_size:
            sample_df = self.comments_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = self.comments_df.copy()
        
        # Use the new multi-emotion analysis
        sample_df = self.emotion_detector.analyze_dataframe(sample_df, 'body')
        
        # Merge with original comments (include all new emotion columns)
        emotion_columns = ['primary_emotion', 'emotion_1', 'emotion_2', 'emotion_3', 
                          'emotion_combination', 'emotion_intensity', 'emotion_complexity']
        emotion_columns += [f'emotion_{e}' for e in self.emotion_detector.emotions]
        emotion_columns += [f'emotion_{i}_confidence' for i in range(1, 4)]
        
        self.comments_df = self.comments_df.merge(
            sample_df[['id'] + emotion_columns], 
            on='id', 
            how='left'
        )
        
        # Fill missing values
        self.comments_df['primary_emotion'].fillna('neutral', inplace=True)
        self.comments_df['emotion_1'].fillna('neutral', inplace=True)
        self.comments_df['emotion_2'].fillna('none', inplace=True)
        self.comments_df['emotion_3'].fillna('none', inplace=True)
        self.comments_df['emotion_combination'].fillna('neutral', inplace=True)
        self.comments_df['emotion_intensity'].fillna(0.0, inplace=True)
        self.comments_df['emotion_complexity'].fillna(1, inplace=True)
        
        logger.info(f"Multi-emotion detection completed for {len(sample_df)} comments")
        
    def build_network(self):
        """Build the reply network"""
        logger.info("Building reply network")
        
        # Use the network builder
        self.network_builder.comments_df = self.comments_df
        self.network_builder.posts_df = self.posts_df
        
        # Preprocess and build network
        processed_comments = self.network_builder.preprocess_comments()
        self.network = self.network_builder.build_reply_network(processed_comments)
        
        # Add emotion attributes
        self.network = self.network_builder.add_emotion_attributes(self.network, self.comments_df)
        
        logger.info(f"Network built with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
        
    def calculate_fast_network_metrics(self, sample_size: int = 10000):
        """Calculate network metrics using sampling for speed"""
        logger.info("Calculating fast network metrics using sampling")
        
        if self.network is None:
            raise ValueError("Network not built yet")
        
        metrics = {}
        
        # Basic network statistics (fast)
        metrics['basic_stats'] = {
            'nodes': self.network.number_of_nodes(),
            'edges': self.network.number_of_edges(),
            'density': nx.density(self.network),
            'is_directed': self.network.is_directed(),
            'is_weighted': False
        }
        
        # Sample nodes for centrality calculations
        if self.network.number_of_nodes() > sample_size:
            sample_nodes = random.sample(list(self.network.nodes()), sample_size)
            sample_network = self.network.subgraph(sample_nodes)
        else:
            sample_network = self.network
        
        # Fast centrality measures on sample
        logger.info("Calculating centrality measures on sample...")
        metrics['centrality_sample'] = {
            'degree_centrality': nx.degree_centrality(sample_network),
            'in_degree_centrality': nx.in_degree_centrality(sample_network) if sample_network.is_directed() else None,
            'out_degree_centrality': nx.out_degree_centrality(sample_network) if sample_network.is_directed() else None,
        }
        
        # Degree distribution (full network)
        degrees = [d for n, d in self.network.degree()]
        metrics['degree_distribution'] = {
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'min_degree': np.min(degrees),
            'max_degree': np.max(degrees)
        }
        
        # Connected components (fast)
        if self.network.is_directed():
            components = list(nx.strongly_connected_components(self.network))
        else:
            components = list(nx.connected_components(self.network))
        
        largest_cc = max(components, key=len) if components else set()
        metrics['connected_components'] = {
            'num_components': len(components),
            'largest_component_size': len(largest_cc),
            'largest_component_ratio': len(largest_cc) / self.network.number_of_nodes() if self.network.number_of_nodes() > 0 else 0
        }
        
        logger.info("Fast network metrics calculated")
        return metrics
        
    def analyze_emotion_spread_fast(self, sample_size: int = 10000):
        """Analyze emotion spread patterns using sampling"""
        logger.info("Analyzing emotion spread patterns (fast)")
        
        if self.network is None:
            raise ValueError("Network not built yet")
        
        analysis = {}
        
        # Get nodes with emotion data
        emotion_nodes = [n for n in self.network.nodes() if 'primary_emotion' in self.network.nodes[n]]
        
        if not emotion_nodes:
            logger.warning("No emotion data found in network")
            return analysis
        
        # Emotion distribution
        emotion_counts = {}
        for node in emotion_nodes:
            emotion = self.network.nodes[node]['primary_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        analysis['emotion_distribution'] = emotion_counts
        
        # Sample edges for emotion spread analysis
        all_edges = list(self.network.edges())
        if len(all_edges) > sample_size:
            sample_edges = random.sample(all_edges, sample_size)
        else:
            sample_edges = all_edges
        
        # Analyze emotion spread along sampled edges
        emotion_edges = []
        for u, v in sample_edges:
            if 'primary_emotion' in self.network.nodes[u] and 'primary_emotion' in self.network.nodes[v]:
                emotion_edges.append({
                    'source_emotion': self.network.nodes[u]['primary_emotion'],
                    'target_emotion': self.network.nodes[v]['primary_emotion'],
                    'source_intensity': self.network.nodes[u].get('emotion_intensity', 0),
                    'target_intensity': self.network.nodes[v].get('emotion_intensity', 0)
                })
        
        analysis['emotion_transitions'] = emotion_edges
        
        # Calculate emotion contagion metrics
        if emotion_edges:
            same_emotion_count = sum(1 for edge in emotion_edges 
                                   if edge['source_emotion'] == edge['target_emotion'])
            analysis['emotion_contagion_rate'] = same_emotion_count / len(emotion_edges)
        
        logger.info("Fast emotion spread analysis completed")
        return analysis
    
    def _analyze_multi_emotions(self):
        """Analyze multi-emotion patterns"""
        logger.info("Analyzing multi-emotion patterns")
        
        # Filter comments with emotion data
        emotion_df = self.comments_df.dropna(subset=['emotion_1', 'emotion_2', 'emotion_3'])
        
        if len(emotion_df) == 0:
            return {}
        
        analysis = {
            'total_comments_with_multi_emotions': len(emotion_df),
            'emotion_combinations': emotion_df['emotion_combination'].value_counts().head(15).to_dict(),
            'avg_emotion_complexity': float(emotion_df['emotion_complexity'].mean()),
            'emotion_complexity_distribution': emotion_df['emotion_complexity'].value_counts().sort_index().to_dict(),
            'single_vs_multiple': {
                'single_emotion': len(emotion_df[~emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)]),
                'multiple_emotions': len(emotion_df[emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)])
            },
            'top_emotions_by_position': {}
        }
        
        # Analyze top emotions by position (1st, 2nd, 3rd)
        for i in range(1, 4):
            emotion_col = f'emotion_{i}'
            if emotion_col in emotion_df.columns:
                analysis['top_emotions_by_position'][f'position_{i}'] = \
                    emotion_df[emotion_col].value_counts().head(10).to_dict()
        
        # Analyze emotion confidence patterns
        confidence_analysis = {}
        for i in range(1, 4):
            conf_col = f'emotion_{i}_confidence'
            if conf_col in emotion_df.columns:
                confidence_analysis[f'position_{i}_confidence'] = {
                    'mean': float(emotion_df[conf_col].mean()),
                    'std': float(emotion_df[conf_col].std()),
                    'min': float(emotion_df[conf_col].min()),
                    'max': float(emotion_df[conf_col].max())
                }
        
        analysis['confidence_analysis'] = confidence_analysis
        
        logger.info("Multi-emotion analysis completed")
        return analysis
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations")
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Emotion distribution
        self._plot_emotion_distribution()
        
        # 2. Network degree distribution
        self._plot_degree_distribution()
        
        # 3. Emotion by subreddit
        self._plot_emotion_by_subreddit()
        
        # 4. Multi-emotion analysis
        self._plot_multi_emotion_analysis()
        
        # 5. Emotion combinations
        self._plot_emotion_combinations()
        
        # 4. Network visualization (sample)
        self._plot_network_sample()
        
        logger.info("Visualizations created")
        
    def _plot_emotion_distribution(self):
        """Plot emotion distribution"""
        plt.figure(figsize=(12, 6))
        
        # Emotion distribution
        emotion_counts = self.comments_df['primary_emotion'].value_counts()
        
        plt.subplot(1, 2, 1)
        emotion_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribution of Primary Emotions')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Emotion intensity distribution
        plt.subplot(1, 2, 2)
        plt.hist(self.comments_df['emotion_intensity'], bins=30, alpha=0.7, color='lightcoral')
        plt.title('Distribution of Emotion Intensity')
        plt.xlabel('Emotion Intensity')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data/processed/emotion_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_degree_distribution(self):
        """Plot network degree distribution"""
        if self.network is None:
            return
            
        plt.figure(figsize=(10, 6))
        
        degrees = [d for n, d in self.network.degree()]
        
        plt.subplot(1, 2, 1)
        plt.hist(degrees, bins=50, alpha=0.7, color='lightgreen')
        plt.title('Degree Distribution')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # Log-log plot for power law check
        degree_counts = pd.Series(degrees).value_counts().sort_index()
        plt.loglog(degree_counts.index, degree_counts.values, 'o-', alpha=0.7)
        plt.title('Degree Distribution (Log-Log)')
        plt.xlabel('Degree')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('data/processed/degree_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_emotion_by_subreddit(self):
        """Plot emotion distribution by subreddit"""
        plt.figure(figsize=(14, 8))
        
        # Create emotion-subreddit heatmap
        emotion_subreddit = pd.crosstab(
            self.comments_df['primary_emotion'], 
            self.comments_df['subreddit']
        )
        
        sns.heatmap(emotion_subreddit, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Emotion Distribution by Subreddit')
        plt.xlabel('Subreddit')
        plt.ylabel('Emotion')
        
        plt.tight_layout()
        plt.savefig('data/processed/emotion_by_subreddit.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multi_emotion_analysis(self):
        """Plot multi-emotion analysis"""
        logger.info("Creating multi-emotion analysis plots")
        
        # Filter comments with emotion data
        emotion_df = self.comments_df.dropna(subset=['emotion_1', 'emotion_2', 'emotion_3'])
        
        if len(emotion_df) == 0:
            logger.warning("No emotion data available for multi-emotion analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Emotion Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top 3 emotions distribution
        top_emotions = []
        for i in range(1, 4):
            emotion_col = f'emotion_{i}'
            if emotion_col in emotion_df.columns:
                emotions = emotion_df[emotion_col].value_counts().head(10)
                top_emotions.extend(emotions.index.tolist())
        
        unique_emotions = list(set(top_emotions))
        
        # Count occurrences of each emotion in top 3
        emotion_counts = {emotion: 0 for emotion in unique_emotions}
        for i in range(1, 4):
            emotion_col = f'emotion_{i}'
            if emotion_col in emotion_df.columns:
                for emotion in emotion_df[emotion_col]:
                    if emotion in emotion_counts:
                        emotion_counts[emotion] += 1
        
        # Plot top emotions
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        emotions, counts = zip(*sorted_emotions)
        
        axes[0, 0].bar(emotions, counts, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Top Emotions in Multi-Emotion Analysis')
        axes[0, 0].set_xlabel('Emotion')
        axes[0, 0].set_ylabel('Count (in top 3)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Emotion complexity distribution
        if 'emotion_complexity' in emotion_df.columns:
            complexity_counts = emotion_df['emotion_complexity'].value_counts().sort_index()
            axes[0, 1].bar(complexity_counts.index, complexity_counts.values, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Emotion Complexity Distribution')
            axes[0, 1].set_xlabel('Number of Significant Emotions')
            axes[0, 1].set_ylabel('Count')
        
        # 3. Emotion confidence distribution
        confidence_data = []
        for i in range(1, 4):
            conf_col = f'emotion_{i}_confidence'
            if conf_col in emotion_df.columns:
                confidence_data.extend(emotion_df[conf_col].dropna().tolist())
        
        if confidence_data:
            axes[1, 0].hist(confidence_data, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Emotion Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Count')
        
        # 4. Emotion intensity vs complexity
        if 'emotion_intensity' in emotion_df.columns and 'emotion_complexity' in emotion_df.columns:
            axes[1, 1].scatter(emotion_df['emotion_complexity'], emotion_df['emotion_intensity'], 
                             alpha=0.6, color='purple')
            axes[1, 1].set_title('Emotion Intensity vs Complexity')
            axes[1, 1].set_xlabel('Emotion Complexity')
            axes[1, 1].set_ylabel('Emotion Intensity')
        
        plt.tight_layout()
        plt.savefig('data/processed/multi_emotion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emotion_combinations(self):
        """Plot emotion combinations analysis"""
        logger.info("Creating emotion combinations plots")
        
        # Filter comments with emotion combination data
        emotion_df = self.comments_df.dropna(subset=['emotion_combination'])
        
        if len(emotion_df) == 0:
            logger.warning("No emotion combination data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Emotion Combinations Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top emotion combinations
        combination_counts = emotion_df['emotion_combination'].value_counts().head(15)
        
        axes[0].barh(range(len(combination_counts)), combination_counts.values, color='coral', alpha=0.7)
        axes[0].set_yticks(range(len(combination_counts)))
        axes[0].set_yticklabels(combination_counts.index)
        axes[0].set_title('Top Emotion Combinations')
        axes[0].set_xlabel('Count')
        
        # 2. Single vs multiple emotions
        single_emotions = emotion_df[emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)]
        multiple_emotions = emotion_df[~emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)]
        
        labels = ['Single Emotion', 'Multiple Emotions']
        sizes = [len(multiple_emotions), len(single_emotions)]
        colors = ['lightblue', 'lightcoral']
        
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Single vs Multiple Emotions')
        
        plt.tight_layout()
        plt.savefig('data/processed/emotion_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_network_sample(self):
        """Plot a sample of the network"""
        if self.network is None:
            logger.warning("No network available for visualization")
            return
            
        logger.info("Creating network visualization")
        
        # Find a meaningful sample for visualization
        # Start with nodes that have connections (degree > 0)
        connected_nodes = [n for n, d in self.network.degree() if d > 0]
        
        if len(connected_nodes) == 0:
            logger.warning("No connected nodes found in network")
            return
            
        # Take a sample of connected nodes (max 500 for visualization)
        if len(connected_nodes) > 500:
            sample_nodes = random.sample(connected_nodes, 500)
        else:
            sample_nodes = connected_nodes
            
        # Create subgraph with these nodes and their immediate neighbors
        sample_network = self.network.subgraph(sample_nodes).copy()
        
        # If the sample is still too sparse, try to find a denser subgraph
        if sample_network.number_of_edges() < 50:
            # Find nodes with higher degree
            high_degree_nodes = [n for n, d in self.network.degree() if d >= 2]
            if len(high_degree_nodes) > 0:
                # Take a smaller sample of high-degree nodes
                sample_size = min(200, len(high_degree_nodes))
                sample_nodes = random.sample(high_degree_nodes, sample_size)
                sample_network = self.network.subgraph(sample_nodes).copy()
        
        logger.info(f"Network sample: {sample_network.number_of_nodes()} nodes, {sample_network.number_of_edges()} edges")
        
        if sample_network.number_of_edges() == 0:
            logger.warning("No edges in network sample - creating alternative visualization")
            self._create_network_statistics_plot()
            return
            
        plt.figure(figsize=(14, 12))
        
        # Create layout with better parameters for sparse networks
        pos = nx.spring_layout(sample_network, k=2, iterations=100, scale=2)
        
        # Color nodes by emotion
        emotion_colors = {
            'joy': '#FFD700', 'sadness': '#4169E1', 'anger': '#DC143C', 
            'fear': '#8A2BE2', 'surprise': '#FF8C00', 'disgust': '#8B4513',
            'neutral': '#808080', 'optimism': '#32CD32'
        }
        
        # Get node colors and sizes based on degree
        node_colors = []
        node_sizes = []
        
        for node in sample_network.nodes():
            emotion = sample_network.nodes[node].get('primary_emotion', 'neutral')
            node_colors.append(emotion_colors.get(emotion, '#808080'))
            
            # Size based on degree (minimum 50, maximum 300)
            degree = sample_network.degree(node)
            node_sizes.append(max(50, min(300, 50 + degree * 20)))
        
        # Draw network
        nx.draw(sample_network, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8,
                with_labels=False,
                arrows=True,
                arrowstyle='->',
                arrowsize=10,
                edge_color='gray',
                edge_alpha=0.3,
                linewidths=0.5)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=emotion)
                          for emotion, color in emotion_colors.items()]
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'Reddit Reply Network Sample\n({sample_network.number_of_nodes()} nodes, {sample_network.number_of_edges()} edges)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/processed/network_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Network visualization completed")
    
    def _create_network_statistics_plot(self):
        """Create alternative visualization when network is too sparse"""
        logger.info("Creating network statistics plot")
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots for network statistics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reddit Network Statistics', fontsize=16, fontweight='bold')
        
        # 1. Degree distribution
        degrees = [d for n, d in self.network.degree()]
        axes[0, 0].hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Degree Distribution')
        axes[0, 0].set_xlabel('Degree')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')
        
        # 2. Connected components size distribution
        if self.network.is_directed():
            components = list(nx.strongly_connected_components(self.network))
        else:
            components = list(nx.connected_components(self.network))
        
        component_sizes = [len(comp) for comp in components]
        axes[0, 1].hist(component_sizes, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Connected Component Sizes')
        axes[0, 1].set_xlabel('Component Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        
        # 3. Emotion distribution in network
        emotion_counts = {}
        for node in self.network.nodes():
            emotion = self.network.nodes[node].get('primary_emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        emotions, counts = zip(*emotion_counts.items())
        axes[1, 0].bar(emotions, counts, color='coral', alpha=0.7)
        axes[1, 0].set_title('Emotion Distribution in Network')
        axes[1, 0].set_xlabel('Emotion')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Network statistics summary
        stats_text = f"""
Network Statistics:
• Nodes: {self.network.number_of_nodes():,}
• Edges: {self.network.number_of_edges():,}
• Density: {nx.density(self.network):.6f}
• Average Degree: {sum(degrees)/len(degrees):.2f}
• Connected Components: {len(components)}
• Largest Component: {max(component_sizes):,}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_title('Network Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/processed/network_sample.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Network statistics plot completed")
        
    def save_results(self):
        """Save all analysis results"""
        logger.info("Saving analysis results")
        
        # Create output directory
        os.makedirs('data/processed', exist_ok=True)
        
        # Save processed data
        self.comments_df.to_csv('data/processed/comments_with_emotions.csv', index=False)
        self.posts_df.to_csv('data/processed/posts_processed.csv', index=False)
        
        # Save network (sample for size)
        if self.network is not None:
            # Save a sample of the network for visualization
            if self.network.number_of_nodes() > 10000:
                sample_nodes = random.sample(list(self.network.nodes()), 10000)
                sample_network = self.network.subgraph(sample_nodes).copy()
            else:
                sample_network = self.network.copy()
            # Convert all node attributes to strings for GML
            for n, d in sample_network.nodes(data=True):
                for k in d:
                    if not isinstance(d[k], str):
                        d[k] = str(d[k])
            nx.write_gml(sample_network, 'data/processed/reddit_network_sample.gml')
        
        # Save analysis results
        with open('data/processed/analysis_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results = {}
            for key, value in self.analysis_results.items():
                if isinstance(value, dict):
                    results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            results[key][k] = {kk: float(vv) if isinstance(vv, np.number) else vv 
                                             for kk, vv in v.items()}
                        else:
                            results[key][k] = float(v) if isinstance(v, np.number) else v
                else:
                    results[key] = value
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report()
        
        logger.info("Results saved to data/processed/")
        
    def _create_summary_report(self):
        """Create a summary report"""
        # Precompute values to avoid backslash in f-string expressions
        multi_emotion_df = self.comments_df.dropna(subset=['emotion_1', 'emotion_2', 'emotion_3'])
        single_emotion_count = len(multi_emotion_df[~multi_emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)])
        multiple_emotion_count = len(multi_emotion_df[multi_emotion_df['emotion_combination'].str.contains('\\+', na=False, regex=True)])
        top_emotion_comb = multi_emotion_df['emotion_combination'].mode().iloc[0] if not multi_emotion_df['emotion_combination'].mode().empty else 'N/A'
        avg_emotion_complexity = multi_emotion_df['emotion_complexity'].mean() if 'emotion_complexity' in multi_emotion_df.columns else 0.0

        report = f"""
# Reddit Social Network Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total Posts: {len(self.posts_df):,}
- Total Comments: {len(self.comments_df):,}
- Subreddits: {', '.join(self.comments_df['subreddit'].unique())}

## Emotion Analysis
- Comments with emotions detected: {len(self.comments_df[self.comments_df['primary_emotion'] != 'neutral']):,}
- Most common emotion: {self.comments_df['primary_emotion'].mode().iloc[0] if not self.comments_df['primary_emotion'].mode().empty else 'N/A'}
- Average emotion intensity: {self.comments_df['emotion_intensity'].mean():.3f}

## Multi-Emotion Analysis
- Comments with multi-emotion data: {len(multi_emotion_df):,}
- Average emotion complexity: {avg_emotion_complexity:.2f} emotions per comment
- Single emotion comments: {single_emotion_count:,}
- Multiple emotion comments: {multiple_emotion_count:,}
- Top emotion combination: {top_emotion_comb}

## Network Analysis
"""
        
        if self.network is not None:
            report += f"""
- Network nodes: {self.network.number_of_nodes():,}
- Network edges: {self.network.number_of_edges():,}
- Network density: {nx.density(self.network):.6f}
- Average degree: {sum(dict(self.network.degree()).values()) / self.network.number_of_nodes():.2f}
"""
        
        # Save report
        with open('data/processed/analysis_report.md', 'w') as f:
            f.write(report)
        
    def run_complete_analysis(self, emotion_sample_size: int = 5000, 
                            network_sample_size: int = 10000, 
                            emotion_spread_sample_size: int = 10000):
        """Run the complete fast analysis pipeline with configurable sample sizes
        
        Args:
            emotion_sample_size: Number of comments to sample for emotion detection
            network_sample_size: Number of nodes to sample for centrality calculations
            emotion_spread_sample_size: Number of edges to sample for emotion spread analysis
        """
        logger.info(f"Starting fast analysis pipeline with sample sizes: "
                   f"emotion={emotion_sample_size}, network={network_sample_size}, "
                   f"spread={emotion_spread_sample_size}")
        
        try:
            # 1. Load data
            self.load_latest_data()
            
            # 2. Preprocess data
            self.preprocess_data()
            
            # 3. Detect emotions
            self.detect_emotions(sample_size=emotion_sample_size)
            
            # 4. Build network
            self.build_network()
            
            # 5. Analyze network metrics (fast)
            self.analysis_results['network_metrics'] = self.calculate_fast_network_metrics(
                sample_size=network_sample_size
            )
            
            # 6. Analyze emotion spread (fast)
            self.analysis_results['emotion_spread'] = self.analyze_emotion_spread_fast(
                sample_size=emotion_spread_sample_size
            )
            
            # 7. Add multi-emotion analysis results
            self.analysis_results['multi_emotion_analysis'] = self._analyze_multi_emotions()
            
            # 8. Create visualizations
            self.create_visualizations()
            
            # 9. Save results
            self.save_results()
            
            logger.info("Fast analysis pipeline finished successfully!")
            
        except Exception as e:
            logger.error(f"Error in fast analysis pipeline: {e}")
            raise

def main():
    """Main function to run the fast analysis"""
    pipeline = FastRedditAnalysisPipeline()
    
    # Full dataset analysis
    pipeline.run_complete_analysis(emotion_sample_size=417023, network_sample_size=417023, emotion_spread_sample_size=154117)

if __name__ == "__main__":
    main() 