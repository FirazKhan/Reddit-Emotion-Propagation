import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from collections import defaultdict
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditNetworkBuilder:
    """Build and analyze Reddit reply networks"""
    
    def __init__(self):
        self.graph = None
        self.comments_df = None
        self.posts_df = None
        
    def load_data(self, comments_file: str, posts_file: str = None):
        """Load Reddit data from CSV files"""
        logger.info(f"Loading comments from {comments_file}")
        self.comments_df = pd.read_csv(comments_file)
        
        if posts_file:
            logger.info(f"Loading posts from {posts_file}")
            self.posts_df = pd.read_csv(posts_file)
        
        logger.info(f"Loaded {len(self.comments_df)} comments")
        if self.posts_df is not None:
            logger.info(f"Loaded {len(self.posts_df)} posts")
    
    def preprocess_comments(self) -> pd.DataFrame:
        """Preprocess comments for network construction"""
        if self.comments_df is None:
            raise ValueError("No comments data loaded")
        
        # Create a copy to avoid modifying original
        df = self.comments_df.copy()
        
        # Filter out deleted/removed comments
        df = df[df['body'].notna()]
        df = df[df['body'] != '[deleted]']
        df = df[df['body'] != '[removed]']
        
        # Convert timestamps
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        
        # Ensure we have required columns
        required_cols = ['id', 'parent_id', 'body', 'score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Preprocessed {len(df)} comments")
        return df
    
    def build_reply_network(self, comments_df: pd.DataFrame) -> nx.DiGraph:
        """Build a directed reply network from comments"""
        logger.info("Building reply network")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Create a mapping from full parent_id to comment_id
        # Reddit format: t1_commentid for comments, t3_postid for posts
        parent_to_comment = {}
        
        # Add nodes (comments) and build parent mapping
        for _, comment in comments_df.iterrows():
            comment_id = comment['id']
            G.add_node(comment_id, 
                      body=comment['body'],
                      score=comment.get('score', 0),
                      subreddit=comment.get('subreddit', 'unknown'),
                      created_utc=comment.get('created_utc', None))
            
            # Map the full parent_id to this comment_id
            parent_to_comment[comment_id] = comment_id
            # Also map the t1_ format
            parent_to_comment[f"t1_{comment_id}"] = comment_id
        
        # Add edges (replies) - only for comment-to-comment relationships
        edge_count = 0
        for _, comment in comments_df.iterrows():
            parent_id = comment['parent_id']
            comment_id = comment['id']
            
            # Only process t1_ relationships (comment-to-comment)
            # Skip t3_ relationships (post-to-comment) as we want reply trees
            if parent_id.startswith('t1_'):
                # Try to find the parent comment in our mapping
                if parent_id in parent_to_comment:
                    parent_comment_id = parent_to_comment[parent_id]
                    if parent_comment_id != comment_id:  # Avoid self-loops
                        G.add_edge(parent_comment_id, comment_id)
                        edge_count += 1
        
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        logger.info(f"Created {edge_count} comment-to-comment reply relationships")
        return G
    
    def add_emotion_attributes(self, G: nx.DiGraph, comments_df: pd.DataFrame) -> nx.DiGraph:
        """Add emotion attributes to network nodes"""
        logger.info("Adding emotion attributes to network")
        
        # Get emotion columns
        emotion_cols = [col for col in comments_df.columns if col.startswith('emotion_')]
        
        if not emotion_cols:
            logger.warning("No emotion columns found in data")
            return G
        
        # Add emotion attributes to nodes
        for _, comment in comments_df.iterrows():
            if comment['id'] in G.nodes():
                # Add emotion scores
                for emotion_col in emotion_cols:
                    emotion_name = emotion_col.replace('emotion_', '')
                    G.nodes[comment['id']][f'emotion_{emotion_name}'] = comment.get(emotion_col, 0.0)
                
                # Add primary emotion
                if 'primary_emotion' in comment:
                    G.nodes[comment['id']]['primary_emotion'] = comment['primary_emotion']
                
                # Add emotion intensity
                if 'emotion_intensity' in comment:
                    G.nodes[comment['id']]['emotion_intensity'] = comment['emotion_intensity']
        
        logger.info(f"Added emotion attributes to {len(G.nodes())} nodes")
        return G
    
    def calculate_network_metrics(self, G: nx.DiGraph) -> Dict:
        """Calculate comprehensive network metrics"""
        logger.info("Calculating network metrics")
        
        metrics = {}
        
        # Basic network statistics
        metrics['basic_stats'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_directed': G.is_directed(),
            'is_weighted': hasattr(G, 'is_weighted') and G.is_weighted()
        }
        
        # Connected components
        if G.is_directed():
            components = list(nx.strongly_connected_components(G))
            largest_cc = max(components, key=len)
            metrics['connected_components'] = {
                'num_components': len(components),
                'largest_component_size': len(largest_cc),
                'largest_component_ratio': len(largest_cc) / G.number_of_nodes()
            }
        else:
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            metrics['connected_components'] = {
                'num_components': len(components),
                'largest_component_size': len(largest_cc),
                'largest_component_ratio': len(largest_cc) / G.number_of_nodes()
            }
        
        # Centrality measures
        metrics['centrality'] = {
            'degree_centrality': nx.degree_centrality(G),
            'in_degree_centrality': nx.in_degree_centrality(G) if G.is_directed() else None,
            'out_degree_centrality': nx.out_degree_centrality(G) if G.is_directed() else None,
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality_numpy(G)
        }
        
        # Clustering and community detection
        metrics['clustering'] = {
            'average_clustering': nx.average_clustering(G),
            'transitivity': nx.transitivity(G)
        }
        
        # Path lengths
        if nx.is_connected(G.to_undirected()):
            metrics['path_lengths'] = {
                'average_shortest_path_length': nx.average_shortest_path_length(G),
                'diameter': nx.diameter(G)
            }
        
        # Degree distribution
        degrees = [d for n, d in G.degree()]
        metrics['degree_distribution'] = {
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'min_degree': np.min(degrees),
            'max_degree': np.max(degrees)
        }
        
        logger.info("Network metrics calculated")
        return metrics
    
    def analyze_emotion_spread(self, G: nx.DiGraph) -> Dict:
        """Analyze how emotions spread through the network"""
        logger.info("Analyzing emotion spread patterns")
        
        analysis = {}
        
        # Get nodes with emotion data
        emotion_nodes = [n for n in G.nodes() if 'primary_emotion' in G.nodes[n]]
        
        if not emotion_nodes:
            logger.warning("No emotion data found in network")
            return analysis
        
        # Emotion distribution
        emotion_counts = defaultdict(int)
        for node in emotion_nodes:
            emotion = G.nodes[node]['primary_emotion']
            emotion_counts[emotion] += 1
        
        analysis['emotion_distribution'] = dict(emotion_counts)
        
        # Analyze emotion spread along edges
        emotion_edges = []
        for u, v in G.edges():
            if 'primary_emotion' in G.nodes[u] and 'primary_emotion' in G.nodes[v]:
                emotion_edges.append({
                    'source_emotion': G.nodes[u]['primary_emotion'],
                    'target_emotion': G.nodes[v]['primary_emotion'],
                    'source_intensity': G.nodes[u].get('emotion_intensity', 0),
                    'target_intensity': G.nodes[v].get('emotion_intensity', 0)
                })
        
        analysis['emotion_transitions'] = emotion_edges
        
        # Calculate emotion contagion metrics
        if emotion_edges:
            same_emotion_count = sum(1 for edge in emotion_edges 
                                   if edge['source_emotion'] == edge['target_emotion'])
            analysis['emotion_contagion_rate'] = same_emotion_count / len(emotion_edges)
        
        # Analyze emotion centrality
        emotion_centrality = defaultdict(list)
        for node in emotion_nodes:
            emotion = G.nodes[node]['primary_emotion']
            centrality = G.nodes[node].get('degree_centrality', 0)
            emotion_centrality[emotion].append(centrality)
        
        analysis['emotion_centrality'] = {
            emotion: {
                'mean': np.mean(centralities),
                'std': np.std(centralities),
                'count': len(centralities)
            }
            for emotion, centralities in emotion_centrality.items()
        }
        
        logger.info("Emotion spread analysis completed")
        return analysis
    
    def create_network_visualization(self, G: nx.DiGraph, output_file: str = "network_visualization.png"):
        """Create a network visualization"""
        logger.info("Creating network visualization")
        
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by primary emotion
        colors = []
        for node in G.nodes():
            emotion = G.nodes[node].get('primary_emotion', 'neutral')
            # Simple color mapping
            color_map = {
                'joy': 'yellow', 'sadness': 'blue', 'anger': 'red', 
                'fear': 'purple', 'surprise': 'orange', 'disgust': 'brown',
                'love': 'pink', 'optimism': 'lightgreen', 'pessimism': 'gray',
                'trust': 'cyan', 'anticipation': 'gold', 'neutral': 'lightgray'
            }
            colors.append(color_map.get(emotion, 'lightgray'))
        
        # Draw the network
        nx.draw(G, pos, 
                node_color=colors,
                node_size=50,
                alpha=0.7,
                with_labels=False,
                edge_color='gray',
                arrows=True if G.is_directed() else False)
        
        plt.title("Reddit Reply Network with Emotion Coloring")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Network visualization saved to {output_file}")
    
    def save_network(self, G: nx.DiGraph, output_file: str):
        """Save network to file"""
        logger.info(f"Saving network to {output_file}")
        
        # Save as GraphML for compatibility
        nx.write_graphml(G, output_file)
        
        # Also save as JSON for easy inspection
        json_file = output_file.replace('.graphml', '.json')
        network_data = {
            'nodes': [],
            'edges': []
        }
        
        for node, attrs in G.nodes(data=True):
            network_data['nodes'].append({
                'id': node,
                'attributes': attrs
            })
        
        for u, v, attrs in G.edges(data=True):
            network_data['edges'].append({
                'source': u,
                'target': v,
                'attributes': attrs
            })
        
        with open(json_file, 'w') as f:
            json.dump(network_data, f, indent=2, default=str)
        
        logger.info(f"Network saved in GraphML and JSON formats")

def main():
    """Main function to test network building"""
    # Initialize network builder
    builder = RedditNetworkBuilder()
    
    # Example usage (assuming you have data files)
    print("Reddit Network Builder for ECMM447 Project")
    print("=" * 50)
    print("This script builds reply networks from Reddit data")
    print("To use:")
    print("1. Load your Reddit data")
    print("2. Preprocess comments")
    print("3. Build reply network")
    print("4. Add emotion attributes")
    print("5. Calculate network metrics")
    print("6. Analyze emotion spread")

if __name__ == "__main__":
    main() 