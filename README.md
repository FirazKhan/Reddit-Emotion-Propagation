# Reddit-Emotion-Propagation
A research project exploring how emotions like anger, joy, and sadness spread through Reddit comment threads. Combines fine-grained emotion classification (GoEmotions) with network analysis to model digital emotional contagion. Techniques include centrality analysis, community detection, and predictive modeling of emotional cascades.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/reddit-emotion-network-analysis.git
   cd reddit-emotion-network-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Reddit API credentials**
   ```bash
   # Create .env file with your Reddit API credentials
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

## 📈 Usage

### Quick Start - Run Complete Analysis
```bash
python fast_analysis_pipeline.py
```

### Individual Components

1. **Data Collection**
   ```bash
   python reddit_fast_scraper.py
   ```

2. **Emotion Analysis**
   ```python
   from emotion_detector import EmotionDetector
   detector = EmotionDetector()
   emotions = detector.analyze_text("Your text here")
   ```

3. **Network Analysis**
   ```python
   from network_builder import NetworkBuilder
   builder = NetworkBuilder()
   network = builder.build_network(comments_df)
   ```

## 🔬 Methodology

### Data Collection
- **Subreddits**: relationship_advice, AmITheAsshole, news, depression
- **Time Period**: April 2024 - January 2025
- **Collection Method**: PRAW API with optimized batch processing
- **Processing Time**: 3.5 hours for complete dataset

### Emotion Detection
- **Model**: Transformer-based emotion classification
- **Emotions**: 10 categories + neutral (anger, joy, sadness, fear, etc.)
- **Multi-emotion**: Top 3 emotions per comment with confidence scores
- **Accuracy**: 65-70% primary emotions, 75-80% multi-emotion detection

### Network Analysis
- **Nodes**: Individual comments (417,023)
- **Edges**: Reply relationships (154,117)
- **Metrics**: Degree distribution, clustering coefficient, centrality measures
- **Visualization**: Gephi-compatible GML export for advanced visualization

## 📊 Key Results

### Emotion Distribution
- **Neutral**: 53.7% (dominant across all communities)
- **Anger**: 8.7% (higher in AmITheAsshole: 9.7%)
- **Pessimism**: 8.1% (prevalent in depression subreddit)
- **Joy**: 4.2% (scattered distribution)

### Network Properties
- **Scale-free**: Power-law degree distribution
- **Low Density**: Sparse network with isolated clusters
- **Community Structure**: Emotion-based clustering patterns
- **Propagation**: Limited emotion contagion, more community-specific patterns

## 📚 Dependencies

- **Data Science**: pandas, numpy, matplotlib, seaborn
- **Network Analysis**: networkx, scipy
- **NLP**: transformers, torch, nltk, textblob
- **Data Collection**: praw, requests, python-dotenv
- **Visualization**: plotly, wordcloud
- **Machine Learning**: scikit-learn

## 📖 Academic Paper

The complete research findings are documented in `report.pdf` - an IEEE conference paper format including:
- Literature review of emotion propagation research
- Detailed methodology and technical implementation
- Comprehensive results with statistical analysis
- Discussion of limitations and future work
- Full bibliography and academic citations

## 🤝 Contributing

This project was developed as part of ECMM447 Social Network Analysis coursework. Contributions, suggestions, and improvements are welcome!

## 📄 License

This project is available for academic and research purposes. Please cite appropriately if using this work.

## 🙏 Acknowledgments

- Reddit API (PRAW) for data access
- Hugging Face Transformers for emotion detection models
- NetworkX community for network analysis tools
- ECMM447 course instructors and peers for guidance

## 📧 Contact

For questions about this research or collaboration opportunities, please open an issue or contact the repository maintainer.

---

**Note**: This repository contains the complete pipeline for Reddit emotion network analysis. The processed dataset (`comments_with_emotions.csv`) contains 417K+ comments with emotion labels and can be used for further research in emotion propagation, community analysis, or social network studies.
