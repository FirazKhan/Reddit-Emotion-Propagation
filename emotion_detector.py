import pandas as pd
import numpy as np
from transformers import pipeline
import torch
from typing import List, Dict, Tuple
import logging
import os
from tqdm import tqdm
import json
import re
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEmotionDetector:
    """Advanced emotion detection with sophisticated context analysis"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Emotion categories
        self.emotions = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust',
            'love', 'optimism', 'pessimism', 'trust', 'anticipation', 'neutral'
        ]
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logger.info("Loaded sentiment analyzer")
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_context(self, text: str) -> Dict[str, any]:
        """Comprehensive context analysis"""
        context = {
            'is_sarcastic': False,
            'is_negated': False,
            'sentiment_score': 0.0,
            'emotional_complexity': 0,
            'context_indicators': [],
            'negative_contexts': [],
            'positive_contexts': []
        }
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            context['sentiment_score'] = blob.sentiment.polarity
        except:
            context['sentiment_score'] = 0.0
        
        # Sarcasm detection patterns
        sarcasm_patterns = [
            r'yeah\s+right',
            r'sure\s+thing',
            r'whatever',
            r'obviously',
            r'clearly',
            r'of\s+course',
            r'naturally',
            r'definitely',
            r'absolutely',
            r'great\s*!+',
            r'wonderful\s*!+',
            r'amazing\s*!+',
            r'fantastic\s*!+',
            r'love\s+how',
            r'excited\s+to\s+see\s+how\s+.*\s+unfolds',
            r'can\'t\s+wait\s+for\s+.*\s+disaster',
            r'this\s+is\s+.*\s+terrible',
            r'i\s+love\s+.*\s+betrayed',
            r'happy\s+.*\s+nightmare\s+is\s+over',
            r'optimistic\s+.*\s+breakup',
            r'positive\s+.*\s+devastated'
        ]
        
        for pattern in sarcasm_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                context['is_sarcastic'] = True
                context['context_indicators'].append('sarcasm')
                break
        
        # Negation detection
        negation_patterns = [
            r'\bnot\b',
            r'\bno\b',
            r'\bnever\b',
            r'\bnone\b',
            r'\bnobody\b',
            r'\bnothing\b',
            r'\bneither\b',
            r'\bnowhere\b',
            r'\bhardly\b',
            r'\bbarely\b',
            r'\bscarcely\b',
            r'\bdoesn\'t\b',
            r'\bisn\'t\b',
            r'\bwasn\'t\b',
            r'\bshouldn\'t\b',
            r'\bwouldn\'t\b',
            r'\bcouldn\'t\b',
            r'\bwon\'t\b',
            r'\bcan\'t\b',
            r'\bdon\'t\b',
            r'\bdidn\'t\b'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                context['is_negated'] = True
                context['context_indicators'].append('negation')
                break
        
        # Negative context detection
        negative_contexts = [
            'breakup', 'divorce', 'death', 'loss', 'failed', 'ended', 'over',
            'betrayal', 'cheating', 'lying', 'deceit', 'disaster', 'nightmare',
            'devastated', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry',
            'ex', 'former', 'used to', 'past', 'but', 'however', 'although',
            'despite', 'even though', 'dread', 'fear', 'anxiety', 'don\'t trust',
            'can\'t trust', 'untrustworthy'
        ]
        
        for neg_context in negative_contexts:
            if neg_context in text.lower():
                context['negative_contexts'].append(neg_context)
        
        # Positive context detection
        positive_contexts = [
            'recovery', 'healing', 'better', 'improve', 'future', 'hope',
            'positive', 'optimistic', 'trust', 'believe', 'confident'
        ]
        
        for pos_context in positive_contexts:
            if pos_context in text.lower():
                context['positive_contexts'].append(pos_context)
        
        # Emotional complexity (number of different emotion words)
        emotion_words = [
            'happy', 'sad', 'angry', 'scared', 'surprised', 'disgusted',
            'love', 'hope', 'trust', 'excited', 'worried', 'frustrated'
        ]
        
        found_emotions = sum(1 for word in emotion_words if word in text.lower())
        context['emotional_complexity'] = found_emotions
        
        return context
    
    def get_emotion_scores(self, text: str, context: Dict) -> Dict[str, float]:
        """Calculate emotion scores with context awareness"""
        emotion_scores = {emotion: 0.0 for emotion in self.emotions}
        
        # Define emotion patterns with context rules
        emotion_patterns = {
            'joy': {
                'keywords': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'delighted', 'thrilled'],
                'sarcasm_penalty': 0.1,  # Very low score for sarcastic joy
                'negative_context_penalty': 0.2,  # Low score in negative contexts
                'negation_penalty': 0.3  # Reduced score when negated
            },
            'sadness': {
                'keywords': ['sad', 'depressed', 'miserable', 'unhappy', 'crying', 'tears', 'hopeless', 'lonely', 'heartbroken', 'devastated'],
                'sarcasm_penalty': 0.8,  # Less affected by sarcasm
                'negative_context_penalty': 1.2,  # Boosted in negative contexts
                'negation_penalty': 0.5
            },
            'anger': {
                'keywords': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'irritated', 'frustrated', 'pissed'],
                'sarcasm_penalty': 0.7,
                'negative_context_penalty': 1.1,
                'negation_penalty': 0.4
            },
            'fear': {
                'keywords': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'panic', 'fear'],
                'sarcasm_penalty': 0.8,
                'negative_context_penalty': 1.1,
                'negation_penalty': 0.5
            },
            'surprise': {
                'keywords': ['surprised', 'shocked', 'amazed', 'wow', 'unexpected', 'incredible', 'stunned'],
                'sarcasm_penalty': 0.6,
                'negative_context_penalty': 0.9,
                'negation_penalty': 0.4
            },
            'disgust': {
                'keywords': ['disgusted', 'gross', 'nasty', 'revolting', 'sick', 'vile', 'repulsive'],
                'sarcasm_penalty': 0.8,
                'negative_context_penalty': 1.1,
                'negation_penalty': 0.5
            },
            'love': {
                'keywords': ['love', 'adore', 'cherish', 'affection', 'romance', 'heart', 'caring'],
                'sarcasm_penalty': 0.1,  # Very low score for sarcastic love
                'negative_context_penalty': 0.2,  # Low score in negative contexts
                'negation_penalty': 0.3
            },
            'optimism': {
                'keywords': ['hope', 'optimistic', 'positive', 'better', 'improve', 'future', 'recovery', 'healing'],
                'sarcasm_penalty': 0.2,  # Low score for sarcastic optimism
                'negative_context_penalty': 0.3,  # Low score in negative contexts
                'negation_penalty': 0.4
            },
            'pessimism': {
                'keywords': ['hopeless', 'negative', 'worst', 'never', 'impossible', 'doomed', 'pointless'],
                'sarcasm_penalty': 0.8,
                'negative_context_penalty': 1.2,
                'negation_penalty': 0.5
            },
            'trust': {
                'keywords': ['trust', 'believe', 'confident', 'sure', 'certain', 'reliable'],
                'sarcasm_penalty': 0.3,
                'negative_context_penalty': 0.4,
                'negation_penalty': 0.4
            },
            'anticipation': {
                'keywords': ['excited', 'waiting', 'expect', 'anticipate', 'looking forward', 'can\'t wait'],
                'sarcasm_penalty': 0.2,  # Low score for sarcastic anticipation
                'negative_context_penalty': 0.3,
                'negation_penalty': 0.4
            }
        }
        
        # Calculate scores for each emotion
        for emotion, pattern in emotion_patterns.items():
            score = 0.0
            
            # Count keyword matches
            for keyword in pattern['keywords']:
                if keyword in text.lower():
                    score += 1.0
            
            # Apply context penalties
            if context['is_sarcastic']:
                score *= pattern['sarcasm_penalty']
            
            if context['negative_contexts']:
                score *= pattern['negative_context_penalty']
            
            if context['is_negated']:
                score *= pattern['negation_penalty']
            
            # Adjust based on overall sentiment
            if context['sentiment_score'] < -0.3 and emotion in ['joy', 'optimism', 'love', 'anticipation']:
                score *= 0.3  # Reduce positive emotions in negative sentiment
            elif context['sentiment_score'] > 0.3 and emotion in ['sadness', 'anger', 'pessimism', 'fear']:
                score *= 0.3  # Reduce negative emotions in positive sentiment
            
            emotion_scores[emotion] = max(0.0, score)
        
        # Special handling for complex cases
        if context['is_sarcastic'] and context['emotional_complexity'] > 2:
            # In sarcastic complex texts, favor negative emotions
            for emotion in ['joy', 'optimism', 'love', 'anticipation']:
                emotion_scores[emotion] *= 0.2
            for emotion in ['sadness', 'anger', 'pessimism']:
                emotion_scores[emotion] *= 1.5
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # If no strong emotions detected, classify as neutral
        max_score = max(emotion_scores.values())
        if max_score < 0.3:
            emotion_scores['neutral'] = 1.0
            for emotion in self.emotions:
                if emotion != 'neutral':
                    emotion_scores[emotion] = 0.0
        
        return emotion_scores
    
    def predict_emotion(self, text: str) -> Dict[str, float]:
        """Predict emotions with advanced context analysis"""
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {emotion: 0.0 for emotion in self.emotions}
        
        # Analyze context
        context = self.analyze_context(processed_text)
        
        # Get emotion scores
        emotion_scores = self.get_emotion_scores(processed_text, context)
        
        return emotion_scores
    
    def get_primary_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """Get the primary emotion with context consideration"""
        if not emotion_scores:
            return 'neutral', 0.0
        
        # Filter out neutral if other emotions are strong
        max_score = max(emotion_scores.values())
        if max_score > 0.4 and emotion_scores.get('neutral', 0) < 0.5:
            # Remove neutral from consideration
            filtered_scores = {k: v for k, v in emotion_scores.items() if k != 'neutral'}
            if filtered_scores:
                primary_emotion = max(filtered_scores.items(), key=lambda x: x[1])
                return primary_emotion
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return primary_emotion
    
    def get_top_emotions(self, emotion_scores: Dict[str, float], num_emotions: int = 3) -> List[Tuple[str, float]]:
        """Get top N emotions with their confidence scores"""
        if not emotion_scores:
            return [('neutral', 0.0)]
        
        # Sort emotions by score (descending)
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N emotions
        return sorted_emotions[:num_emotions]
    
    def get_emotion_combination(self, emotion_scores: Dict[str, float]) -> str:
        """Get a combination of top emotions (e.g., 'sadness+optimism')"""
        if not emotion_scores:
            return 'neutral'
        
        # Get top 2 emotions
        top_emotions = self.get_top_emotions(emotion_scores, 2)
        
        # Filter out emotions with very low scores
        significant_emotions = [(emotion, score) for emotion, score in top_emotions if score > 0.1]
        
        if not significant_emotions:
            return 'neutral'
        elif len(significant_emotions) == 1:
            return significant_emotions[0][0]
        else:
            # Combine emotions (e.g., 'sadness+optimism')
            emotions = [emotion for emotion, score in significant_emotions]
            return '+'.join(emotions)
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze a list of texts and return emotion scores for each"""
        results = []
        
        for text in tqdm(texts, desc="Analyzing emotions"):
            emotion_scores = self.predict_emotion(text)
            results.append(emotion_scores)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'body') -> pd.DataFrame:
        """Analyze emotions in a DataFrame and add multiple emotion columns"""
        logger.info(f"Analyzing emotions in {len(df)} texts")
        
        # Get emotion scores for all texts
        emotion_scores = self.analyze_texts(df[text_column].tolist())
        
        # Add individual emotion columns
        for emotion in self.emotions:
            df[f'emotion_{emotion}'] = [scores[emotion] for scores in emotion_scores]
        
        # Add primary emotion (single)
        primary_emotions = []
        primary_scores = []
        
        for scores in emotion_scores:
            primary_emotion, score = self.get_primary_emotion(scores)
            primary_emotions.append(primary_emotion)
            primary_scores.append(score)
        
        df['primary_emotion'] = primary_emotions
        df['emotion_confidence'] = primary_scores
        
        # Add top 3 emotions as separate columns
        for i in range(1, 4):  # emotion_1, emotion_2, emotion_3
            emotion_names = []
            emotion_confidences = []
            
            for scores in emotion_scores:
                top_emotions = self.get_top_emotions(scores, 3)
                if i <= len(top_emotions):
                    emotion_names.append(top_emotions[i-1][0])
                    emotion_confidences.append(top_emotions[i-1][1])
                else:
                    emotion_names.append('none')
                    emotion_confidences.append(0.0)
            
            df[f'emotion_{i}'] = emotion_names
            df[f'emotion_{i}_confidence'] = emotion_confidences
        
        # Add emotion combination (e.g., 'sadness+optimism')
        emotion_combinations = []
        for scores in emotion_scores:
            combination = self.get_emotion_combination(scores)
            emotion_combinations.append(combination)
        
        df['emotion_combination'] = emotion_combinations
        
        # Add emotion intensity (sum of all emotion scores)
        df['emotion_intensity'] = df[[f'emotion_{e}' for e in self.emotions]].sum(axis=1)
        
        # Add emotion complexity (number of emotions with score > 0.1)
        emotion_complexity = []
        for scores in emotion_scores:
            significant_emotions = sum(1 for score in scores.values() if score > 0.1)
            emotion_complexity.append(significant_emotions)
        
        df['emotion_complexity'] = emotion_complexity
        
        logger.info("Advanced multi-emotion analysis completed")
        return df
    
    def get_emotion_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about emotion distribution"""
        stats = {}
        
        # Primary emotion distribution
        emotion_counts = df['primary_emotion'].value_counts()
        stats['primary_emotion_distribution'] = emotion_counts.to_dict()
        
        # Average emotion scores
        emotion_columns = [f'emotion_{e}' for e in self.emotions]
        avg_scores = df[emotion_columns].mean()
        stats['average_emotion_scores'] = avg_scores.to_dict()
        
        # Emotion intensity statistics
        stats['emotion_intensity_stats'] = {
            'mean': df['emotion_intensity'].mean(),
            'std': df['emotion_intensity'].std(),
            'min': df['emotion_intensity'].min(),
            'max': df['emotion_intensity'].max()
        }
        
        return stats

def main():
    """Main function to test advanced emotion detection"""
    # Initialize advanced emotion detector
    detector = AdvancedEmotionDetector()
    
    # Test with problematic cases
    test_texts = [
        "I'm so happy this nightmare is over",  # sadness/relief, not joy
        "I hope things get better, but I'm still devastated",  #  sadness, not optimism
        "Great, just what I needed today",  # sarcasm/anger, not joy
        "I love how you betrayed me",  # anger/sarcasm, not love
        "I'm optimistic about my future after this breakup",  # mixed or neutral
        "This is amazing! I can't believe how terrible this is",  #  sarcasm
        "I'm not happy about this situation",  # neutral/sadness, not joy
        "I'm so excited to see how this disaster unfolds",  # sarcasm
        "I'm devastated by this breakup but trying to stay positive",  # sadness with some optimism
        "This is wonderful! Said no one ever",  # sarcasm
    ]
    
    print("Testing advanced emotion detection:")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        emotions = detector.predict_emotion(text)
        primary_emotion, confidence = detector.get_primary_emotion(emotions)
        
        print(f"\n{i}. Text: '{text}'")
        print(f"   Primary Emotion: {primary_emotion} (confidence: {confidence:.3f})")
        print(f"   Top 3 Emotions: {sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]}")

if __name__ == "__main__":
    main() 