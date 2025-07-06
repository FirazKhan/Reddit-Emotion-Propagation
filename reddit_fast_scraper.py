import praw
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials from .env
load_dotenv()
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
REDDIT_USERNAME = os.getenv('REDDIT_USERNAME')
REDDIT_PASSWORD = os.getenv('REDDIT_PASSWORD')

# Target subreddits
TARGET_SUBREDDITS = [
    'relationship_advice',
    'AmITheAsshole',
    'news',
    'depression'
]

# Fast collection parameters (1.5 hour target)
TARGET_COMMENTS = 25000   # Reduced for faster collection
TARGET_THREADS = 1000     # Reduced for faster collection
MIN_REPLY_DEPTH = 2       # Reduced for more data
MIN_COMMENT_LENGTH = 10   # Reduced for more data
MAX_COLLECTION_TIME = 1800  # Max 30 minutes per subreddit (1.5 hours total)


def get_reddit_instance():
    """Initialize Reddit instance with proper error handling"""
    try:
        # Ensure user_agent follows Reddit's format
        if not REDDIT_USER_AGENT or REDDIT_USER_AGENT == 'ECMM447Project by u/your_reddit_username':
            user_agent = f"ECMM447Project/1.0 (by /u/{REDDIT_USERNAME})"
        else:
            user_agent = REDDIT_USER_AGENT
        
        logger.info(f"Initializing Reddit instance with user_agent: {user_agent}")
        
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=user_agent,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD
        )
        
        # Test authentication
        logger.info("Testing Reddit authentication...")
        user = reddit.user.me()
        logger.info(f"Successfully authenticated as: {user}")
        
        return reddit
        
    except Exception as e:
        logger.error(f"Failed to initialize Reddit instance: {e}")
        logger.error("Please check your credentials in the .env file")
        raise


def count_words(text):
    """Count words in text"""
    if not text or pd.isna(text):
        return 0
    return len(str(text).split())


def quick_depth_check(comment, max_check=2):
    """Very quick check for thread depth"""
    if max_check <= 0:
        return 0
    
    replies = list(comment.replies)
    if not replies:
        return 0
    
    return 1 + max(quick_depth_check(reply, max_check - 1) for reply in replies[:3])


def scrape_subreddit_fast(reddit, subreddit_name, target_posts=250):
    """Fast scraping with minimal filtering"""
    logger.info(f"Fast scraping r/{subreddit_name} (target: {target_posts} posts)")
    
    start_time = time.time()
    posts_data = []
    comments_data = []
    posts_processed = 0
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Use only 'top' for fastest collection
        sort_methods = ['top']
        
        for sort_method in sort_methods:
            if posts_processed >= target_posts or (time.time() - start_time) > MAX_COLLECTION_TIME:
                break
                
            logger.info(f"Fast scraping r/{subreddit_name} using {sort_method} sorting")
            
            try:
                submissions = subreddit.top(time_filter='year', limit=target_posts)
                
                for submission in submissions:
                    if posts_processed >= target_posts or (time.time() - start_time) > MAX_COLLECTION_TIME:
                        break
                    
                    try:
                        # Quick check: skip if too few comments
                        if submission.num_comments < 5:
                            continue
                        
                        # Get comments with minimal expansion for speed
                        submission.comments.replace_more(limit=5)  # Very limited expansion
                        all_comments = submission.comments.list()
                        
                        # Quick filter by length
                        valid_comments = []
                        for comment in all_comments:
                            if count_words(comment.body) >= MIN_COMMENT_LENGTH:
                                valid_comments.append(comment)
                        
                        # Very quick depth check
                        if len(valid_comments) >= 3:  
                            max_depth = 0
                            for comment in valid_comments[:5]:  
                                depth = quick_depth_check(comment)
                                max_depth = max(max_depth, depth)
                            
                            if max_depth >= MIN_REPLY_DEPTH:
                                # Add submission data
                                posts_data.append({
                                    'id': submission.id,
                                    'title': submission.title,
                                    'selftext': submission.selftext,
                                    'score': submission.score,
                                    'num_comments': submission.num_comments,
                                    'created_utc': submission.created_utc,
                                    'author': str(submission.author) if submission.author else '[deleted]',
                                    'subreddit': subreddit_name,
                                    'url': submission.url,
                                    'max_thread_depth': max_depth,
                                    'valid_comments_count': len(valid_comments)
                                })
                                
                                # Add comment data
                                for comment in valid_comments:
                                    comments_data.append({
                                        'id': comment.id,
                                        'parent_id': comment.parent_id,
                                        'body': comment.body,
                                        'score': comment.score,
                                        'created_utc': comment.created_utc,
                                        'author': str(comment.author) if comment.author else '[deleted]',
                                        'subreddit': subreddit_name,
                                        'submission_id': submission.id,
                                        'word_count': count_words(comment.body),
                                        'thread_depth': quick_depth_check(comment)
                                    })
                                
                                posts_processed += 1
                                if posts_processed % 20 == 0:
                                    logger.info(f"Fast processed {posts_processed}/{target_posts} posts from r/{subreddit_name} (comments: {len(comments_data)})")
                        
                        # fast rate limiting
                        time.sleep(0.02)
                        
                    except Exception as e:
                        logger.warning(f"Error processing submission {submission.id}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error with {sort_method} sorting for r/{subreddit_name}: {e}")
                continue
        
        logger.info(f"Fast scraped {len(posts_data)} posts and {len(comments_data)} comments from r/{subreddit_name}")
        return posts_data, comments_data
        
    except Exception as e:
        logger.error(f"Failed to scrape r/{subreddit_name}: {e}")
        return [], []


def main():
    """Main function to run the fast scraper"""
    try:
        # Initialize Reddit instance
        reddit = get_reddit_instance()
        
        all_posts = []
        all_comments = []
        total_posts_target = TARGET_THREADS // len(TARGET_SUBREDDITS)
        
        logger.info(f"Starting FAST data collection (1.5 hour target):")
        logger.info(f"Target: {TARGET_COMMENTS:,} comments, {TARGET_THREADS:,} threads")
        logger.info(f"Min thread depth: {MIN_REPLY_DEPTH}")
        logger.info(f"Min comment length: {MIN_COMMENT_LENGTH} words")
        logger.info(f"Max time per subreddit: {MAX_COLLECTION_TIME//60} minutes")
        
        # Scrape each subreddit
        for subreddit in TARGET_SUBREDDITS:
            try:
                posts, comments = scrape_subreddit_fast(reddit, subreddit, total_posts_target)
                all_posts.extend(posts)
                all_comments.extend(comments)
                
                logger.info(f"Progress: {len(all_posts)} posts, {len(all_comments)} comments collected so far")
                
                # Minimal rate limiting between subreddits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to process subreddit {subreddit}: {e}")
                continue
        
        # Save to CSV
        if all_posts or all_comments:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('data/raw', exist_ok=True)
            
            if all_posts:
                posts_df = pd.DataFrame(all_posts)
                posts_file = f'data/raw/reddit_posts_fast_{timestamp}.csv'
                posts_df.to_csv(posts_file, index=False)
                logger.info(f"Saved {len(posts_df)} posts to {posts_file}")
            
            if all_comments:
                comments_df = pd.DataFrame(all_comments)
                comments_file = f'data/raw/reddit_comments_fast_{timestamp}.csv'
                comments_df.to_csv(comments_file, index=False)
                logger.info(f"Saved {len(comments_df)} comments to {comments_file}")
            
            # Summary statistics
            logger.info(f"FAST data collection completed successfully!")
            logger.info(f"Total posts: {len(all_posts):,}")
            logger.info(f"Total comments: {len(all_comments):,}")
            logger.info(f"Average comments per post: {len(all_comments)/len(all_posts):.1f}" if all_posts else "N/A")
            
            # Thread depth statistics
            if all_posts:
                depths = [post['max_thread_depth'] for post in all_posts]
                logger.info(f"Average thread depth: {sum(depths)/len(depths):.1f}")
                logger.info(f"Max thread depth: {max(depths)}")
            
            # Comment length statistics
            if all_comments:
                word_counts = [comment['word_count'] for comment in all_comments]
                logger.info(f"Average comment length: {sum(word_counts)/len(word_counts):.1f} words")
                logger.info(f"Min comment length: {min(word_counts)} words")
                logger.info(f"Max comment length: {max(word_counts)} words")
        else:
            logger.error("No data was collected from any subreddit")
            
    except Exception as e:
        logger.error(f"Fatal error in main function: {e}")
        raise

if __name__ == "__main__":
    main() 