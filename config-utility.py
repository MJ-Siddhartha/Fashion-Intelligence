# src/utils/config.py
import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    visual_model_path: str = 'models/visual_model/dinov2_base'
    nlp_model_path: str = 'models/nlp_model/spacy_en_core_web_sm'
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

@dataclass
class DataProcessingConfig:
    """Configuration for data processing"""
    raw_data_path: str = 'data/raw/'
    processed_data_path: str = 'data/processed/'
    supported_image_formats: List[str] = ['jpg', 'jpeg', 'png', 'gif']
    max_text_length: int = 1024

@dataclass
class OntologyConfig:
    """Configuration for ontology construction"""
    base_categories: Dict[str, List[str]] = {
        'Apparel': ['Top', 'Bottom', 'Dress', 'Outerwear'],
        'Accessories': ['Jewelry', 'Bags', 'Footwear', 'Headwear']
    }
    feature_similarity_threshold: float = 0.7

@dataclass
class SystemConfig:
    """Overarching system configuration"""
    debug_mode: bool = False
    log_level: str = 'INFO'
    allow_human_feedback: bool = True
    max_concurrent_jobs: int = 4

# Utility functions
def setup_logging():
    """Configure logging for the entire system"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('system.log'),
            logging.StreamHandler()
        ]
    )

def validate_data_paths():
    """Ensure required data paths exist"""
    required_paths = [
        'data/raw',
        'data/processed',
        'models/visual_model',
        'models/nlp_model',
        'reports'
    ]
    
    for path in required_paths:
        os.makedirs(path, exist_ok=True)

# Performance metrics utility
def calculate_extraction_metrics(results):
    """
    Calculate comprehensive feature extraction metrics
    
    Args:
        results (dict): Processing results
    
    Returns:
        dict of performance metrics
    """
    total_items = results.get('total_items', 0)
    processed_categories = results.get('processed_categories', {})
    
    metrics = {
        'total_processed_items': total_items,
        'category_coverage': len(processed_categories),
        'feature_extraction_rates': {
            category: (
                results['processed_categories'][category]['features_extracted'] / 
                results['processed_categories'][category]['total_items']
            ) * 100 for category in processed_categories
        }
    }
    
    return metrics

# Main execution configuration
if __name__ == '__main__':
    setup_logging()
    validate_data_paths()
