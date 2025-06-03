"""
ML Engineer Portfolio - Professional Asset Management System
Handles the integration of professional visual assets for the portfolio.
"""
import requests
from typing import List, Dict, Optional
import random

class ProfessionalAssetManager:
    """Manages professional visual assets for the ML Engineer portfolio"""
    
    IMAGE_CATEGORIES = {
        "portfolio": ["machine learning", "artificial intelligence", "data science", "technology", "coding"],
        "ml": ["neural network", "algorithm", "data visualization", "machine learning", "ai"],
        "data": ["data science", "analytics", "big data", "visualization", "dashboard"],
        "tech": ["technology", "programming", "code", "computer", "digital"],
        "business": ["professional", "business", "office", "workspace", "meeting"]
    }
    
    @staticmethod
    def get_project_images(project_type: str, count: int = 6) -> List[str]:
        """Fetch contextually relevant professional images for ML projects"""
        categories = ProfessionalAssetManager.IMAGE_CATEGORIES.get(
            project_type.lower(), ["machine learning", "artificial intelligence", "data science"]
        )
        
        images = []
        for i in range(count):
            category = categories[i % len(categories)]
            # Generate unique URLs to avoid caching issues
            seed = random.randint(1000, 9999)
            img_url = f"https://source.unsplash.com/800x600/?{category}&sig={seed}"
            images.append(img_url)
        
        return images
    
    @staticmethod
    def get_hero_image(project_type: str) -> str:
        """Get high-quality hero image for main portfolio section"""
        categories = ProfessionalAssetManager.IMAGE_CATEGORIES.get(
            project_type.lower(), ["artificial intelligence"]
        )
        primary_category = categories[0]
        seed = random.randint(10000, 99999)
        return f"https://source.unsplash.com/1200x600/?{primary_category}&sig={seed}"
    
    @staticmethod
    def get_skill_icons() -> Dict[str, str]:
        """Get icons for common ML/data science skills"""
        return {
            "Python": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg",
            "TensorFlow": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg",
            "PyTorch": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg",
            "Scikit-learn": "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg",
            "Pandas": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg",
            "NumPy": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg",
            "Docker": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg",
            "AWS": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/amazonwebservices/amazonwebservices-original.svg",
            "GCP": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg",
            "Kubernetes": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg",
            "SQL": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/mysql/mysql-original.svg",
            "Git": "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg"
        }