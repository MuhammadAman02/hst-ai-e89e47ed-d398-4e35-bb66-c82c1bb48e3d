"""
ML Engineer Portfolio - Main Content and Page Definitions
Defines the structure and content of the portfolio website.
"""
from nicegui import ui, app
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from app.core.assets import ProfessionalAssetManager

# Configure global styling
with ui.head():
    ui.add_head_html('''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    ''')
    ui.add_head_html('<meta name="viewport" content="width=device-width, initial-scale=1.0">')

# Load professional assets for the portfolio
asset_manager = ProfessionalAssetManager()
portfolio_images = asset_manager.get_project_images("portfolio", count=6)
hero_image = asset_manager.get_hero_image("portfolio")

# Sample ML Engineer data - Replace with your information
PORTFOLIO_DATA = {
    "name": "Alex Johnson",
    "title": "Machine Learning Engineer",
    "summary": "Experienced ML Engineer with 5+ years of expertise in developing and deploying machine learning models for real-world applications. Specialized in computer vision, natural language processing, and recommendation systems.",
    "skills": {
        "Machine Learning": 95,
        "Deep Learning": 90,
        "Python": 95,
        "TensorFlow/PyTorch": 85,
        "Computer Vision": 80,
        "NLP": 85,
        "MLOps": 75,
        "Data Engineering": 70,
        "Cloud Platforms": 80,
        "SQL/NoSQL": 75
    },
    "experience": [
        {
            "role": "Senior ML Engineer",
            "company": "AI Solutions Inc.",
            "period": "2021 - Present",
            "description": "Lead ML engineer for computer vision projects. Developed and deployed models for object detection and image segmentation with 95% accuracy.",
            "technologies": ["PyTorch", "TensorFlow", "AWS", "Docker", "Kubernetes"]
        },
        {
            "role": "Machine Learning Engineer",
            "company": "DataTech Systems",
            "period": "2018 - 2021",
            "description": "Developed recommendation systems that increased user engagement by 35%. Implemented NLP models for sentiment analysis and text classification.",
            "technologies": ["Python", "Scikit-learn", "Keras", "GCP", "SQL"]
        },
        {
            "role": "Data Scientist",
            "company": "Tech Innovations",
            "period": "2016 - 2018",
            "description": "Analyzed large datasets to extract actionable insights. Built predictive models for customer behavior and business forecasting.",
            "technologies": ["Python", "R", "Pandas", "NumPy", "Matplotlib"]
        }
    ],
    "education": [
        {
            "degree": "M.S. in Computer Science",
            "institution": "Stanford University",
            "year": "2016",
            "focus": "Machine Learning and Artificial Intelligence"
        },
        {
            "degree": "B.S. in Mathematics",
            "institution": "University of California, Berkeley",
            "year": "2014",
            "focus": "Statistics and Data Analysis"
        }
    ],
    "projects": [
        {
            "title": "Real-time Object Detection System",
            "description": "Developed a real-time object detection system using YOLOv5 that processes video streams with 30 FPS on edge devices.",
            "image": portfolio_images[0],
            "technologies": ["PyTorch", "YOLO", "OpenCV", "CUDA"],
            "metrics": {"Accuracy": "93.5%", "FPS": "30", "mAP": "0.89"}
        },
        {
            "title": "NLP-powered Customer Service Bot",
            "description": "Created an intelligent chatbot using BERT and GPT models that handles 70% of customer inquiries without human intervention.",
            "image": portfolio_images[1],
            "technologies": ["Transformers", "BERT", "FastAPI", "React"],
            "metrics": {"Accuracy": "87%", "Response Time": "1.2s", "User Satisfaction": "4.7/5"}
        },
        {
            "title": "Recommendation Engine for E-commerce",
            "description": "Built a hybrid recommendation system that increased conversion rates by 28% and average order value by 15%.",
            "image": portfolio_images[2],
            "technologies": ["TensorFlow", "Keras", "PostgreSQL", "Redis"],
            "metrics": {"CTR Improvement": "+28%", "AOV Increase": "+15%", "Engagement": "+32%"}
        },
        {
            "title": "Predictive Maintenance for Manufacturing",
            "description": "Implemented a time-series forecasting model that predicts equipment failures with 92% accuracy, reducing downtime by 45%.",
            "image": portfolio_images[3],
            "technologies": ["Prophet", "XGBoost", "Scikit-learn", "Docker"],
            "metrics": {"Accuracy": "92%", "Downtime Reduction": "45%", "ROI": "320%"}
        }
    ],
    "certifications": [
        "Google Cloud Professional Machine Learning Engineer",
        "AWS Certified Machine Learning - Specialty",
        "Deep Learning Specialization - deeplearning.ai",
        "TensorFlow Developer Certificate"
    ],
    "contact": {
        "email": "alex.johnson@example.com",
        "linkedin": "linkedin.com/in/alexjohnson",
        "github": "github.com/alexjohnson",
        "twitter": "twitter.com/alexjohnson"
    }
}

# Create sample data for ML visualizations
def generate_sample_training_data():
    """Generate sample training data for visualization"""
    epochs = list(range(1, 51))
    train_accuracy = [min(0.5 + 0.008 * e + np.random.normal(0, 0.02), 0.99) for e in epochs]
    val_accuracy = [min(0.5 + 0.007 * e + np.random.normal(0, 0.03), 0.97) for e in epochs]
    train_loss = [0.8 * np.exp(-0.05 * e) + 0.1 + np.random.normal(0, 0.02) for e in epochs]
    val_loss = [0.8 * np.exp(-0.04 * e) + 0.15 + np.random.normal(0, 0.03) for e in epochs]
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_loss': train_loss,
        'val_loss': val_loss
    })

def generate_confusion_matrix():
    """Generate a sample confusion matrix for visualization"""
    # Create a sample 4x4 confusion matrix
    classes = ['Cat', 'Dog', 'Bird', 'Fish']
    cm = np.array([
        [85, 10, 3, 2],
        [7, 90, 2, 1],
        [5, 3, 88, 4],
        [2, 1, 5, 92]
    ])
    return cm, classes

def create_model_performance_chart():
    """Create a training/validation performance chart"""
    df = generate_sample_training_data()
    
    # Create two subplots
    fig = go.Figure()
    
    # Add traces for accuracy
    fig.add_trace(go.Scatter(
        x=df['epoch'], 
        y=df['train_accuracy'], 
        mode='lines',
        name='Training Accuracy',
        line=dict(color='#3b82f6', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['epoch'], 
        y=df['val_accuracy'], 
        mode='lines',
        name='Validation Accuracy',
        line=dict(color='#10b981', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Training Performance',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
    )
    
    return fig

def create_confusion_matrix_chart():
    """Create a confusion matrix visualization"""
    cm, classes = generate_confusion_matrix()
    
    # Create heatmap
    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True
    )
    
    fig.update_layout(
        title='Confusion Matrix - Image Classification Model',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
    )
    
    return fig

def create_feature_importance_chart():
    """Create a feature importance visualization"""
    features = ['Image Contrast', 'Edge Detection', 'Color Histogram', 
                'Texture Analysis', 'Shape Descriptor', 'HOG Features',
                'SIFT Features', 'Pixel Intensity', 'Spatial Arrangement']
    importance = [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08]
    
    # Sort by importance
    sorted_indices = np.argsort(importance)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = [importance[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_features,
        x=sorted_importance,
        orientation='h',
        marker=dict(
            color='#3b82f6',
            line=dict(color='rgba(0, 0, 0, 0)', width=1)
        )
    ))
    
    fig.update_layout(
        title='Feature Importance - Computer Vision Model',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
    )
    
    return fig

# Define the main portfolio page
@ui.page('/')
def portfolio_page():
    # Add custom CSS
    ui.add_head_html('''
    <style>
    :root {
        --primary-50: #eff6ff;
        --primary-500: #3b82f6;
        --primary-600: #2563eb;
        --primary-700: #1d4ed8;
        --primary-900: #1e3a8a;
        
        --secondary-500: #8b5cf6;
        --secondary-600: #7c3aed;
        
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-500: #6b7280;
        --neutral-700: #374151;
        --neutral-900: #111827;
        
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        
        --font-display: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
    }
    
    body {
        font-family: var(--font-body);
        color: var(--neutral-900);
        background-color: var(--neutral-50);
    }
    
    .hero-section {
        background: linear-gradient(135deg, rgba(29, 78, 216, 0.8) 0%, rgba(124, 58, 237, 0.8) 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: var(--radius-lg);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('https://source.unsplash.com/1200x600/?artificial-intelligence,machine-learning');
        background-size: cover;
        background-position: center;
        opacity: 0.2;
        z-index: -1;
    }
    
    .section-title {
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: var(--primary-700);
        border-bottom: 2px solid var(--primary-200);
        padding-bottom: 0.5rem;
    }
    
    .card {
        background: white;
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-md);
        padding: 1.5rem;
        border: 1px solid var(--neutral-200);
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    
    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-4px);
        border-color: var(--primary-500);
    }
    
    .project-card {
        display: flex;
        flex-direction: column;
    }
    
    .project-image {
        height: 200px;
        background-size: cover;
        background-position: center;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
    }
    
    .skill-bar {
        height: 8px;
        background-color: var(--neutral-200);
        border-radius: 4px;
        margin-bottom: 1.5rem;
        overflow: hidden;
    }
    
    .skill-progress {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-600) 0%, var(--secondary-500) 100%);
        border-radius: 4px;
    }
    
    .experience-item {
        position: relative;
        padding-left: 28px;
        margin-bottom: 2rem;
    }
    
    .experience-item::before {
        content: "";
        position: absolute;
        left: 0;
        top: 6px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: var(--primary-600);
    }
    
    .experience-item::after {
        content: "";
        position: absolute;
        left: 5px;
        top: 24px;
        width: 2px;
        height: calc(100% + 12px);
        background-color: var(--neutral-300);
    }
    
    .experience-item:last-child::after {
        display: none;
    }
    
    .tech-tag {
        display: inline-block;
        background-color: var(--primary-50);
        color: var(--primary-700);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid var(--primary-200);
    }
    
    .metric-item {
        background-color: var(--neutral-100);
        padding: 0.5rem 1rem;
        border-radius: var(--radius-sm);
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .contact-link {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        color: var(--primary-700);
        text-decoration: none;
    }
    
    .contact-link:hover {
        color: var(--primary-500);
    }
    
    .contact-link i {
        margin-right: 0.75rem;
        font-size: 1.25rem;
    }
    
    @media (max-width: 768px) {
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .project-image {
            height: 160px;
        }
    }
    </style>
    ''')
    
    # Navigation
    with ui.header().classes('flex justify-between items-center p-4 bg-white shadow-sm'):
        ui.label(f"{PORTFOLIO_DATA['name']} | {PORTFOLIO_DATA['title']}").classes('text-xl font-bold text-blue-700')
        with ui.row().classes('gap-4'):
            ui.link('Home', '#home').classes('text-blue-700 hover:text-blue-500')
            ui.link('Projects', '#projects').classes('text-blue-700 hover:text-blue-500')
            ui.link('Skills', '#skills').classes('text-blue-700 hover:text-blue-500')
            ui.link('Experience', '#experience').classes('text-blue-700 hover:text-blue-500')
            ui.link('Contact', '#contact').classes('text-blue-700 hover:text-blue-500')
    
    # Main content container
    with ui.column().classes('w-full max-w-6xl mx-auto p-4 gap-12'):
        # Hero Section
        with ui.column().classes('hero-section mb-12').id('home'):
            ui.label(f"Hi, I'm {PORTFOLIO_DATA['name']}").classes('text-4xl font-bold mb-2')
            ui.label(PORTFOLIO_DATA['title']).classes('text-2xl mb-6')
            ui.label(PORTFOLIO_DATA['summary']).classes('text-lg max-w-3xl')
            
            with ui.row().classes('mt-8 gap-4'):
                with ui.link(target=f"mailto:{PORTFOLIO_DATA['contact']['email']}").classes('bg-white text-blue-700 px-6 py-3 rounded-lg font-semibold shadow-md hover:bg-blue-50 transition-all'):
                    ui.icon('email', size='lg').classes('mr-2')
                    ui.label('Contact Me')
                
                with ui.link(target=f"https://{PORTFOLIO_DATA['contact']['github']}").classes('bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold shadow-md hover:bg-blue-800 transition-all'):
                    ui.icon('code', size='lg').classes('mr-2')
                    ui.label('View Projects')
        
        # ML Visualization Section
        with ui.column().classes('mb-12 bg-white p-6 rounded-xl shadow-md'):
            ui.label('ML Model Visualizations').classes('text-2xl font-bold mb-6 text-blue-700')
            ui.label('Interactive visualizations of machine learning models and results').classes('text-lg mb-6 text-gray-600')
            
            with ui.tabs().classes('w-full') as tabs:
                performance_tab = ui.tab('Model Performance')
                confusion_tab = ui.tab('Confusion Matrix')
                feature_tab = ui.tab('Feature Importance')
            
            with ui.tab_panels(tabs, value=performance_tab).classes('w-full mt-4'):
                with ui.tab_panel(performance_tab):
                    ui.plotly(create_model_performance_chart()).classes('w-full')
                
                with ui.tab_panel(confusion_tab):
                    ui.plotly(create_confusion_matrix_chart()).classes('w-full')
                
                with ui.tab_panel(feature_tab):
                    ui.plotly(create_feature_importance_chart()).classes('w-full')
        
        # Projects Section
        with ui.column().classes('mb-12').id('projects'):
            ui.label('Featured Projects').classes('section-title text-2xl')
            
            with ui.grid(columns=2).classes('gap-6'):
                for project in PORTFOLIO_DATA['projects']:
                    with ui.card().classes('project-card'):
                        ui.image(project['image']).classes('w-full h-48 object-cover rounded-lg mb-4')
                        ui.label(project['title']).classes('text-xl font-bold mb-2 text-blue-700')
                        ui.label(project['description']).classes('mb-4 text-gray-700')
                        
                        with ui.row().classes('flex-wrap mb-4'):
                            for tech in project['technologies']:
                                ui.label(tech).classes('tech-tag')
                        
                        ui.label('Key Metrics').classes('font-semibold mb-2')
                        with ui.column().classes('w-full gap-2'):
                            for metric, value in project['metrics'].items():
                                with ui.row().classes('metric-item'):
                                    ui.label(metric).classes('font-medium')
                                    ui.label(value).classes('font-bold text-blue-700')
        
        # Skills Section
        with ui.column().classes('mb-12').id('skills'):
            ui.label('Skills & Expertise').classes('section-title text-2xl')
            
            with ui.grid(columns=2).classes('gap-6'):
                for skill, level in PORTFOLIO_DATA['skills'].items():
                    with ui.column().classes('mb-2'):
                        with ui.row().classes('justify-between mb-1'):
                            ui.label(skill).classes('font-medium')
                            ui.label(f"{level}%").classes('text-blue-700 font-bold')
                        with ui.element('div').classes('skill-bar'):
                            ui.element('div').classes('skill-progress').style(f'width: {level}%')
        
        # Experience Section
        with ui.column().classes('mb-12').id('experience'):
            ui.label('Professional Experience').classes('section-title text-2xl')
            
            with ui.column().classes('mt-6'):
                for exp in PORTFOLIO_DATA['experience']:
                    with ui.element('div').classes('experience-item'):
                        with ui.row().classes('justify-between mb-1'):
                            ui.label(exp['role']).classes('text-lg font-bold text-blue-700')
                            ui.label(exp['period']).classes('text-gray-600')
                        ui.label(exp['company']).classes('text-gray-800 font-medium mb-2')
                        ui.label(exp['description']).classes('mb-3 text-gray-700')
                        
                        with ui.row().classes('flex-wrap'):
                            for tech in exp['technologies']:
                                ui.label(tech).classes('tech-tag')
        
        # Education Section
        with ui.column().classes('mb-12'):
            ui.label('Education & Certifications').classes('section-title text-2xl')
            
            with ui.grid(columns=2).classes('gap-6 mb-8'):
                for edu in PORTFOLIO_DATA['education']:
                    with ui.card().classes('card'):
                        ui.label(edu['degree']).classes('text-lg font-bold text-blue-700 mb-1')
                        ui.label(edu['institution']).classes('font-medium mb-1')
                        ui.label(edu['year']).classes('text-gray-600 mb-2')
                        ui.label(f"Focus: {edu['focus']}").classes('text-gray-700')
            
            ui.label('Certifications').classes('text-xl font-bold mb-4')
            with ui.column().classes('gap-2'):
                for cert in PORTFOLIO_DATA['certifications']:
                    with ui.row().classes('items-center'):
                        ui.icon('verified', color='green').classes('mr-2')
                        ui.label(cert).classes('text-gray-800')
        
        # Contact Section
        with ui.column().classes('mb-12').id('contact'):
            ui.label('Get In Touch').classes('section-title text-2xl')
            
            with ui.grid(columns=2).classes('gap-6'):
                with ui.card().classes('card'):
                    ui.label('Contact Information').classes('text-xl font-bold mb-4 text-blue-700')
                    
                    with ui.column().classes('gap-4'):
                        with ui.link(target=f"mailto:{PORTFOLIO_DATA['contact']['email']}").classes('contact-link'):
                            ui.icon('email').classes('mr-2')
                            ui.label(PORTFOLIO_DATA['contact']['email'])
                        
                        with ui.link(target=f"https://{PORTFOLIO_DATA['contact']['linkedin']}").classes('contact-link'):
                            ui.icon('link').classes('mr-2')
                            ui.label(PORTFOLIO_DATA['contact']['linkedin'])
                        
                        with ui.link(target=f"https://{PORTFOLIO_DATA['contact']['github']}").classes('contact-link'):
                            ui.icon('code').classes('mr-2')
                            ui.label(PORTFOLIO_DATA['contact']['github'])
                        
                        with ui.link(target=f"https://{PORTFOLIO_DATA['contact']['twitter']}").classes('contact-link'):
                            ui.icon('chat').classes('mr-2')
                            ui.label(PORTFOLIO_DATA['contact']['twitter'])
                
                with ui.card().classes('card'):
                    ui.label('Send a Message').classes('text-xl font-bold mb-4 text-blue-700')
                    
                    with ui.column().classes('gap-4 w-full'):
                        ui.input('Name').classes('w-full')
                        ui.input('Email').classes('w-full')
                        ui.input('Subject').classes('w-full')
                        ui.textarea('Message').classes('w-full')
                        
                        with ui.button('Send Message', color='blue').classes('mt-2'):
                            ui.icon('send').classes('mr-2')
        
        # Footer
        with ui.footer().classes('text-center py-6 text-gray-600 border-t border-gray-200 mt-12'):
            ui.label(f"© {PORTFOLIO_DATA['name']} | ML Engineer Portfolio {pd.Timestamp.now().year}")
            ui.label('Built with Python and NiceGUI').classes('text-sm mt-1')