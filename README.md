# ML Engineer Portfolio

A professional, interactive portfolio for showcasing machine learning engineering skills, projects, and experience. Built with Python and NiceGUI.

![ML Engineer Portfolio](https://source.unsplash.com/800x400/?machine-learning,portfolio)

## Features

- **Professional Hero Section**: Showcase your ML engineering expertise with a visually appealing introduction
- **Interactive ML Visualizations**: Display your machine learning models with interactive Plotly charts
- **Project Showcase**: Highlight your best ML projects with images, descriptions, and key metrics
- **Skills & Expertise**: Visual representation of your technical skills and proficiency levels
- **Professional Experience**: Timeline of your career progression and achievements
- **Education & Certifications**: Showcase your academic background and professional certifications
- **Contact Information**: Easy ways for potential employers or clients to reach you

## Quick Start

### Local Development

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```
4. Open your browser and navigate to `http://127.0.0.1:8000`

### Docker Deployment

Build and run the Docker container:

```bash
docker build -t ml-portfolio .
docker run -p 8080:8080 ml-portfolio
```

## Customization

To personalize this portfolio with your information:

1. Edit the `PORTFOLIO_DATA` dictionary in `app/main.py` with your details:
   - Name, title, and professional summary
   - Skills and expertise levels
   - Work experience
   - Education and certifications
   - Projects with descriptions and metrics
   - Contact information

2. Replace the sample ML visualizations with your own by modifying the visualization functions in `app/main.py`.

## Deployment Options

This portfolio can be easily deployed to various platforms:

- **Fly.io**: Deploy with `flyctl deploy`
- **Heroku**: Deploy with the Heroku CLI or GitHub integration
- **Render**: Connect your GitHub repository for automatic deployments
- **AWS/GCP/Azure**: Deploy using the provided Dockerfile

## Technologies Used

- **NiceGUI**: Modern Python UI framework
- **Plotly**: Interactive data visualizations
- **Pandas**: Data handling for visualizations
- **Python-dotenv**: Environment configuration
- **Docker**: Containerization for easy deployment

## License

MIT

## Author

[Your Name] - [Your Email]