# Deployment Guide

This guide provides instructions for deploying the Call Center Volume Forecasting Dashboard to various platforms.

## Prerequisites

- Your code is version controlled (Git repository)
- All dependencies are listed in `requirements.txt`
- The dashboard auto-generates data on first run

## Deployment Options

### Option 1: Streamlit Cloud (Recommended - Easiest & Free)

Streamlit Cloud is the simplest way to deploy Streamlit apps. It's free and automatically deploys from GitHub.

#### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `dashboard.py`
   - Click "Deploy"

3. **Your app will be live at**: `https://YOUR_APP_NAME.streamlit.app`

#### Configuration:
- The `.streamlit/config.toml` file is already configured
- Data will be auto-generated on first run
- No additional configuration needed

---

### Option 2: Docker Deployment

Deploy using Docker for containerized deployment on any platform that supports Docker.

#### Build Docker Image:

```bash
docker build -t call-center-forecast .
```

#### Run Locally:

```bash
docker run -p 8501:8501 call-center-forecast
```

Then visit `http://localhost:8501`

#### Deploy to Docker Hub:

```bash
# Tag the image
docker tag call-center-forecast YOUR_USERNAME/call-center-forecast:latest

# Push to Docker Hub
docker push YOUR_USERNAME/call-center-forecast:latest
```

#### Deploy to Platforms:

**Railway:**
1. Go to [railway.app](https://railway.app)
2. Create new project
3. Select "Deploy from Docker Hub"
4. Enter your Docker Hub image name
5. Set port to 8501

**Render:**
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Select "Docker" as environment
5. Render will detect the Dockerfile automatically

**Fly.io:**
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Follow prompts to deploy

---

### Option 3: Heroku Deployment

Deploy to Heroku using containers or buildpacks.

#### Using Docker on Heroku:

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create YOUR_APP_NAME
   ```

3. **Configure for Docker:**
   ```bash
   echo "web: streamlit run dashboard.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Deploy:**
   ```bash
   heroku container:push web
   heroku container:release web
   heroku open
   ```

---

### Option 4: AWS/Google Cloud/Azure

For cloud provider deployments, use Docker containers with their container services:

- **AWS**: Use AWS App Runner, ECS, or Elastic Beanstalk
- **Google Cloud**: Use Cloud Run (recommended)
- **Azure**: Use Azure Container Instances or App Service

#### Example: Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/call-center-forecast

# Deploy to Cloud Run
gcloud run deploy call-center-forecast \
  --image gcr.io/YOUR_PROJECT/call-center-forecast \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Environment Variables

If you need to configure environment variables:

### Streamlit Cloud
Add secrets in the Streamlit Cloud dashboard under "Secrets"

### Docker
Pass environment variables:
```bash
docker run -p 8501:8501 -e VARIABLE_NAME=value call-center-forecast
```

### Heroku
```bash
heroku config:set VARIABLE_NAME=value
```

---

## Troubleshooting

### Data Generation Issues
- The dashboard auto-generates data on first run
- Ensure `generate_data.py` is included in deployment
- Check file permissions for data directory creation

### Port Issues
- Streamlit Cloud: Uses port 8501 automatically
- Docker: Ensure EXPOSE directive matches port mapping
- Heroku: Use `$PORT` environment variable

### Memory Issues
- Prophet can be memory-intensive
- Consider increasing container memory limits
- Streamlit Cloud free tier has memory limits

### Build Failures
- Check all dependencies in `requirements.txt`
- Prophet requires compilation, ensure build tools are available
- Verify Python version compatibility (3.8+)

---

## Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` is up to date
- [ ] Dashboard auto-generates data
- [ ] Dockerfile tested locally (if using Docker)
- [ ] Environment variables configured (if needed)
- [ ] Deployed and tested on target platform

---

## Post-Deployment

After deployment:
1. Test all dashboard features
2. Verify data generation works
3. Check forecast calculations
4. Monitor resource usage
5. Set up monitoring/alerting if needed

---

## Support

For deployment issues:
- Check platform-specific logs
- Verify all dependencies are installed
- Ensure Python version compatibility
- Review error messages in deployment logs

