# GenChem Deployment Configuration

## Streamlit Cloud Deployment

1. **Main App File**: `streamlit_app/app.py`
2. **Requirements**: `requirements.txt`
3. **Config**: `.streamlit/config.toml`

## Quick Deploy to Streamlit Cloud

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `streamlit_app/app.py`
5. Deploy!

## Alternative Deployment Options

### Heroku
- Use the included `Procfile`
- Set buildpack to Python
- Deploy with: `git push heroku main`

### Railway
- Connect GitHub repository
- Set start command: `streamlit run streamlit_app/app.py --server.port=$PORT`

### Render
- Connect GitHub repository  
- Set build command: `pip install -r requirements.txt`
- Set start command: `streamlit run streamlit_app/app.py --server.port=$PORT`

## Environment Variables

For production deployment, you may need to set:
- `PORT`: Server port (usually set automatically)
- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_SERVER_ENABLE_CORS=false`

## Notes

- The app includes graceful fallbacks for missing REINVENT4 dependencies
- GPU acceleration is optional but recommended for training
- Some advanced features may require additional setup in production
