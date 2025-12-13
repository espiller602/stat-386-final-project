# Streamlit App Deployment Guide

## Prerequisites

1. Your code is pushed to GitHub at: `https://github.com/espiller602/stat-386-final-project`
2. You have a Streamlit Cloud account (free at https://streamlit.io/cloud)

## Important: Data File Issue

The `clean_playoff_passing.csv` file is currently in `.gitignore` (because it's large). For Streamlit Cloud to work, you have two options:

### Option 1: Include the CSV file (Recommended)

Temporarily remove it from `.gitignore` and commit it:

```bash
# Edit .gitignore to comment out or remove the clean_playoff_passing.csv line
# Then:
git add clean_playoff_passing.csv
git commit -m "Add data file for Streamlit deployment"
git push origin main
```

**Note:** This will add a large file to your repo. If it's too large (>100MB), GitHub may reject it.

### Option 2: Generate data on first run

The app can generate the data if the "data sources" directory exists. However, this directory is also in `.gitignore`. You would need to:
- Include the data sources directory in the repo, OR
- Modify the app to download data from a URL, OR
- Use Streamlit's file uploader to let users upload data

## Deployment Steps

1. **Go to Streamlit Cloud**: https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure your app**:
   - **Repository**: `espiller602/stat-386-final-project`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom name (e.g., `nfl-playoff-predictor`)

5. **Advanced settings** (if needed):
   - Python version: 3.8 or higher
   - The app will automatically use `requirements.txt` from your repo

6. **Click "Deploy"**

7. **Wait for deployment** - Streamlit will:
   - Install dependencies from `requirements.txt`
   - Run your app
   - Provide you with a public URL

## After Deployment

- Your app will be available at: `https://your-app-name.streamlit.app`
- Updates are automatic when you push to the `main` branch
- Check the app logs in Streamlit Cloud dashboard if there are any issues

## Troubleshooting

- **"File not found" errors**: Make sure `clean_playoff_passing.csv` is in the repo or the app can generate it
- **Import errors**: Verify all packages in `requirements.txt` are correct
- **Slow loading**: The app uses `@st.cache_data` which should help, but first load may be slow

