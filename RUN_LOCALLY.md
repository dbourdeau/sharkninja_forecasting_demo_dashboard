# Run Dashboard Locally - Simple Guide

## Quick Start (2 Steps!)

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Dashboard

**Easiest way (Windows):**
- Just double-click `run_demo.bat`
- That's it! It will:
  - Generate data automatically if needed
  - Start the dashboard
  - Open in your browser

**Or from command line:**
```bash
streamlit run dashboard.py
```

The dashboard will automatically:
- ✅ Generate data if it doesn't exist
- ✅ Open at `http://localhost:8501`
- ✅ Show the forecasting dashboard

## That's It!

Once running, you'll see:
- 📊 **Forecast Tab**: Interactive forecast visualization
- 📈 **Components Tab**: Trend, seasonality breakdown
- 🔗 **Axiom Ray Analysis**: Correlation analysis
- 📉 **Model Performance**: Evaluation metrics

## Need Help?

**Dependencies not installing?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port already in use?**
```bash
streamlit run dashboard.py --server.port 8502
```

**Want to regenerate data?**
Delete the `data` folder and restart the dashboard - it will regenerate automatically!

---

**That's all you need to run it locally!** 🚀

