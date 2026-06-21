@echo off
title AI Product Recommender Runner
echo Starting AI Product Recommendation System...
echo.

:: Change directory to the project folder
cd /d "C:\Users\LOHITH REDDY K\.gemini\antigravity\scratch\ai-product-recommender"

:: Run the Streamlit application
python -m streamlit run app.py

:: If the app stops, keep the window open so they can see any errors
pause
