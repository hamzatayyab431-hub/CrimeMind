@echo off
title CrimeMind Backend Server
color 0b

echo ===================================================
echo    INITIALIZING CRIMEMIND NEURAL NETWORK...
echo ===================================================
echo.
echo Starting Streamlit backend on port 8501...
start /B streamlit run app.py --server.headless true

echo.
echo Booting up the interface... Please wait.
timeout /t 3 /nobreak >nul

echo.
echo Launching CrimeMind Landing Page...
start index.html

echo.
echo ===================================================
echo  SYSTEM ONLINE. LEAVE THIS WINDOW OPEN IN THE BACKGROUND.
echo  PRESS CTRL+C TO SHUT DOWN THE SERVER WHEN FINISHED.
echo ===================================================
