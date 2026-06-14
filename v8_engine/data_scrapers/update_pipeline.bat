@echo off
echo Starting Data Pipeline...
echo ===================================
echo [1/2] Running Transfermarkt Scraper (Squad Values)
C:\Users\13263\AppData\Local\Python\pythoncore-3.14-64\python.exe transfermarkt_scraper.py
if %errorlevel% neq 0 (
    echo [ERROR] Transfermarkt Scraper failed!
    exit /b %errorlevel%
)

echo.
echo [2/2] Running FBref Scraper (Tactical Data)
C:\Users\13263\AppData\Local\Python\pythoncore-3.14-64\python.exe fbref_scraper.py
if %errorlevel% neq 0 (
    echo [ERROR] FBref Scraper failed!
    exit /b %errorlevel%
)

echo.
echo ===================================
echo Pipeline completed successfully! Data is ready for V8 Engine prediction.
pause
