@echo off
chcp 65001 >nul
title Fruit and Vegetable Image Processing Pipeline
setlocal enabledelayedexpansion

echo ===============================================================================
echo     FRUIT AND VEGETABLE IMAGE RECOGNITION - AUTOMATED PIPELINE
echo ===============================================================================
echo [INFO] Starting End-to-End Pipeline Execution...
echo.

:: Ensure Git is in PATH if installed in local programs
set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"

:: Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python 3.9+.
    pause
    exit /b 1
)

echo [STAGE 0/4] Checking and Installing Python Dependencies...
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install required dependencies.
    pause
    exit /b %ERRORLEVEL%
)
echo [STAGE 0/4] Dependencies are up to date.
echo.

:: STAGE 1: Data Collection / Verification
echo ===============================================================================
echo [STAGE 1/4] Running Data Collection and Setup...
echo ===============================================================================
python src\download_data.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Data collection step failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: STAGE 2: Exploratory Data Analysis (EDA)
echo ===============================================================================
echo [STAGE 2/4] Running Exploratory Data Analysis (EDA)...
echo ===============================================================================
python src\eda.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] EDA step failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: STAGE 3: Image Preprocessing & Augmentation Demo
echo ===============================================================================
echo [STAGE 3/4] Running Image Preprocessing and Augmentation Pipeline...
echo ===============================================================================
python src\run_preprocessing_demo.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Image Preprocessing step failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: STAGE 4: Data Splitting & Leakage Audit
echo ===============================================================================
echo [STAGE 4/4] Running Data Splitting and Leakage Audit...
echo ===============================================================================
python src\data_splitting.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Data Splitting step failed.
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo ===============================================================================
echo                      ALL PIPELINE STAGES COMPLETED!
echo ===============================================================================
echo.
echo Generated Reports:
echo   - Master Project Report:   reports\PROJECT_REPORT.md
echo   - EDA Report:              reports\eda\eda_report.md
echo   - Preprocessing Report:    reports\preprocessing\preprocessing_report.md
echo   - Data Splitting Report:   reports\data_splitting\data_splitting_report.md
echo.
echo Generated Visual Figures:
echo   - EDA Figures:             reports\eda\figures\
echo   - Preprocessing Figures:   reports\preprocessing\figures\
echo   - Data Splitting Figures:  reports\data_splitting\figures\
echo ===============================================================================
echo.
pause
