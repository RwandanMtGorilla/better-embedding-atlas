@echo off
REM Build all JavaScript and Python packages for Windows

setlocal enabledelayedexpansion

echo ========================================
echo Building @embedding-atlas/utils
echo ========================================
cd /d "%~dp0..\packages\utils"
call npm run package
if %errorlevel% neq 0 (
    echo ERROR: Failed to build utils
    exit /b 1
)

echo ========================================
echo Building @embedding-atlas/component
echo ========================================
cd /d "%~dp0..\packages\component"
call npm run package
if %errorlevel% neq 0 (
    echo ERROR: Failed to build component
    exit /b 1
)

echo ========================================
echo Building @embedding-atlas/table
echo ========================================
cd /d "%~dp0..\packages\table"
call npm run package
if %errorlevel% neq 0 (
    echo ERROR: Failed to build table
    exit /b 1
)

echo ========================================
echo Skipping @embedding-atlas/umap-wasm (pre-built)
echo ========================================

echo ========================================
echo Building @embedding-atlas/viewer
echo ========================================
cd /d "%~dp0..\packages\viewer"
call npm run build
if %errorlevel% neq 0 (
    echo ERROR: Failed to build viewer
    exit /b 1
)

echo ========================================
echo All packages built successfully!
echo ========================================

cd /d "%~dp0.."
