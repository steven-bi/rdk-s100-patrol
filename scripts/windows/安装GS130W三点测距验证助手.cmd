@echo off
chcp 65001 >nul
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Install-GS130WDistanceValidator.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Installation did not complete. Exit code: %EXIT_CODE%
if "%EXIT_CODE%"=="0" echo Installation command completed successfully.
echo This window will remain open so you can take a screenshot.
pause
exit /b %EXIT_CODE%
