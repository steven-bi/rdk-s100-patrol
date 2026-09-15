@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Start-GS130WManualCapture.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Capture command did not complete. Exit code: %EXIT_CODE%
if "%EXIT_CODE%"=="0" echo Capture command ended safely.
echo This window will remain open so you can take a screenshot.
pause
exit /b %EXIT_CODE%
