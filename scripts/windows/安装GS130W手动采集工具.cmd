@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Install-GS130WManualCollector.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo Installation did not complete. Exit code: %EXIT_CODE%
if "%EXIT_CODE%"=="0" echo Installation command completed successfully.
echo This window will remain open so you can take a screenshot.
pause
exit /b %EXIT_CODE%
