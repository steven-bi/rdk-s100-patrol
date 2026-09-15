@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%Pull-RdkPatrol.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo 拉回未成功，错误码：%EXIT_CODE%
pause
exit /b %EXIT_CODE%
