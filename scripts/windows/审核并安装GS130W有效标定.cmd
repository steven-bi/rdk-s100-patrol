@echo off
setlocal
chcp 65001 >nul

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0Review-And-Install-GS130WValidatedCalibration.ps1" %*
set "installerExit=%ERRORLEVEL%"

echo.
if "%installerExit%"=="0" (
    echo The review/install tool finished successfully.
) else (
    echo The review/install tool stopped with exit code %installerExit%.
    echo Keep this window open and review the error above.
)
echo.
pause
exit /b %installerExit%
