@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo QSMEF - Pruebas de regresion
 echo ============================================
python -m pip install -r requirements_qsmef.txt
if errorlevel 1 (
  echo.
  echo ERROR: No se pudieron instalar las dependencias.
  pause
  exit /b 1
)
python -m unittest -v test_qsmef_regression.py
set "RESULTADO=%ERRORLEVEL%"
echo.
if "%RESULTADO%"=="0" (
  echo Pruebas finalizadas correctamente.
) else (
  echo ERROR: Una o mas pruebas fallaron.
)
pause
exit /b %RESULTADO%
