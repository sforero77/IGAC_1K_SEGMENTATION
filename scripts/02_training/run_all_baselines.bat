@echo off
REM Entrena los tres baselines clásicos (RF, SVM, OBIA) en secuencia.
REM Tiempo estimado total: ~6-10 horas en CPU multinúcleo.
REM Ejecutar desde la raíz del proyecto.

echo ======================================================================
echo Entrenando RandomForest baseline...
echo ======================================================================
python scripts\02_training\train_rf_baseline.py
if %errorlevel% neq 0 goto :error

echo.
echo ======================================================================
echo Entrenando LinearSVM (SGD-hinge) baseline...
echo ======================================================================
python scripts\02_training\train_svm_baseline.py
if %errorlevel% neq 0 goto :error

echo.
echo ======================================================================
echo Entrenando OBIA baseline (SLIC + RandomForest sobre segmentos)...
echo ======================================================================
python scripts\02_training\train_obia_baseline.py
if %errorlevel% neq 0 goto :error

echo.
echo ======================================================================
echo Evaluando los tres baselines sobre el split de test...
echo ======================================================================
python scripts\03_evaluation\evaluate_baseline.py --model rf
python scripts\03_evaluation\evaluate_baseline.py --model svm
python scripts\03_evaluation\evaluate_baseline.py --model obia

echo.
echo OK Todos los baselines entrenados y evaluados.
echo Resultados en results\test_evaluation\{rf,svm,obia}_baseline\
goto :eof

:error
echo ERROR durante la ejecucion. Revisa el log.
exit /b 1
