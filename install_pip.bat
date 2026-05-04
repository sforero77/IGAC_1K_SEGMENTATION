@echo off
echo ============================================
echo INSTALACION RAPIDA CON PIP
echo ============================================

echo.
echo [1/6] Creando ambiente virtual...
python -m venv venv

echo.
echo [2/6] Activando ambiente...
call venv\Scripts\activate.bat

echo.
echo [3/6] Actualizando pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo [4/6] Instalando PyTorch con CUDA 12.1...
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [5/6] Instalando paquetes core...
pip install segmentation-models-pytorch==0.3.3 albumentations==1.3.1 tensorboard==2.15.1 torchmetrics==1.2.1

echo.
echo [6/6] Instalando resto de dependencias...
pip install numpy pandas scipy scikit-learn scikit-image matplotlib seaborn plotly tqdm pyyaml python-dotenv jupyterlab ipykernel ipywidgets opencv-python pillow imgaug

echo.
echo [7/8] Instalando dependencias geoespaciales...
echo NOTA: Si falla, instalar manualmente con wheels pre-compilados
pip install rasterio geopandas fiona shapely pyproj

echo.
echo [8/8] Instalando transformers (opcional)...
pip install transformers timm

echo.
echo ============================================
echo INSTALACION COMPLETADA
echo ============================================
echo.
echo Ejecuta el test:
echo   python test_installation.py
echo.
pause
