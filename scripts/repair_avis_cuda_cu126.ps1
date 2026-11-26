$ErrorActionPreference = "Stop"

# This project environment uses PyTorch 2.12 + CUDA 12.6. ONNX Runtime
# 1.21-1.26 GPU wheels are built against CUDA 12.8; use the 1.20 line
# (CUDA 12.x + cuDNN 9) to keep both runtimes compatible in one process.

Write-Host "[AVIS] Checking the installed PyTorch CUDA build..."
$torchCuda = (& python -c "import torch; print(torch.version.cuda)") 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch cannot initialize in a clean Python process. Fix the PyTorch/CUDA or Microsoft Visual C++ runtime installation first. Details: $torchCuda"
}
$torchCuda = ($torchCuda | Select-Object -Last 1).ToString().Trim()
Write-Host "[AVIS] PyTorch CUDA build: $torchCuda"

if ($torchCuda -ne "12.6") {
    throw "This repair script is for PyTorch CUDA 12.6. Detected CUDA '$torchCuda'. Do not install this ONNX Runtime build; use the ONNX Runtime GPU version matching your PyTorch CUDA version."
}

Write-Host "[AVIS] Removing conflicting CPU/GPU ONNX Runtime packages..."
python -m pip uninstall -y onnxruntime onnxruntime-gpu
if ($LASTEXITCODE -ne 0) {
    throw "Could not remove the existing ONNX Runtime packages."
}

Write-Host "[AVIS] Installing ONNX Runtime GPU 1.20.1 (CUDA 12.x / cuDNN 9)..."
python -m pip install --no-cache-dir "onnxruntime-gpu==1.20.1"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install onnxruntime-gpu==1.20.1."
}

Write-Host "[AVIS] Verifying that PyTorch loads before ONNX Runtime..."
python -c "import torch; import onnxruntime as ort; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('ONNX Runtime:', ort.__version__); print('ORT providers:', ort.get_available_providers())"
if ($LASTEXITCODE -ne 0) {
    throw "The CUDA runtime verification failed. Check the error above before starting AVIS."
}

Write-Host "[AVIS] CUDA runtime packages are aligned. Start AVIS with: python .\main.py --avis"
