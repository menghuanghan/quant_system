import torch
import paddle
from paddleocr import PaddleOCR

print("--- 基础环境检查 ---")
print(f"1. PyTorch CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - 显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"   - 当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

print(f"\n2. Paddle GPU 是否可用: {paddle.is_compiled_with_cuda()}")
if paddle.is_compiled_with_cuda():
    print(f"   - 当前使用设备: {paddle.get_device()}")

print("\n3. OCR 引擎压力测试 (验证显存挂载)...")
try:
    # 强制初始化 OCR 模型到 GPU
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)
    print("   - [成功] OCR 引擎已成功挂载 GPU！")
except Exception as e:
    print(f"   - [失败] OCR 挂载 GPU 出错: {e}")