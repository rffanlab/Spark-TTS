# spark_tts_backend.py

import os
import torch
import soundfile as sf
import logging
import platform
from datetime import datetime
from typing import Optional # Added for type hinting clarity

# Ensure these imports work based on your project structure.
# You might need to adjust the paths if 'cli' and 'sparktts' aren't
# directly in the same folder or accessible via the Python path.
try:
    from cli.SparkTTS import SparkTTS
    from sparktts.utils.token_parser import LEVELS_MAP_UI
except ImportError as e:
    # Provide a more informative error if these crucial imports fail
    print(f"FATAL ERROR: Could not import SparkTTS components ({e}).")
    print("Please ensure the 'cli' and 'sparktts' directories from the original project")
    print("are in the same directory as this script, or correctly installed/added to PYTHONPATH.")
    # Optionally, raise the error again to stop execution if these are critical
    raise

# Configure logging (optional but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device_id=0):
    """Load the model once at the beginning."""
    logging.info(f"正在从以下位置加载模型: {model_dir}")

    # --- Check if model directory exists ---
    if not os.path.isdir(model_dir):
        logging.error(f"指定的模型目录不存在: {model_dir}")
        raise FileNotFoundError(f"模型目录未找到: {model_dir}")
    # --- Check if required files/subdirs exist (example) ---
    # Add checks for specific files if needed, e.g., config.json
    # config_path = os.path.join(model_dir, "config.json")
    # if not os.path.isfile(config_path):
    #     logging.error(f"模型目录缺少必要文件: {config_path}")
    #     raise FileNotFoundError(f"模型目录缺少必要文件: {config_path}")

    device = None
    try:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            # macOS with MPS support (Apple Silicon)
            device = torch.device(f"mps:{device_id}")
            logging.info(f"检测到 MPS (Apple Silicon)，使用设备: {device}")
        elif torch.cuda.is_available():
            # System with CUDA support
            # Validate device_id
            num_cuda_devices = torch.cuda.device_count()
            if device_id >= num_cuda_devices:
                 logging.warning(f"请求的 CUDA 设备 ID {device_id} 无效 (可用设备: 0-{num_cuda_devices-1}). 将使用 cuda:0。")
                 device_id = 0
            device = torch.device(f"cuda:{device_id}")
            logging.info(f"检测到 CUDA，使用设备: {device}")
        else:
            # Fall back to CPU
            device = torch.device("cpu")
            logging.info("未检测到 GPU 加速，使用 CPU")
    except Exception as e:
        logging.warning(f"检测设备时出错 ({e}), 回退到 CPU")
        device = torch.device("cpu")

    try:
        # --- Ensure SparkTTS can be instantiated ---
        if 'SparkTTS' not in globals():
             logging.error("SparkTTS 类未成功导入。无法初始化模型。")
             raise NameError("SparkTTS class not available due to import error.")

        model = SparkTTS(model_dir, device)
        logging.info("模型加载成功.")
        return model
    except Exception as e:
        logging.error(f"加载模型时出现严重错误: {e}")
        # Log the traceback for detailed debugging
        logging.exception("详细错误信息:")
        raise  # Re-raise the exception to be caught by the GUI

def run_tts(
    model,                   # The loaded SparkTTS model instance
    text: str,               # Text to synthesize
    prompt_text: Optional[str] = None,      # Text corresponding to the prompt audio
    prompt_speech_path: Optional[str] = None, # Path to the prompt audio file
    gender: Optional[str] = None,           # Target gender ('male' or 'female')
    pitch: Optional[float] = None,          # Target pitch adjustment factor (e.g., 1.0 is normal)
    speed: Optional[float] = None,          # Target speed adjustment factor (e.g., 1.0 is normal)
    save_dir: str = "tts_results",          # Directory to save the output audio
):
    """Perform TTS inference and save the generated audio."""
    if not text:
        logging.warning("输入文本为空，跳过 TTS。")
        return None # Indicate failure or skip

    if not model:
         logging.error("TTS 模型无效或未加载，无法执行推理。")
         return None

    logging.info(f"音频将保存到: {save_dir}")
    logging.info(f"输入文本: '{text[:50]}...'") # Log beginning of text
    if prompt_speech_path:
        logging.info(f"使用参考音频: {prompt_speech_path}")
        if not os.path.exists(prompt_speech_path):
            logging.error(f"参考音频文件不存在: {prompt_speech_path}")
            raise FileNotFoundError(f"参考音频文件不存在: {prompt_speech_path}")
    if prompt_text:
        logging.info(f"参考音频文本: '{prompt_text[:50]}...'")
    if gender:
        logging.info(f"指定性别: {gender}")
    if pitch is not None: # Check for None explicitly
        logging.info(f"指定音高值: {pitch}")
    if speed is not None: # Check for None explicitly
        logging.info(f"指定语速值: {speed}")


    # Process prompt_text based on length (consistent with original logic)
    prompt_text_cleaned = None
    if prompt_text is not None and len(prompt_text) >= 2:
        prompt_text_cleaned = prompt_text

    # Ensure the save directory exists
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"创建保存目录失败 '{save_dir}': {e}")
        raise # Stop if we can't save

    # Generate unique filename using timestamp and partial text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize text for use in filename (remove problematic characters)
    safe_text_prefix = "".join(c for c in text[:15] if c.isalnum() or c in (' ', '_')).rstrip()
    if not safe_text_prefix: safe_text_prefix = "audio" # Fallback if text is all symbols
    save_path = os.path.join(save_dir, f"{timestamp}_{safe_text_prefix}.wav")

    logging.info("开始进行 TTS 推理...")

    try:
        # Perform inference and save the output audio
        # Use torch.no_grad() for efficiency during inference
        with torch.no_grad():
            # --- CORRECTED CALL ---
            # Pass arguments using the expected keyword names, likely matching
            # the variable names from the original Gradio function call.
            wav = model.inference(
                text=text,                       # Required text input
                prompt_speech_path=prompt_speech_path, # Path to reference audio
                prompt_text=prompt_text_cleaned, # Text of reference audio
                gender=gender,                   # Target gender (Corrected name)
                pitch=pitch,                     # Target pitch (Corrected name)
                speed=speed,                     # Target speed (Corrected name)
                # Add any other parameters the actual inference method might accept here
            )

        if wav is None:
             logging.error("推理失败，模型返回 None。请检查模型或输入。")
             return None # Indicate failure

        # --- CORRECTED SAMPLING RATE ACCESS ---
        sampling_rate = 16000 # Default value
        try:
            # Attempt 1: Direct attribute access (if model stores it directly)
            if hasattr(model, 'sampling_rate'):
                sampling_rate = model.sampling_rate
                logging.info(f"从 model.sampling_rate 获取到采样率: {sampling_rate}")
            # Attempt 2: Access through a potential config object (common pattern)
            elif hasattr(model, 'config') and hasattr(model.config, 'data') and hasattr(model.config.data, 'sampling_rate'):
                 sampling_rate = model.config.data.sampling_rate
                 logging.info(f"从 model.config.data.sampling_rate 获取到采样率: {sampling_rate}")
            # Attempt 3: Access through 'hps' if it *was* the intended way but maybe nested differently
            elif hasattr(model, 'hps') and hasattr(model.hps, 'data') and hasattr(model.hps.data, 'sampling_rate'):
                sampling_rate = model.hps.data.sampling_rate
                logging.info(f"从 model.hps.data.sampling_rate 获取到采样率: {sampling_rate}")
            # Add more attempts here based on inspection of cli/SparkTTS.py if needed
            else:
                 logging.warning("无法在模型对象中找到 'sampling_rate' 属性。将使用默认值 16000 Hz。")
                 # Check if maybe the wav object itself holds the rate? (Less common)
                 # if hasattr(wav, 'sampling_rate'): sampling_rate = wav.sampling_rate

            # Ensure the obtained rate is an integer
            sampling_rate = int(sampling_rate)

        except Exception as e:
            logging.warning(f"获取采样率时出错 ({e})。将使用默认值 16000 Hz。")
            sampling_rate = 16000

        logging.info(f"将使用采样率 {sampling_rate} 保存音频。")
        sf.write(save_path, wav, samplerate=sampling_rate)
        logging.info(f"音频已成功保存: {save_path}")
        return save_path
    except TypeError as e:
        # Catch TypeError specifically to give a more targeted error message
        logging.error(f"调用 model.inference 时发生类型错误: {e}")
        logging.error("这通常意味着传递给 inference 方法的参数名称或数量不正确。")
        logging.error("请检查 spark_tts_backend.py 中的 run_tts 函数与 cli/SparkTTS.py 中 inference 方法的定义是否匹配。")
        logging.exception("详细错误信息:")
        return None # Indicate failure
    except AttributeError as e:
        # Catch AttributeError specifically for issues like missing 'hps' or other properties
        logging.error(f"访问模型属性时发生错误: {e}")
        logging.error("这可能意味着代码尝试访问一个不存在的模型属性 (例如 'hps', 'config', 或 'sampling_rate')。")
        logging.error("请检查 spark_tts_backend.py 中访问模型属性的代码是否正确。")
        logging.exception("详细错误信息:")
        return None # Indicate failure
    except Exception as e:
        logging.error(f"TTS 推理过程中发生其他错误: {e}")
        logging.exception("详细错误信息:") # Log the full traceback
        # Optionally re-raise or return None/error indicator
        return None # Indicate failure

# Mapping from GUI slider values (1-5) to model expected float values
# Ensure this matches the definition in the original code if LEVELS_MAP_UI is different
# If the original LEVELS_MAP_UI is exactly {1: 0.8, ..., 5: 1.2}, use it directly:
try:
    # Try importing the map directly if it's defined in the original structure
    from sparktts.utils.token_parser import LEVELS_MAP_UI as UI_LEVELS_MAP
    logging.info("使用 sparktts.utils.token_parser 中的 LEVELS_MAP_UI")
except (ImportError, NameError):
    # Fallback if import fails or LEVELS_MAP_UI doesn't exist
    logging.warning("无法从 sparktts.utils.token_parser 导入 LEVELS_MAP_UI。使用默认映射。")
    UI_LEVELS_MAP = {
        1: 0.8,  # Slow/Low
        2: 0.9,
        3: 1.0,  # Normal
        4: 1.1,
        5: 1.2   # Fast/High
    }
    logging.info(f"使用的默认 UI 等级映射: {UI_LEVELS_MAP}")


# Example usage if run directly (for basic testing)
if __name__ == '__main__':
    print("--- spark_tts_backend.py ---")
    print("这是一个后端模块，请运行 spark_tts_gui.py 来启动应用程序。")
    print("可以尝试进行基本模型初始化测试...")
    # Simple test (optional, requires model to be present)
    try:
        # Check if SparkTTS was imported correctly earlier
        if 'SparkTTS' not in globals():
             print("错误：无法测试，因为 SparkTTS 未导入。")
        else:
            test_model_dir = "pretrained_models/Spark-TTS-0.5B" # Adjust if needed
            print(f"尝试从 '{test_model_dir}' 加载模型...")
            test_model = initialize_model(model_dir=test_model_dir)
            if test_model:
                print("模型初始化测试成功。")
                # Optional: Run a very basic TTS test if model loaded
                # print("尝试运行一个简短的 TTS 测试...")
                # result_path = run_tts(test_model, "这是一个后端测试。")
                # if result_path:
                #     print(f"TTS 测试成功，文件保存在: {result_path}")
                # else:
                #     print("TTS 测试运行失败 (可能返回 None)。")
            else:
                 print("模型初始化测试失败 (返回 None)。")
    except FileNotFoundError as e:
        print(f"模型初始化测试失败: {e}")
        print("请确保模型目录存在且路径正确。")
    except NameError as e:
        print(f"模型初始化测试失败: {e}")
        print("请确保 SparkTTS 库及其依赖项已正确安装或在 Python 路径中。")
    except Exception as e:
        print(f"后端测试期间发生意外错误: {e}")