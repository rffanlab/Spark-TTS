# spark_tts_gui.py

import sys
import os
import logging
from pathlib import Path
import re # Import regular expression module for more advanced filtering if needed

# --- PySide6 Imports ---
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QTextEdit, QPushButton, QFileDialog, QLineEdit,
        QSlider, QComboBox, QMessageBox, QProgressDialog, QSizePolicy, QGroupBox
    )
    from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
    from PySide6.QtGui import QDesktopServices # For opening file location
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
except ImportError:
    print("错误：缺少 PySide6 库。")
    print("请使用 pip install pyside6 安装它。")
    sys.exit(1)

# --- Backend Import ---
# This block attempts the import and provides guidance if it fails.
# It relies on spark_tts_backend.py being in the same directory or Python path.
try:
    # Explicitly add the script's directory to sys.path to help find the backend
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    print(f"尝试从 '{script_dir}' 导入后端模块...") # Debug print

    from spark_tts_backend import initialize_model, run_tts, UI_LEVELS_MAP
    print("后端模块 'spark_tts_backend.py' 导入成功。") # Debug print
except ImportError as e:
    print("\n" + "="*60)
    print(f"错误：无法导入后端模块 'spark_tts_backend.py'。")
    print(f"具体错误: {e}")
    print("\n请确保以下几点：")
    print("1. `spark_tts_backend.py` 文件与 `spark_tts_gui.py` 在同一个目录下。")
    print("2. 您是从包含这两个文件的目录运行 `python spark_tts_gui.py` 的。")
    print("3. `spark_tts_backend.py` 文件本身没有语法错误或未满足的依赖项")
    print("   (特别是来自 `cli` 和 `sparktts` 的导入)。")
    print("4. 检查 `spark_tts_backend.py` 中是否有打印 'FATAL ERROR' 的消息。")
    print("="*60 + "\n")
    # Keep the application running briefly to show the error in a message box
    # before exiting.
    app_temp = QApplication.instance() # Get instance if exists
    if not app_temp:
         app_temp = QApplication(sys.argv) # Create if needed for msgbox
    QMessageBox.critical(None, "启动错误", "无法加载必要的后端代码 (spark_tts_backend.py)。\n请查看控制台输出获取详细信息。")
    sys.exit(1)


# Configure logging (can share configuration with backend if needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- TTS Worker Thread ---
class TTSWorker(QThread):
    """Runs TTS inference in a separate thread to avoid blocking the GUI."""
    # Signals must be defined as class attributes
    finished = Signal(str)  # Emits the path to the generated audio file or None on failure
    error = Signal(str)     # Emits error messages
    progress = Signal(str)  # Emits progress messages (optional)

    def __init__(self, model, args_dict):
        super().__init__()
        self.model = model
        self.args = args_dict
        self._is_cancelled = False

    def run(self):
        try:
            logging.info(f"TTS 线程开始，参数: {self.args}")
            self.progress.emit("正在初始化推理...") # Example progress update
            if self._is_cancelled:
                 self.error.emit("任务在开始前被取消。")
                 return

            # Run the actual TTS function from the backend
            result_path = run_tts(self.model, **self.args)

            if self._is_cancelled:
                 self.error.emit("任务在处理过程中被取消。")
                 # Optionally delete the partially generated file if applicable
                 if result_path and os.path.exists(result_path):
                     try:
                         os.remove(result_path)
                         logging.info(f"已删除取消任务生成的文件: {result_path}")
                     except OSError as e:
                         logging.warning(f"无法删除取消任务的文件 '{result_path}': {e}")
                 return

            # Emit result or error based on backend function's return
            if result_path:
                self.finished.emit(result_path)
            else:
                # Pass a more specific error message if possible (backend now returns None on error)
                 self.error.emit("TTS 推理失败。请查看控制台日志获取详细错误信息。")

        except FileNotFoundError as e:
             logging.exception("TTS 线程出错 - 文件未找到")
             self.error.emit(f"文件错误: {e}")
        except Exception as e:
            logging.exception("TTS 线程执行时发生意外错误") # Log traceback
            self.error.emit(f"TTS 执行时发生意外错误: {e}")

    def cancel(self):
        self._is_cancelled = True
        logging.info("TTS Worker 已标记为取消。")


# --- Main Application Window ---
class SparkTTS_GUI(QMainWindow):
    def __init__(self, model, model_dir):
        super().__init__()
        self.model = model
        self.model_dir = model_dir
        self.last_generated_audio_path = None
        self.last_generated_filename = None # <-- Added: Store filename for display
        self.tts_thread = None
        self.progress_dialog = None

        self.setWindowTitle(f"SparkTTS 图形界面 (模型: {Path(model_dir).name})")
        self.setGeometry(100, 100, 750, 600) # Adjusted size

        # --- Audio Playback Setup ---
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput() # Required for volume/muting etc. on some platforms
        self.player.setAudioOutput(self.audio_output)
        # Connect signals for playback status and errors
        self.player.errorOccurred.connect(self.media_player_error)
        self.player.mediaStatusChanged.connect(self.media_status_changed)
        self.player.playbackStateChanged.connect(self.playback_state_changed)


        # --- Central Widget and Main Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # --- Tabs ---
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # --- Create Tabs ---
        # Pass `self` so tabs can potentially access main window methods if needed
        self.create_voice_clone_tab()
        self.create_voice_creation_tab()

        # --- Global Output/Player Controls ---
        # Moved player controls outside tabs for global access
        self.output_group = QGroupBox("播放控制")
        self.output_layout = QHBoxLayout()
        self.output_group.setLayout(self.output_layout)

        self.play_button = QPushButton("▶️ 播放")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_audio)

        self.stop_button = QPushButton("⏹️ 停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_audio)

        self.output_status_label = QLabel("点击生成按钮开始合成。")
        self.output_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.open_folder_button = QPushButton("打开文件夹")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.setToolTip("打开包含最后生成文件的文件夹")
        self.open_folder_button.clicked.connect(self.open_output_folder)


        self.output_layout.addWidget(self.play_button)
        self.output_layout.addWidget(self.stop_button)
        self.output_layout.addWidget(self.output_status_label, 1) # Stretch factor
        self.output_layout.addWidget(self.open_folder_button)

        self.main_layout.addWidget(self.output_group)


        # --- Status Bar ---
        self.statusBar().showMessage("模型加载成功，准备就绪。")


    # --- Voice Clone Tab ---
    def create_voice_clone_tab(self):
        tab_clone = QWidget()
        layout_clone = QVBoxLayout(tab_clone)
        layout_clone.setAlignment(Qt.AlignmentFlag.AlignTop) # Content aligns top

        # Input Text Group
        text_group = QGroupBox("要合成的文本")
        text_layout = QVBoxLayout()
        self.text_input_clone = QTextEdit()
        self.text_input_clone.setPlaceholderText("在此输入需要转换成语音的文字 (换行符和制表符将被替换为空格)...")
        self.text_input_clone.setMinimumHeight(80) # Min height
        text_layout.addWidget(self.text_input_clone)
        text_group.setLayout(text_layout)
        layout_clone.addWidget(text_group)


        # Prompt Audio Group
        prompt_group = QGroupBox("参考设置 (声音克隆)")
        prompt_layout = QVBoxLayout()

        # Prompt Audio Selection
        prompt_layout.addWidget(QLabel("参考音频 (用于克隆音色):"))
        hbox_prompt_audio = QHBoxLayout()
        self.prompt_audio_path_edit = QLineEdit()
        self.prompt_audio_path_edit.setPlaceholderText("选择或拖放 .wav 或 .mp3 文件")
        self.prompt_audio_path_edit.setReadOnly(True)
        btn_select_audio = QPushButton("选择文件...")
        btn_select_audio.setToolTip("选择一个本地音频文件作为声音克隆的参考")
        btn_select_audio.clicked.connect(self.select_prompt_audio)
        hbox_prompt_audio.addWidget(self.prompt_audio_path_edit)
        hbox_prompt_audio.addWidget(btn_select_audio)
        prompt_layout.addLayout(hbox_prompt_audio)

        # Prompt Text (Optional)
        prompt_layout.addWidget(QLabel("参考音频对应的文本 (可选，推荐用于同语言克隆):"))
        self.prompt_text_input_clone = QTextEdit()
        self.prompt_text_input_clone.setPlaceholderText("如果提供了参考音频，在此输入其对应的文本内容 (换行符和制表符将被替换为空格)...")
        self.prompt_text_input_clone.setMaximumHeight(60) # Max height
        prompt_layout.addWidget(self.prompt_text_input_clone)
        prompt_group.setLayout(prompt_layout)
        layout_clone.addWidget(prompt_group)

        layout_clone.addStretch(1) # Pushes the button to the bottom

        # Generate Button
        self.btn_generate_clone = QPushButton("🚀 开始声音克隆")
        self.btn_generate_clone.setStyleSheet("QPushButton { padding: 10px; font-size: 16px; }")
        self.btn_generate_clone.setToolTip("使用上方输入的文本和选择的参考音频生成语音")
        self.btn_generate_clone.clicked.connect(self.run_voice_clone)
        layout_clone.addWidget(self.btn_generate_clone)


        self.tabs.addTab(tab_clone, "声音克隆")

    # --- Voice Creation Tab ---
    def create_voice_creation_tab(self):
        tab_create = QWidget()
        layout_create = QVBoxLayout(tab_create)
        layout_create.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Input Text Group
        text_group = QGroupBox("要合成的文本")
        text_layout = QVBoxLayout()
        self.text_input_create = QTextEdit()
        self.text_input_create.setPlaceholderText("在此输入需要转换成语音的文字 (换行符和制表符将被替换为空格)...")
        self.text_input_create.setMinimumHeight(100)
        # Example Text Button
        # btn_example_text = QPushButton("加载示例")
        # btn_example_text.clicked.connect(lambda: self.text_input_create.setText("你可以通过调整音高和语速等参数，生成一个定制化的声音。"))
        # text_layout.addWidget(btn_example_text, alignment=Qt.AlignmentFlag.AlignRight)
        text_layout.addWidget(self.text_input_create)
        text_group.setLayout(text_layout)
        layout_create.addWidget(text_group)


        # Parameters Group
        params_group = QGroupBox("声音参数调整")
        params_layout = QHBoxLayout()

        # Gender Selection
        gender_layout = QVBoxLayout()
        gender_layout.addWidget(QLabel("选择性别:"))
        self.gender_combo = QComboBox()
        # Store 'male'/'female' as user data for easy retrieval
        self.gender_combo.addItem("男声", "male")
        self.gender_combo.addItem("女声", "female")
        gender_layout.addWidget(self.gender_combo)
        gender_layout.addStretch()
        params_layout.addLayout(gender_layout)


        # Slider creation helper function
        def create_slider_group(label, min_val, max_val, default_val, tick_interval=1):
            slider_layout = QVBoxLayout()
            slider_layout.addWidget(QLabel(label))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(tick_interval)
            # Add value label feedback (Optional but nice)
            value_label = QLabel(f"{default_val}")
            slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label, alignment=Qt.AlignmentFlag.AlignCenter) # Center the value
            slider_layout.addStretch()
            return slider_layout, slider

        # Pitch Slider
        pitch_group_layout, self.pitch_slider = create_slider_group("音高 (1低 - 5高):", 1, 5, 3)
        # Add labels for slider ends
        hbox_pitch_labels = QHBoxLayout()
        hbox_pitch_labels.addWidget(QLabel("低"))
        hbox_pitch_labels.addStretch()
        hbox_pitch_labels.addWidget(QLabel("高"))
        pitch_group_layout.insertLayout(2, hbox_pitch_labels) # Insert labels below slider
        params_layout.addLayout(pitch_group_layout)


        # Speed Slider
        speed_group_layout, self.speed_slider = create_slider_group("语速 (1慢 - 5快):", 1, 5, 3)
        # Add labels for slider ends
        hbox_speed_labels = QHBoxLayout()
        hbox_speed_labels.addWidget(QLabel("慢"))
        hbox_speed_labels.addStretch()
        hbox_speed_labels.addWidget(QLabel("快"))
        speed_group_layout.insertLayout(2, hbox_speed_labels) # Insert labels below slider
        params_layout.addLayout(speed_group_layout)

        params_group.setLayout(params_layout)
        layout_create.addWidget(params_group)

        layout_create.addStretch(1) # Pushes the button to the bottom

        # Generate Button
        self.btn_generate_create = QPushButton("✨ 开始声音创建")
        self.btn_generate_create.setStyleSheet("QPushButton { padding: 10px; font-size: 16px; }")
        self.btn_generate_create.setToolTip("根据上方输入的文本和调整的参数生成语音")
        self.btn_generate_create.clicked.connect(self.run_voice_creation)
        layout_create.addWidget(self.btn_generate_create)

        self.tabs.addTab(tab_create, "声音创建")


    # --- Text Cleaning Function ---
    def clean_text(self, input_text):
        """Removes or replaces unwanted characters like \n, \t."""
        if not input_text:
            return ""
        # Replace \n and \t with a space
        cleaned = input_text.replace('\n', ' ').replace('\t', ' ')
        # Optional: Replace multiple spaces with a single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        # Optional: Add more cleaning rules here if needed
        # e.g., remove specific symbols, etc.
        return cleaned

    # --- Event Handlers ---
    @Slot()
    def select_prompt_audio(self):
        """Opens a file dialog to select the prompt audio."""
        # Use the last directory if available, otherwise default to home
        start_dir = os.path.dirname(self.prompt_audio_path_edit.text()) if self.prompt_audio_path_edit.text() else str(Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频文件",
            start_dir,
            "音频文件 (*.wav *.mp3 *.flac);;所有文件 (*)" # Filter for common audio types
        )
        if file_path:
            self.prompt_audio_path_edit.setText(file_path)
            logging.info(f"已选择参考音频: {file_path}")
            self.statusBar().showMessage(f"已选择参考音频: {Path(file_path).name}", 5000) # Show for 5 secs

    @Slot()
    def run_voice_clone(self):
        """Prepares arguments and starts TTS thread for voice cloning."""
        raw_text = self.text_input_clone.toPlainText().strip()
        raw_prompt_text = self.prompt_text_input_clone.toPlainText().strip()
        prompt_speech = self.prompt_audio_path_edit.text().strip()

        # --- Clean Text Inputs ---
        text = self.clean_text(raw_text)
        prompt_text = self.clean_text(raw_prompt_text) # Also clean prompt text

        # --- Input Validation ---
        if not text:
            QMessageBox.warning(self, "输入缺失", "请输入要合成的文本。")
            self.text_input_clone.setFocus()
            return
        if not prompt_speech:
            QMessageBox.warning(self, "输入缺失", "请选择一个参考音频文件进行声音克隆。")
            # Maybe visually indicate the button or field?
            return
        if not os.path.exists(prompt_speech):
             QMessageBox.critical(self, "文件错误", f"选择的参考音频文件不存在或无法访问:\n{prompt_speech}")
             self.prompt_audio_path_edit.clear() # Clear invalid path
             return

        # Prepare arguments dictionary for the backend function
        args = {
            "text": text, # Use cleaned text
            "prompt_speech_path": prompt_speech,
            "prompt_text": prompt_text if len(prompt_text) >= 2 else None, # Use cleaned prompt text
            "gender": None,
            "pitch": None,
            "speed": None,
        }

        self.start_tts_thread(args)

    @Slot()
    def run_voice_creation(self):
        """Prepares arguments and starts TTS thread for voice creation."""
        raw_text = self.text_input_create.toPlainText().strip()
        if not raw_text: # Check raw text first
            QMessageBox.warning(self, "输入缺失", "请输入要合成的文本。")
            self.text_input_create.setFocus()
            return

        # --- Clean Text Input ---
        text = self.clean_text(raw_text)
        if not text: # Check if text becomes empty after cleaning
            QMessageBox.warning(self, "输入无效", "清理后的文本为空，请输入有效内容。")
            self.text_input_create.setFocus()
            return


        # Get gender value ('male' or 'female') from combo box user data
        gender = self.gender_combo.currentData()

        pitch_level = self.pitch_slider.value()
        speed_level = self.speed_slider.value()

        # Map slider GUI values (1-5) to actual float values expected by the model
        pitch = UI_LEVELS_MAP.get(pitch_level, 1.0) # Default to 1.0 if key not found
        speed = UI_LEVELS_MAP.get(speed_level, 1.0)

        # Prepare arguments dictionary for the backend function
        args = {
            "text": text, # Use cleaned text
            "prompt_speech_path": None,
            "prompt_text": None,
            "gender": gender,
            "pitch": pitch,
            "speed": speed,
        }

        self.start_tts_thread(args)

    # --- TTS Thread Management ---
    def start_tts_thread(self, args_dict):
        """Handles starting the TTS worker thread and updating the UI."""
        if self.tts_thread and self.tts_thread.isRunning():
            QMessageBox.warning(self, "正在处理", "另一个 TTS 任务正在进行中，请稍候。")
            return

        # Disable generate buttons and update global status/player controls
        self.set_ui_busy(True)
        # Reset status before starting
        self.last_generated_filename = None # Clear previous filename display info
        self.output_status_label.setText("⏳ 正在初始化...")
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.open_folder_button.setEnabled(False)
        self.statusBar().showMessage("正在启动 TTS 任务...")

        # --- Progress Dialog ---
        self.progress_dialog = QProgressDialog("正在生成语音...", "取消", 0, 0, self)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setWindowTitle("请稍候")
        self.progress_dialog.canceled.connect(self.cancel_tts)
        self.progress_dialog.show()


        # --- Create and Start Worker Thread ---
        self.tts_thread = TTSWorker(self.model, args_dict)
        self.tts_thread.finished.connect(self.on_tts_finished)
        self.tts_thread.error.connect(self.on_tts_error)
        self.tts_thread.progress.connect(self.update_progress)
        self.tts_thread.finished.connect(self.tts_thread.deleteLater)
        self.tts_thread.error.connect(self.tts_thread.deleteLater)
        self.tts_thread.start()
        logging.info("TTS worker thread started.")

    @Slot(str)
    def update_progress(self, message):
        """Updates the progress dialog message."""
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)
            logging.info(f"Progress Update: {message}")

    @Slot(str)
    def on_tts_finished(self, result_path):
        """Handles successful completion of the TTS thread."""
        self.last_generated_audio_path = result_path
        self.last_generated_filename = Path(result_path).name # <-- Store filename
        logging.info(f"TTS 成功完成，文件: {result_path}")

        if self.progress_dialog:
             self.progress_dialog.close()
             self.progress_dialog = None

        # <-- Update status using the stored filename -->
        self.output_status_label.setText(f"✅ 生成成功: {self.last_generated_filename}")
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.open_folder_button.setEnabled(True)
        self.statusBar().showMessage(f"生成成功: {self.last_generated_filename}", 10000)

        self.set_ui_busy(False)
        self.tts_thread = None

    @Slot(str)
    def on_tts_error(self, error_message):
        """Handles errors reported by the TTS thread."""
        logging.error(f"TTS 任务失败: {error_message}")

        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        QMessageBox.critical(self, "TTS 错误", f"语音生成过程中发生错误:\n{error_message}")
        # <-- Clear filename on error -->
        self.last_generated_filename = None
        self.output_status_label.setText("❌ 生成失败。")
        self.statusBar().showMessage(f"错误: {error_message}", 10000)

        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        # Keep open folder button potentially active if default dir exists
        self.open_folder_button.setEnabled(os.path.isdir("tts_results"))

        self.set_ui_busy(False)
        self.tts_thread = None

    @Slot()
    def cancel_tts(self):
        """Attempts to cancel the running TTS thread."""
        logging.info("用户请求取消 TTS 任务。")
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.cancel()
            # Update UI immediately
            self.output_status_label.setText("正在取消...") # <-- Display cancellation attempt
            self.statusBar().showMessage("正在尝试取消任务...")
        else:
            logging.warning("无法取消：没有正在运行的 TTS 任务。")


    def set_ui_busy(self, busy):
        """Enable/disable UI elements during processing."""
        self.btn_generate_clone.setEnabled(not busy)
        self.btn_generate_create.setEnabled(not busy)


    # --- Audio Playback ---
    @Slot()
    def play_audio(self):
        """Plays the last generated audio file."""
        if self.last_generated_audio_path and os.path.exists(self.last_generated_audio_path):
            logging.info(f"尝试播放: {self.last_generated_audio_path}")
            self.player.stop()
            audio_url = QUrl.fromLocalFile(os.path.abspath(self.last_generated_audio_path))
            self.player.setSource(audio_url)
            self.player.play()
            # Initial status set in playback_state_changed
            self.stop_button.setEnabled(True)
            self.play_button.setEnabled(False)
        else:
            QMessageBox.warning(self, "播放错误", "未找到有效的音频文件。\n请先生成一个音频。")
            logging.warning("无法播放，音频文件路径无效或不存在。")
            self.play_button.setEnabled(False)

    @Slot()
    def stop_audio(self):
        """Stops audio playback."""
        if self.player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.player.stop()
            logging.info("音频播放已停止。")
            # Status update handled by playback_state_changed
            self.stop_button.setEnabled(False)
            self.play_button.setEnabled(self.last_generated_filename is not None and os.path.exists(self.last_generated_audio_path))


    def _update_playback_status(self, status_prefix):
        """Helper to update status label while preserving filename."""
        if self.last_generated_filename:
             # Combine prefix with the stored filename
             self.output_status_label.setText(f"{status_prefix} ({self.last_generated_filename})")
        else:
             # Fallback if filename isn't set (e.g., before first generation)
             self.output_status_label.setText(status_prefix)


    @Slot(QMediaPlayer.MediaStatus)
    def media_status_changed(self, status):
        """Handles changes in the media player's status."""
        current_text = self.output_status_label.text()
        if "❌" in current_text: return # Don't overwrite error

        if status == QMediaPlayer.MediaStatus.LoadingMedia:
             self._update_playback_status("正在加载...")
        elif status == QMediaPlayer.MediaStatus.LoadedMedia:
             if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
                 self._update_playback_status("已加载")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            logging.info("播放到达文件末尾.")
            self._update_playback_status("✅ 播放完毕") # <-- Use helper
            self.stop_button.setEnabled(False)
            self.play_button.setEnabled(self.last_generated_filename is not None and os.path.exists(self.last_generated_audio_path))
        elif status == QMediaPlayer.MediaStatus.StalledMedia:
             self._update_playback_status("⚠️ 缓冲/中断")
             logging.warning("音频播放中断 (StalledMedia)")
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            logging.error("媒体状态无效，无法播放文件。")
            self.output_status_label.setText("❌ 无法播放此音频文件。") # Keep simple error
            self.last_generated_filename = None # Clear filename as it's invalid
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)


    @Slot(QMediaPlayer.PlaybackState)
    def playback_state_changed(self, state):
        """Handles changes in the player's playback state (Playing, Paused, Stopped)."""
        logging.debug(f"Playback state changed: {state}")
        current_text = self.output_status_label.text()
        # Avoid overwriting error/finished status unless stopping
        is_error = "❌" in current_text
        is_finished = "✅" in current_text

        if state == QMediaPlayer.PlaybackState.PlayingState:
            if not is_error: # Don't change status if error occurred previously
                 self._update_playback_status("▶️ 正在播放...") # <-- Use helper
                 self.stop_button.setEnabled(True)
                 self.play_button.setEnabled(False)
        elif state == QMediaPlayer.PlaybackState.PausedState:
             if not is_error:
                 self._update_playback_status("⏸️ 已暂停") # <-- Use helper
                 self.stop_button.setEnabled(True)
                 self.play_button.setEnabled(True) # Allow resuming
        elif state == QMediaPlayer.PlaybackState.StoppedState:
             # Only update if not already finished/error, or if user manually stopped
             if not is_finished and not is_error:
                 self._update_playback_status("⏹️ 播放停止") # <-- Use helper
             # Always update buttons on stop
             self.stop_button.setEnabled(False)
             self.play_button.setEnabled(not is_error and self.last_generated_filename is not None and os.path.exists(self.last_generated_audio_path))


    @Slot(QMediaPlayer.Error, str)
    def media_player_error(self, error_code, error_string):
        """Handles errors reported by the media player."""
        logging.error(f"媒体播放器错误 ({error_code}): {error_string}")
        QMessageBox.critical(self, "播放错误", f"无法播放音频文件。\n错误: {error_string}")
        self.output_status_label.setText("❌ 播放失败") # Keep simple error
        self.last_generated_filename = None # Clear filename info
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)


    # --- Utility Slots ---
    @Slot()
    def open_output_folder(self):
        """Opens the directory containing the last generated audio file."""
        target_folder = None
        if self.last_generated_audio_path and os.path.exists(self.last_generated_audio_path):
            target_folder = os.path.dirname(os.path.abspath(self.last_generated_audio_path))
        else:
            default_save_dir = "tts_results"
            if os.path.isdir(default_save_dir):
                target_folder = os.path.abspath(default_save_dir)

        if target_folder:
            logging.info(f"尝试打开文件夹: {target_folder}")
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(target_folder)):
                QMessageBox.warning(self, "打开失败", f"无法自动打开文件夹:\n{target_folder}\n请手动导航到该位置。")
        else:
            QMessageBox.information(self, "无文件/文件夹", "尚未生成任何音频文件，或默认结果文件夹 'tts_results' 不存在。")


    # --- Cleanup on Close ---
    def closeEvent(self, event):
        """Handles window closing event."""
        logging.info("关闭应用程序...")
        self.player.stop()

        if self.tts_thread and self.tts_thread.isRunning():
             reply = QMessageBox.question(self, "任务进行中",
                                          "一个 TTS 任务仍在运行。\n是否要尝试取消并退出？",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
             if reply == QMessageBox.StandardButton.Yes:
                 logging.info("尝试在退出前取消 TTS 任务...")
                 self.cancel_tts()
                 event.accept()
             else:
                 event.ignore()
                 return
        else:
            event.accept()


# --- Main Execution Function ---
def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    model_dir_default = "pretrained_models/Spark-TTS-0.5B"
    model_dir = model_dir_default

    if not os.path.isdir(model_dir):
         msg = f"错误：找不到模型目录 '{model_dir}'。\n\n" \
               f"请执行以下操作之一：\n" \
               f"1. 确保名为 '{Path(model_dir).name}' 的文件夹与此应用程序在同一目录中。\n" \
               f"2. 或修改脚本中的 `model_dir_default` 变量以指向正确的模型路径。"
         QMessageBox.critical(None, "模型目录错误", msg)
         logging.error(f"启动中止：模型目录 '{model_dir}' 未找到。")
         sys.exit(1)

    loading_msg = QMessageBox(QMessageBox.Icon.Information, "请稍候",
                              f"正在加载 SparkTTS 模型...\n(来自: {model_dir})\n这可能需要一些时间，尤其是在第一次运行时。",
                              QMessageBox.StandardButton.NoButton)
    loading_msg.setWindowModality(Qt.WindowModality.ApplicationModal)
    loading_msg.show()
    QApplication.processEvents()

    tts_model = None
    try:
        tts_model = initialize_model(model_dir=model_dir, device_id=0)
        loading_msg.accept()
    except FileNotFoundError as e:
         loading_msg.accept()
         QMessageBox.critical(None, "模型加载失败", f"模型初始化失败：\n{e}\n请确保模型目录内容完整。")
         logging.error(f"模型文件未找到: {e}")
         sys.exit(1)
    except NameError as e:
         loading_msg.accept()
         QMessageBox.critical(None, "模型加载失败", f"模型初始化失败：\n{e}\n似乎 SparkTTS 组件未能正确导入。请检查 `spark_tts_backend.py` 的导入部分和依赖项。")
         logging.error(f"模型类或依赖项缺失: {e}")
         sys.exit(1)
    except Exception as e:
        loading_msg.accept()
        QMessageBox.critical(None, "模型加载失败", f"加载 TTS 模型时发生意外错误:\n{e}\n请查看控制台日志获取详细信息。")
        logging.exception("模型初始化期间发生未知错误")
        sys.exit(1)

    if not tts_model:
         loading_msg.accept()
         QMessageBox.critical(None, "模型加载失败", "模型初始化函数返回无效对象 (None)。无法启动应用程序。")
         logging.error("initialize_model 返回 None，无法继续。")
         sys.exit(1)

    window = SparkTTS_GUI(tts_model, model_dir)
    window.show()
    sys.exit(app.exec())


# --- Entry Point ---
if __name__ == "__main__":
    try:
        from PySide6 import QtCore
        if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
             QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
             QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        logging.info("启用了高 DPI 支持 (若适用)。")
    except Exception as e:
        logging.warning(f"无法设置高 DPI 属性: {e}")

    main()