@echo off
REM -----------------------------------------------------
REM  SparkTTS GUI 启动脚本 (Windows Batch)
REM -----------------------------------------------------
REM 设置编码为UTF-8，以防路径或输出包含中文时出现乱码 (可选)
chcp 65001 > nul

REM 获取当前批处理脚本所在的目录
set SCRIPT_DIR=%~dp0

REM 切换工作目录到脚本所在目录
REM /d 参数确保可以切换驱动器盘符
cd /d "%SCRIPT_DIR%"

REM 检查虚拟环境的 Python 是否存在
set VENV_PYTHON=.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
    echo 错误：未找到虚拟环境中的 Python 解释器：
    echo %SCRIPT_DIR%%VENV_PYTHON%
    echo.
    echo 请确保您已在此目录下创建了名为 '.venv' 的虚拟环境，
    echo 并且安装了所有必要的依赖项。
    pause
    exit /b 1
)

REM 检查 GUI 脚本是否存在
if not exist "gui.py" (
    echo 错误：未找到 GUI 脚本 "gui.py"。
    echo 请确保此批处理文件与 gui.py 在同一目录下。
    pause
    exit /b 1
)

REM 使用虚拟环境中的 Python 启动 GUI 脚本
echo 正在使用虚拟环境启动 SparkTTS GUI...
echo Python 路径: %VENV_PYTHON%
echo 启动命令: "%VENV_PYTHON%" gui.py
echo.

"%VENV_PYTHON%" gui.py

REM 如果你想在 GUI 关闭后保持命令行窗口打开以查看可能的错误输出，
REM 可以取消下面这行 pause 命令的注释
REM pause