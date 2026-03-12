import os
import sys
from loguru import logger

def setup_logger(log_dir="logs"):
    """
    配置日志
    Args:
        log_dir: 日志目录
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    # 移除默认处理器
    logger.remove()
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    # 添加文件处理器
    logger.add(
        os.path.join(log_dir, "doc_qa_{time:YYYY-MM-DD}.log"),
        rotation="00:00",  # 每天轮换
        retention="30 days",  # 保留30天
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )
    
    return logger