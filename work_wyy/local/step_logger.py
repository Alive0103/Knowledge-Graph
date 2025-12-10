#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分步骤日志管理系统

功能：
1. 所有日志文件直接保存在trainlog目录下，不创建子目录
2. 每个配置一个日志文件，所有步骤的输出都写入这个文件
3. 所有print输出都会被记录，打印一次就写入一次
4. 自动过滤进度条输出（tqdm）
5. 只记录ES错误，不记录成功的HTTP请求

使用方法：
    from step_logger import StepLogger
    
    logger = StepLogger(base_dir="trainlog", session_name="auto_pipeline")
    
    # 为每个配置创建一个日志文件
    config_log_file = "trainlog/config_1.log"
    
    with logger.step("步骤1: 训练模型", log_file=config_log_file):
        print("训练中...")  # 这个会被记录
        logger.info("训练完成")
    
    with logger.step("步骤2: 测试模型", log_file=config_log_file):
        print("测试中...")  # 这个也会被记录到同一个文件
"""

import os
import sys
import logging
from datetime import datetime
from contextlib import contextmanager
import re
from collections import defaultdict


class TqdmFilter:
    """过滤tqdm进度条输出"""
    def __init__(self):
        self.buffer = ""
        self.in_progress = False
        
    def filter(self, text):
        """过滤进度条相关的输出"""
        if not text:
            return ""
        
        # 按行处理，逐行检查
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过空行
            if not line_stripped:
                continue
            
            # 过滤tqdm进度条模式
            # 格式示例: 
            # "  0%|                                                                                                           | 0/1675 [00:00<?, ?it/s]"
            # "  6%|███▊                                                                | 95/1675 [00:24<06:43,  3.92it/s]"
            # "100%|█████████████████████████████████████████████████████████████████████████████████████████████| 3110/3110 [13:16<00:00,  3.91it/s]"
            # 
            # 关键特征：
            # 1. 包含 "数字%|" 模式
            # 2. 包含 "|数字/数字" 模式
            # 3. 可能包含 "[时间信息it/s]" 模式
            # 
            # 使用更简单的方法：检查是否同时包含这些特征
            is_tqdm_line = False
            
            # 检查是否包含tqdm进度条的关键特征
            has_percent_bar = bool(re.search(r'\d+%\s*\|', line_stripped))  # 包含 "数字%|"
            has_progress = bool(re.search(r'\|\s*\d+/\d+', line_stripped))  # 包含 "|数字/数字"
            
            # 如果同时包含这两个特征，很可能是tqdm进度条
            if has_percent_bar and has_progress:
                # 进一步确认：检查是否包含时间信息（可选）
                has_time_info = bool(re.search(r'\[.*?it/s.*?\]', line_stripped))
                # 或者检查是否只包含进度条相关的字符（数字、%、|、空格、进度条字符）
                # 如果不是训练进度字典（包含'loss'等关键词），则认为是进度条
                is_training_dict = bool(re.search(r"['\"]loss['\"]|['\"]epoch['\"]|['\"]learning_rate['\"]", line_stripped))
                
                if not is_training_dict:
                    is_tqdm_line = True
            
            # 如果不是tqdm进度条，保留这一行
            if not is_tqdm_line:
                filtered_lines.append(line)
        
        # 如果所有行都被过滤掉了，返回空字符串
        if not filtered_lines:
            return ""
        
        # 重新组合行
        result = '\n'.join(filtered_lines)
        # 如果原始文本以换行符结尾，保留它
        if text.endswith('\n') and result:
            result += '\n'
        
        return result


class Tee:
    """同时将输出写入文件和控制台，每次write都立即刷新"""
    def __init__(self, file_path, original_stream, tqdm_filter=None):
        # 以追加模式打开文件，确保多个步骤可以写入同一个文件
        self.file = open(file_path, 'a', encoding='utf-8')
        self.original_stream = original_stream
        self.tqdm_filter = tqdm_filter
        self.file_path = file_path
        
    def write(self, text):
        """写入文本，立即刷新到文件"""
        # 过滤tqdm进度条
        if self.tqdm_filter:
            filtered_text = self.tqdm_filter.filter(text)
            if filtered_text:
                self.file.write(filtered_text)
                self.file.flush()  # 立即刷新，确保不丢失
        else:
            self.file.write(text)
            self.file.flush()  # 立即刷新，确保不丢失
        
        # 始终输出到原始流（控制台）
        self.original_stream.write(text)
        self.original_stream.flush()
        
    def flush(self):
        """刷新文件和控制台"""
        self.file.flush()
        self.original_stream.flush()
    
    def fileno(self):
        """返回原始流的文件描述符，供subprocess使用"""
        return self.original_stream.fileno()
    
    def isatty(self):
        """检查是否是终端"""
        return self.original_stream.isatty()
    
    def readable(self):
        """检查是否可读"""
        return self.original_stream.readable()
    
    def writable(self):
        """检查是否可写"""
        return self.original_stream.writable()
    
    def close(self):
        """关闭文件"""
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()


class StepLogger:
    """分步骤日志管理器"""
    
    def __init__(self, base_dir="trainlog", session_name="session", 
                 log_level=logging.INFO, filter_tqdm=True):
        """
        初始化步骤日志管理器
        
        Args:
            base_dir: 日志保存的基础目录（所有日志文件直接保存在这里，不创建子目录）
            session_name: 会话名称（仅用于摘要日志）
            log_level: 日志级别
            filter_tqdm: 是否过滤tqdm进度条输出
        """
        self.base_dir = base_dir
        self.session_name = session_name
        self.filter_tqdm = filter_tqdm
        
        # 确保基础目录存在
        os.makedirs(base_dir, exist_ok=True)
        
        # 会话开始时间（保存为属性，供外部访问）
        self.session_start_time = datetime.now()
        
        # 步骤计数器
        self.step_counter = 0
        
        # 当前步骤信息
        self.current_step = None
        self.current_step_file = None
        self.current_step_logger = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Tee对象池：为每个日志文件维护一个Tee对象，避免重复创建和关闭
        self.tee_pool = {}  # {file_path: {'stdout': Tee, 'stderr': Tee, 'ref_count': int}}
        
        # tqdm过滤器
        if filter_tqdm:
            self.tqdm_filter = TqdmFilter()
        else:
            self.tqdm_filter = None
        
        # 创建会话摘要日志文件（直接保存在base_dir下）
        self.session_log_file = os.path.join(base_dir, f"{session_name}_summary_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}.log")
        self.session_logger = self._create_logger(
            "session_summary",
            self.session_log_file,
            log_level
        )
        
        self.session_logger.info("=" * 80)
        self.session_logger.info(f"会话开始: {session_name}")
        self.session_logger.info(f"开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.session_logger.info(f"日志目录: {base_dir}")
        self.session_logger.info("=" * 80)
        
        # 配置ES和urllib3日志（只记录错误）
        logging.getLogger('elasticsearch').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.ERROR)
        logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
        logging.getLogger('elasticsearch.trace').setLevel(logging.ERROR)
    
    def _create_logger(self, name, log_file, level=logging.INFO):
        """创建日志记录器"""
        logger = logging.getLogger(f"{self.session_name}.{name}")
        logger.setLevel(level)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 文件处理器（追加模式）
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器（只输出到控制台，不重复记录）
        console_handler = logging.StreamHandler(self.original_stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_tee(self, file_path, stream_type='stdout'):
        """
        获取或创建Tee对象（复用机制）
        
        Args:
            file_path: 日志文件路径
            stream_type: 'stdout' 或 'stderr'
        
        Returns:
            Tee对象
        """
        if file_path not in self.tee_pool:
            # 创建新的Tee对象
            self.tee_pool[file_path] = {
                'stdout': Tee(file_path, self.original_stdout, self.tqdm_filter),
                'stderr': Tee(file_path, self.original_stderr, self.tqdm_filter),
                'ref_count': 0
            }
        
        # 增加引用计数
        self.tee_pool[file_path]['ref_count'] += 1
        
        return self.tee_pool[file_path][stream_type]
    
    def _release_tee(self, file_path):
        """
        释放Tee对象的引用（当引用计数为0时关闭文件）
        
        Args:
            file_path: 日志文件路径
        """
        if file_path in self.tee_pool:
            self.tee_pool[file_path]['ref_count'] -= 1
            
            # 如果引用计数为0，关闭文件
            if self.tee_pool[file_path]['ref_count'] <= 0:
                self.tee_pool[file_path]['stdout'].close()
                self.tee_pool[file_path]['stderr'].close()
                del self.tee_pool[file_path]
    
    @contextmanager
    def step(self, step_name, log_level=logging.INFO, log_file=None):
        """
        步骤上下文管理器
        
        使用示例:
            config_log_file = "trainlog/config_1.log"
            
            with logger.step("步骤1: 训练模型", log_file=config_log_file):
                print("训练中...")  # 会被记录到config_log_file
                logger.info("训练完成")
            
            with logger.step("步骤2: 测试模型", log_file=config_log_file):
                print("测试中...")  # 也会被记录到同一个config_log_file
        
        Args:
            step_name: 步骤名称
            log_level: 日志级别
            log_file: 指定日志文件路径（如果指定，所有步骤输出都写入这个文件）
                     如果不指定，会在base_dir下创建新的日志文件
        """
        self.step_counter += 1
        step_id = f"step_{self.step_counter:02d}"
        
        # 如果指定了日志文件，使用指定的文件；否则创建新的步骤日志文件
        if log_file:
            step_log_file = log_file
        else:
            # 创建步骤日志文件（直接保存在base_dir下）
            safe_step_name = re.sub(r'[^\w\s-]', '', step_name).strip()
            safe_step_name = re.sub(r'[-\s]+', '_', safe_step_name)
            step_log_file = os.path.join(
                self.base_dir,
                f"{step_id}_{safe_step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
        
        # 记录步骤开始
        start_time = datetime.now()
        self.session_logger.info(f"\n[{step_id}] 开始: {step_name}")
        self.session_logger.info(f"  日志文件: {step_log_file}")
        
        # 创建步骤日志记录器
        step_logger = self._create_logger(step_id, step_log_file, log_level)
        step_logger.info("=" * 80)
        step_logger.info(f"步骤: {step_name}")
        step_logger.info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        step_logger.info("=" * 80)
        
        # 保存当前状态
        self.current_step = step_name
        self.current_step_file = step_log_file
        self.current_step_logger = step_logger
        
        # 获取或创建Tee对象（复用机制）
        tee_stdout = self._get_tee(step_log_file, 'stdout')
        tee_stderr = self._get_tee(step_log_file, 'stderr')
        
        # 重定向stdout和stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = tee_stdout
        sys.stderr = tee_stderr
        
        try:
            yield step_logger
        except Exception as e:
            # 记录错误
            error_time = datetime.now()
            step_logger.error("=" * 80)
            step_logger.error(f"步骤执行出错: {step_name}")
            step_logger.error(f"错误时间: {error_time.strftime('%Y-%m-%d %H:%M:%S')}")
            step_logger.error(f"错误类型: {type(e).__name__}")
            step_logger.error(f"错误信息: {str(e)}")
            step_logger.error("=" * 80, exc_info=True)
            
            self.session_logger.error(f"[{step_id}] 失败: {step_name}")
            self.session_logger.error(f"  错误: {type(e).__name__}: {str(e)}")
            
            raise
        finally:
            # 恢复stdout和stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # 释放Tee对象引用（但不立即关闭，因为可能还有其他step在使用）
            self._release_tee(step_log_file)
            
            # 记录步骤结束
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            step_logger.info("=" * 80)
            step_logger.info(f"步骤结束: {step_name}")
            step_logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            step_logger.info(f"耗时: {duration:.2f} 秒")
            step_logger.info("=" * 80)
            
            self.session_logger.info(f"[{step_id}] 完成: {step_name} (耗时: {duration:.2f}秒)")
            
            # 清理
            self.current_step = None
            self.current_step_file = None
            self.current_step_logger = None
    
    def info(self, message):
        """记录信息（使用当前步骤的日志记录器）"""
        if self.current_step_logger:
            self.current_step_logger.info(message)
        else:
            self.session_logger.info(message)
    
    def warning(self, message):
        """记录警告"""
        if self.current_step_logger:
            self.current_step_logger.warning(message)
        else:
            self.session_logger.warning(message)
    
    def error(self, message):
        """记录错误"""
        if self.current_step_logger:
            self.current_step_logger.error(message)
        else:
            self.session_logger.error(message)
    
    def close(self):
        """关闭日志管理器，关闭所有Tee对象"""
        # 关闭所有Tee对象
        for file_path in list(self.tee_pool.keys()):
            pool_entry = self.tee_pool[file_path]
            pool_entry['stdout'].close()
            pool_entry['stderr'].close()
        self.tee_pool.clear()
        
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds()
        
        self.session_logger.info("=" * 80)
        self.session_logger.info(f"会话结束: {self.session_name}")
        self.session_logger.info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.session_logger.info(f"总耗时: {duration:.2f} 秒")
        self.session_logger.info(f"日志目录: {self.base_dir}")
        self.session_logger.info("=" * 80)
        
        # 关闭所有处理器
        for logger_name in [self.session_logger.name]:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
