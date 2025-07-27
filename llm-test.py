import argparse
import asyncio
import time
import numpy as np
import aiohttp
import json
import base64  # New import
import os      # New import
from typing import List, Dict, Any, Tuple, Union # Updated import
import logging
import sys

# 导入系统监控库
import psutil

# 设置默认参数
DEFAULT_API_URL = "URL"
DEFAULT_MODEL = "modelname"  # 注意: 此模型可能不支持多模态。如果需要测试多模态功能，请根据实际情况调整模型名称。
API_AUTH_TOKEN = "none"  # 你的 token 替换这里

# Configure logging
# Initial configuration to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def completion_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages_content: Union[str, List[Dict[str, Any]]], # Changed type hint to accept string or list of content parts
    max_tokens: int,
    stream: bool = False
) -> Tuple[float, int]:
    """发送请求到OpenAI兼容的API并计算处理时间"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_AUTH_TOKEN}"
    }

    messages_payload = []
    if isinstance(messages_content, str):
        # Text-only prompt
        messages_payload = [{"role": "user", "content": messages_content}]
    elif isinstance(messages_content, list):
        # Multimodal content (list of text/image_url objects following OpenAI Vision API format)
        messages_payload = [{"role": "user", "content": messages_content}]
    else:
        logging.error(f"无效的 messages_content 类型: {type(messages_content)}. 预期 str 或 List[Dict].")
        return 0, 0

    payload = {
        "model": model,
        "messages": messages_payload,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    start_time = time.time()
    tokens_received = 0
    
    try:
        if not stream:
            # 不使用流式输出，主要用于文本输入处理速度测试。
            # 对于多模态非流式请求，返回值中的tokens将为1，表示处理了一个请求单元。
            async with session.post(f"{url}/chat/completions", headers=headers, json=payload, timeout=300) as response:
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                result = await response.json()
                end_time = time.time()
                processing_time = end_time - start_time
                
                # For input test, tokens is a measure of prompt length or input processed.
                # For text-only, it's simple word count.
                if isinstance(messages_content, str):
                    tokens = len(messages_content.split()) # Simple word count as input tokens
                else:
                    # For multimodal non-stream, it's hard to define "input tokens" in a comparable way.
                    # We will return 1 to indicate a single multimodal request was processed.
                    # The focus for multimodal will typically be on output tokens (streaming).
                    tokens = 1 
                return processing_time, tokens
        else:
            # 使用流式输出，测试token生成速度 (适用于文本和多模态生成)
            async with session.post(f"{url}/chat/completions", headers=headers, json=payload, timeout=300) as response:
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                # Use the time when the first token is received as the start of effective generation.
                # The total time for the connection includes network latency and initial processing.
                first_token_received_time = 0 
                
                async for line in response.content:
                    if line.strip():
                        try:
                            # Handling different possible stream data formats (e.g., "data: {json}")
                            if line.startswith(b"data: "):
                                line = line[6:]
                            
                            # Check for [DONE] signal first, which indicates end of stream
                            if line.strip() == b"[DONE]":
                                break
                            
                            data = json.loads(line)
                            # Check for content in choices[0].delta (standard OpenAI stream format)
                            if "choices" in data and data["choices"] and "delta" in data["choices"][0]:
                                if "content" in data["choices"][0]["delta"]:
                                    content = data["choices"][0]["delta"]["content"]
                                    # Only count if content is not empty
                                    if content: 
                                        if first_token_received_time == 0:
                                            first_token_received_time = time.time() # Mark time when first content token arrives
                                        tokens_received += 1
                        except json.JSONDecodeError:
                            # logging.debug(f"Could not decode JSON from stream line: {line.strip()}")
                            pass # Silently ignore malformed lines that aren't valid JSON
                        except Exception as e:
                            logging.error(f"处理流式响应行时发生错误: {e} - 行: {line.strip().decode()}")
                            pass # Continue processing other lines even if one fails
                
                end_time = time.time()
                # Calculate generation time. 
                # If tokens were received, use the time from the first token until the end.
                # If no tokens were received (e.g., empty response, or an error after initial connection), 
                # use the total duration of the request from its start.
                if tokens_received > 0 and first_token_received_time > 0:
                    generation_time = end_time - first_token_received_time
                else:
                    # If no tokens generated (e.g., error or empty response), use total time for this attempt.
                    generation_time = end_time - start_time 
                
                return generation_time, tokens_received
    except aiohttp.ClientError as e:
        logging.error(f"HTTP 客户端错误: {e}")
        return 0, 0
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解码错误: {e}")
        return 0, 0
    except asyncio.TimeoutError:
        logging.error(f"请求超时 (300秒)。")
        return 0, 0
    except Exception as e:
        logging.error(f"请求过程中发生意外错误: {e}")
        return 0, 0

async def run_input_test(threads: int, prompt_length: int, api_url: str, model: str) -> List[float]:
    """测试文本输入处理速度 (tokens/second)"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # 创建随机文本作为提示, 确保长度
        base_prompt = "请详细解释人工智能的历史和发展。"
        # Ensure the prompt is at least prompt_length characters long
        prompt = (base_prompt * (prompt_length // len(base_prompt) + 1))[:prompt_length]

        logging.info(f"文本输入测试: 使用 {threads} 个并发，提示长度 {prompt_length} 字符。")
        
        for _ in range(threads):
            # Pass the prompt string directly to messages_content
            tasks.append(completion_request(session, api_url, model, prompt, 1, stream=False))
        
        results = await asyncio.gather(*tasks)
        
        # 计算每个请求的tokens/second
        tokens_per_second = []
        for processing_time, tokens in results:
            if processing_time > 0 and tokens > 0:
                tokens_per_second.append(tokens / processing_time)
            elif processing_time == 0 and tokens > 0: # If time is 0 but tokens > 0, it's extremely fast.
                tokens_per_second.append(float('inf')) 
            else:
                logging.warning(f"文本输入测试: 无效结果 (时间: {processing_time:.4f}s, Tokens: {tokens})")
        
        return tokens_per_second

async def run_output_test(threads: int, tokens_to_generate: int, api_url: str, model: str) -> Tuple[List[float], float, float, float]:
    """测试文本输出生成速度 (tokens/second)"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # 使用简单的提示来生成指定数量的token
        prompt = "请写一篇关于人工智能的短文，包含尽可能多的内容。"
        
        logging.info(f"文本输出测试: 使用 {threads} 个并发，每个请求生成 {tokens_to_generate} 个token。")

        batch_start_time = time.time()
        for _ in range(threads):
            # Pass the prompt string directly to messages_content
            tasks.append(completion_request(session, api_url, model, prompt, tokens_to_generate, stream=True))
        
        individual_results = await asyncio.gather(*tasks)
        batch_end_time = time.time()
        
        # 计算每个请求的tokens/second
        tokens_per_second = []
        total_tokens = 0
        max_individual_generation_time = 0 # Max time for a single connection to finish its generation
        
        for generation_time, tokens in individual_results:
            if generation_time > 0 and tokens > 0:
                tokens_per_second.append(tokens / generation_time)
                total_tokens += tokens
                max_individual_generation_time = max(max_individual_generation_time, generation_time)
            elif generation_time == 0 and tokens > 0:
                tokens_per_second.append(float('inf')) # Indicate very fast generation for this request
                total_tokens += tokens
            else:
                logging.warning(f"文本输出测试: 无效结果 (时间: {generation_time:.4f}s, Tokens: {tokens})")

        # Calculate total throughput based on the total batch time (start of first request to end of last request)
        total_batch_time = batch_end_time - batch_start_time
        total_throughput = total_tokens / total_batch_time if total_batch_time > 0 else 0
        
        # Calculate theoretical throughput: total tokens divided by the time it took for the slowest single connection
        theoretical_throughput = total_tokens / max_individual_generation_time if max_individual_generation_time > 0 else 0
        
        return tokens_per_second, total_throughput, total_tokens, theoretical_throughput

async def run_multimodal_test(threads: int, image_path: str, text_prompt: str, tokens_to_generate: int, api_url: str, model: str) -> Tuple[List[float], float, float, float]:
    """测试多模态模型输出生成速度 (tokens/second)"""
    if not os.path.exists(image_path):
        logging.error(f"错误: 图片文件不存在于指定路径: {image_path}")
        return [], 0, 0, 0

    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 尝试根据文件扩展名确定MIME类型
        image_ext = os.path.splitext(image_path)[1].lower()
        if image_ext == '.jpg' or image_ext == '.jpeg':
            mime_type = "image/jpeg"
        elif image_ext == '.png':
            mime_type = "image/png"
        elif image_ext == '.gif':
            mime_type = "image/gif"
        elif image_ext == '.webp': # Add webp support
            mime_type = "image/webp"
        else:
            logging.warning(f"未知图片格式 '{image_ext}'。将使用 'application/octet-stream' 作为MIME类型，可能导致API解析问题。")
            mime_type = "application/octet-stream" # Fallback to generic binary

        image_data_uri = f"data:{mime_type};base64,{encoded_image}"

    except Exception as e:
        logging.error(f"读取或编码图片失败: {image_path} - {e}")
        return [], 0, 0, 0

    # 构建多模态消息内容，遵循OpenAI Vision API的messages格式
    messages_content = [
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": image_data_uri}}
    ]

    async with aiohttp.ClientSession() as session:
        tasks = []
        
        logging.info(f"多模态测试: 使用 {threads} 个并发，每个请求生成 {tokens_to_generate} 个token，图片路径: {image_path}")

        batch_start_time = time.time()
        for _ in range(threads):
            # Pass the multimodal content list to messages_content
            tasks.append(completion_request(session, api_url, model, messages_content, tokens_to_generate, stream=True))
        
        individual_results = await asyncio.gather(*tasks)
        batch_end_time = time.time()
        
        tokens_per_second = []
        total_tokens = 0
        max_individual_generation_time = 0
        
        for generation_time, tokens in individual_results:
            if generation_time > 0 and tokens > 0:
                tokens_per_second.append(tokens / generation_time)
                total_tokens += tokens
                max_individual_generation_time = max(max_individual_generation_time, generation_time)
            elif generation_time == 0 and tokens > 0:
                tokens_per_second.append(float('inf')) # Indicate very fast generation for this request
                total_tokens += tokens
            else:
                logging.warning(f"多模态输出测试: 无效结果 (时间: {generation_time:.4f}s, Tokens: {tokens})")

        total_batch_time = batch_end_time - batch_start_time
        total_throughput = total_tokens / total_batch_time if total_batch_time > 0 else 0
        
        theoretical_throughput = total_tokens / max_individual_generation_time if max_individual_generation_time > 0 else 0
        
        return tokens_per_second, total_throughput, total_tokens, theoretical_throughput

async def monitor_system_metrics(interval: int, log_func):
    """
    周期性地收集并记录 CPU 性能指标。
    log_func: 用于记录日志的函数 (e.g., logging.info)
    """
    if not psutil:
        log_func("psutil 库未安装，系统监控已禁用。请运行 pip install psutil")
        return

    log_func("开始系统性能监控 (仅CPU)...")

    try:
        while True:
            cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking call for CPU usage
            cpu_temp_info = "N/A"
            if hasattr(psutil, 'sensors_temperatures') and psutil.sensors_temperatures():
                temps = psutil.sensors_temperatures()
                # Attempt to find CPU core temperatures or a general CPU temperature
                found_temp = False
                for sensor_name, entries in temps.items():
                    # Look for common sensor names indicating CPU or core temperatures
                    if 'cpu' in sensor_name.lower() or 'core' in sensor_name.lower() or 'package' in sensor_name.lower():
                        core_temps_list = [f"{t.current:.1f}°C" for t in entries if t.current is not None]
                        if core_temps_list:
                            cpu_temp_info = f"CPU温度: {', '.join(core_temps_list)}"
                            found_temp = True
                            break
                if not found_temp:
                    cpu_temp_info = "详细CPU温度不可用 (请检查 psutil sensors_temperatures)"
            
            log_func(f"系统指标: CPU 使用率 {cpu_percent:.1f}%, {cpu_temp_info}")

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        log_func("系统性能监控任务已取消。")
    except Exception as e:
        log_func(f"系统性能监控发生错误: {e}")

async def async_main():
    parser = argparse.ArgumentParser(description="测试OpenAI兼容接口的本地大语言模型并发性能")
    parser.add_argument("-st", "--start_threads", type=int, default=10, help="并发线程起始数 (步进测试模式) 或压力测试模式下的固定并发数")
    parser.add_argument("-et", "--end_threads", type=int, default=200, help="并发线程结束数 (仅步进测试模式有效)")
    parser.add_argument("-ts", "--step_threads", type=int, default=10, help="并发线程步进增量 (仅步进测试模式有效)")
    
    parser.add_argument("-pp", "--prompt_length", type=int, default=100, help="用于文本输入测试的提示长度 (字符数)")
    parser.add_argument("-tg", "--tokens_to_generate", type=int, default=100, help="用于文本和多模态输出测试的生成token数")
    parser.add_argument("-u", "--url", type=str, default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="模型名称")
    
    parser.add_argument("-i", "--input", action="store_true", help="运行文本输入测试")
    parser.add_argument("-o", "--output", action="store_true", help="运行文本输出测试")
    parser.add_argument("-mm", "--multimodal", action="store_true", help="运行多模态测试 (需要模型支持图像输入)")
    parser.add_argument("-ip", "--image_path", type=str, default="testphoto.jpg", help="多模态测试使用的图片路径 (例如: testphoto.jpg 或 testphoto.png)")
    parser.add_argument("-mtp", "--multimodal_text_prompt", type=str, default="请描述图片内容并写一段相关的创意短文。", help="多模态测试的文本提示")

    parser.add_argument("-l", "--log_file", type=str, default="performance_test_log.txt", help="输出结果到指定日志文件")
    parser.add_argument("-d", "--duration_hours", type=float, default=0, help="持续运行压力测试的时长 (小时)。0表示运行步进测试模式。")
    
    args = parser.parse_args()

    # Add file handler for logging
    try:
        file_handler = logging.FileHandler(args.log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"测试结果将同时输出到控制台和文件: {args.log_file}")
    except IOError as e:
        logging.error(f"无法创建或写入日志文件 {args.log_file}: {e}. 将只输出到控制台。")

    # If no specific test type is selected, run text input and text output tests by default
    if not args.input and not args.output and not args.multimodal:
        args.input = True
        args.output = True
    
    if args.duration_hours > 0:
        # 压力测试模式
        logging.info(f"进入压力测试模式：持续 {args.duration_hours} 小时，并发数：{args.start_threads}")
        stress_test_duration_seconds = args.duration_hours * 3600
        stress_test_concurrency = args.start_threads # 使用 start_threads 作为压力测试的固定并发数

        # 启动系统性能监控任务 (仅CPU)
        monitor_task = asyncio.create_task(monitor_system_metrics(interval=10, log_func=logging.info))

        all_input_rates = [] # For text input test results
        all_output_rates = [] # For text output test results
        all_multimodal_output_rates = [] # For multimodal output test results

        # These lists will aggregate throughput and tokens from ALL generation tests (text and multimodal)
        all_total_throughputs_combined = []
        all_theoretical_throughputs_combined = []
        all_total_tokens_generated_combined = []

        start_stress_test_time = time.time()
        batch_counter = 0

        logging.info(f"API URL: {args.url}, 模型: {args.model}")
        logging.info(f"文本输入测试提示长度: {args.prompt_length}, 文本/多模态输出生成token数: {args.tokens_to_generate}")

        while time.time() - start_stress_test_time < stress_test_duration_seconds:
            batch_start_time = time.time()
            batch_counter += 1
            logging.info(f"\n{'='*15} 压力测试批次 {batch_counter} {'='*15}")
            remaining_time_seconds = stress_test_duration_seconds - (time.time() - start_stress_test_time)
            logging.info(f"  距离测试结束还有约 {remaining_time_seconds / 3600:.2f} 小时 ({remaining_time_seconds:.0f} 秒)")

            # Run text input test if enabled
            if args.input:
                logging.info("------ 运行文本输入测试 (批次) ------")
                input_results = await run_input_test(
                    stress_test_concurrency,
                    args.prompt_length,
                    args.url,
                    args.model
                )
                if input_results:
                    all_input_rates.extend(input_results)
                    logging.info(f"  批次文本输入平均速率: {np.mean(input_results):.2f} tokens/s")
                else:
                    logging.warning("  批次文本输入测试未返回有效结果。")

            # Run text output test if enabled
            if args.output:
                logging.info("------ 运行文本输出测试 (批次) ------")
                output_results = await run_output_test(
                    stress_test_concurrency,
                    args.tokens_to_generate,
                    args.url,
                    args.model
                )
                if output_results:
                    all_output_rates.extend(output_results[0])
                    all_total_throughputs_combined.append(output_results[1])
                    all_total_tokens_generated_combined.append(output_results[2])
                    all_theoretical_throughputs_combined.append(output_results[3])

                    logging.info(f"  批次文本输出平均生成率: {np.mean(output_results[0]):.2f} tokens/s/conn")
                    logging.info(f"  批次文本总吞吐量: {output_results[1]:.2f} tokens/s")
                else:
                    logging.warning("  批次文本输出测试未返回有效结果。")
            
            # Run multimodal test if enabled
            if args.multimodal:
                logging.info("------ 运行多模态测试 (批次) ------")
                multimodal_results = await run_multimodal_test(
                    stress_test_concurrency,
                    args.image_path,
                    args.multimodal_text_prompt,
                    args.tokens_to_generate,
                    args.url,
                    args.model
                )
                if multimodal_results:
                    all_multimodal_output_rates.extend(multimodal_results[0])
                    all_total_throughputs_combined.append(multimodal_results[1])
                    all_total_tokens_generated_combined.append(multimodal_results[2])
                    all_theoretical_throughputs_combined.append(multimodal_results[3])

                    logging.info(f"  批次多模态平均生成率: {np.mean(multimodal_results[0]):.2f} tokens/s/conn")
                    logging.info(f"  批次多模态总吞吐量: {multimodal_results[1]:.2f} tokens/s")
                else:
                    logging.warning("  批次多模态测试未返回有效结果。")

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            logging.info(f"批次 {batch_counter} 耗时: {batch_duration:.2f} 秒")
            
            # Add a small delay between batches if the batch execution time is too short
            # This helps prevent spinning CPU on very fast APIs or overwhelming client-side resources
            if batch_duration < 1:
                await asyncio.sleep(1 - batch_duration)

        # 压力测试总计结果汇总
        logging.info(f"\n{'='*20} 压力测试总计结果 {'='*20}")
        if all_input_rates:
            logging.info(f"  整体平均文本输入预处理速率: {np.mean(all_input_rates):.2f} tokens/second")
            logging.info(f"  整体文本输入预处理标准差: {np.std(all_input_rates):.2f} tokens/second")
        if all_output_rates:
            logging.info(f"  整体平均文本token生成率 (每连接): {np.mean(all_output_rates):.2f} tokens/second")
            logging.info(f"  整体文本token生成标准差 (每连接): {np.std(all_output_rates):.2f} tokens/second")
        if all_multimodal_output_rates:
            logging.info(f"  整体平均多模态token生成率 (每连接): {np.mean(all_multimodal_output_rates):.2f} tokens/second")
            logging.info(f"  整体多模态token生成标准差 (每连接): {np.std(all_multimodal_output_rates):.2f} tokens/second")

        # Combined generation throughput (text + multimodal)
        if all_total_throughputs_combined:
            logging.info(f"  整体平均总吞吐量 (所有生成测试): {np.mean(all_total_throughputs_combined):.2f} tokens/second")
            overall_total_tokens_sum = sum(all_total_tokens_generated_combined)
            overall_actual_duration = time.time() - start_stress_test_time
            overall_throughput_over_duration = overall_total_tokens_sum / overall_actual_duration if overall_actual_duration > 0 else 0
            logging.info(f"  总测试时长内实际吞吐量 (所有生成测试): {overall_throughput_over_duration:.2f} tokens/second")

            if np.mean(all_theoretical_throughputs_combined) > 0:
                overall_efficiency_pct = (np.mean(all_total_throughputs_combined) / np.mean(all_theoretical_throughputs_combined)) * 100
                logging.info(f"  整体平均效率系数 (所有生成测试): {overall_efficiency_pct:.2f}%")
            else:
                logging.warning("无法计算整体平均效率系数，理论吞吐量为0。")
            logging.info(f"  总生成tokens (所有生成测试): {overall_total_tokens_sum}")
        else:
            logging.warning("未运行任何输出测试或所有请求失败，无法计算整体输出指标。")
        
        # Cancel and wait for the system monitor task to finish
        monitor_task.cancel()
        await monitor_task 

    else:
        # 步进测试模式 (原有的逻辑)
        logging.info(f"开始步进性能测试，并发数从 {args.start_threads} 到 {args.end_threads}，步进 {args.step_threads}。")
        logging.info(f"API URL: {args.url}, 模型: {args.model}")
        logging.info(f"文本输入测试提示长度: {args.prompt_length}, 文本/多模态输出生成token数: {args.tokens_to_generate}")

        for current_threads in range(args.start_threads, args.end_threads + 1, args.step_threads):
            logging.info(f"\n{'='*20} 正在测试并发数: {current_threads} {'='*20}")
            
            input_avg_rate = 0.0
            input_std_dev = 0.0
            output_avg_rate = 0.0
            output_std_dev = 0.0
            text_total_throughput = 0.0
            text_theoretical_throughput = 0.0
            text_scaling_efficiency_pct = 0.0
            text_total_tokens_generated = 0

            multimodal_avg_rate = 0.0
            multimodal_std_dev = 0.0
            multimodal_total_throughput = 0.0
            multimodal_theoretical_throughput = 0.0
            multimodal_scaling_efficiency_pct = 0.0
            multimodal_total_tokens_generated = 0

            if args.input:
                logging.info("------ 运行文本输入测试 ------")
                tokens_per_second_input = await run_input_test(
                    current_threads, 
                    args.prompt_length, 
                    args.url, 
                    args.model
                )
                if tokens_per_second_input:
                    input_avg_rate = np.mean(tokens_per_second_input)
                    input_std_dev = np.std(tokens_per_second_input)
                    logging.info("文本输入测试结果:")
                    logging.info(f"  平均预处理速率: {input_avg_rate:.2f} tokens/second")
                    logging.info(f"  标准差: {input_std_dev:.2f} tokens/second")
                else:
                    logging.warning("文本输入测试未返回有效结果或所有请求失败。")
            
            if args.output:
                logging.info("------ 运行文本输出测试 ------")
                tokens_per_second_output, text_total_throughput, text_total_tokens_generated, text_theoretical_throughput = await run_output_test(
                    current_threads, 
                    args.tokens_to_generate, 
                    args.url, 
                    args.model
                )
                if tokens_per_second_output:
                    output_avg_rate = np.mean(tokens_per_second_output)
                    output_std_dev = np.std(tokens_per_second_output)
                    logging.info("文本输出测试结果:")
                    logging.info(f"  平均token生成率: {output_avg_rate:.2f} tokens/second/connection")
                    logging.info(f"  标准差: {output_std_dev:.2f} tokens/second/connection")
                    logging.info(f"  总token吞吐量: {text_total_throughput:.2f} tokens/second (所有{current_threads}个连接)")
                    if text_theoretical_throughput > 0:
                        logging.info(f"  理论最大吞吐量: {text_theoretical_throughput:.2f} tokens/second (基于最慢连接)")
                        text_scaling_efficiency_pct = (text_total_throughput / text_theoretical_throughput * 100)
                        logging.info(f"  效率系数: {text_scaling_efficiency_pct:.2f}% (实际/理论)")
                    else:
                        logging.warning("  理论最大吞吐量为0，无法计算效率系数。")
                    logging.info(f"  总生成tokens: {text_total_tokens_generated}")
                else:
                    logging.warning("文本输出测试未返回有效结果或所有请求失败。")

            if args.multimodal:
                logging.info("------ 运行多模态测试 ------")
                tokens_per_second_multimodal, multimodal_total_throughput, multimodal_total_tokens_generated, multimodal_theoretical_throughput = await run_multimodal_test(
                    current_threads, 
                    args.image_path,
                    args.multimodal_text_prompt,
                    args.tokens_to_generate, 
                    args.url, 
                    args.model
                )
                if tokens_per_second_multimodal:
                    multimodal_avg_rate = np.mean(tokens_per_second_multimodal)
                    multimodal_std_dev = np.std(tokens_per_second_multimodal)
                    logging.info("多模态测试结果:")
                    logging.info(f"  平均token生成率: {multimodal_avg_rate:.2f} tokens/second/connection")
                    logging.info(f"  标准差: {multimodal_std_dev:.2f} tokens/second/connection")
                    logging.info(f"  总token吞吐量: {multimodal_total_throughput:.2f} tokens/second (所有{current_threads}个连接)")
                    if multimodal_theoretical_throughput > 0:
                        logging.info(f"  理论最大吞吐量: {multimodal_theoretical_throughput:.2f} tokens/second (基于最慢连接)")
                        multimodal_scaling_efficiency_pct = (multimodal_total_throughput / multimodal_theoretical_throughput * 100)
                        logging.info(f"  效率系数: {multimodal_scaling_efficiency_pct:.2f}% (实际/理论)")
                    else:
                        logging.warning("  理论最大吞吐量为0，无法计算效率系数。")
                    logging.info(f"  总生成tokens: {multimodal_total_tokens_generated}")
                else:
                    logging.warning("多模态测试未返回有效结果或所有请求失败。")

            # 每步的总结
            logging.info("\n====== 性能测试总结 ======")
            logging.info(f"  当前模型为：{DEFAULT_MODEL}")
            logging.info(f"  并发连接数: {current_threads}")
            
            if args.input:
                logging.info(f"  文本输入测试 - 平均预处理速率: {input_avg_rate:.2f} tokens/second")
                logging.info(f"  文本输入测试 - 标准差: {input_std_dev:.2f} tokens/second")
            if args.output:
                logging.info(f"  文本输出测试 - 总吞吐量: {text_total_throughput:.2f} tokens/second")
                logging.info(f"  文本输出测试 - 每个连接平均生成速率: {output_avg_rate:.2f} tokens/second")
                logging.info(f"  文本输出测试 - 效率系数: {text_scaling_efficiency_pct:.2f}%")
            if args.multimodal:
                logging.info(f"  多模态测试 - 总吞吐量: {multimodal_total_throughput:.2f} tokens/second")
                logging.info(f"  多模态测试 - 每个连接平均生成速率: {multimodal_avg_rate:.2f} tokens/second")
                logging.info(f"  多模态测试 - 效率系数: {multimodal_scaling_efficiency_pct:.2f}%")
            
            logging.info("建议:")
            if args.output: 
                if current_threads == args.start_threads and args.end_threads > args.start_threads:
                    logging.info("- (文本输出) 建议继续增加并发连接数以探索系统最大吞吐量。")
                if output_avg_rate > 0 and output_std_dev / output_avg_rate > 0.5 and current_threads > 1:
                    logging.info("- (文本输出) 连接间性能差异较大，可能存在资源争用或网络抖动。")
                if current_threads > args.start_threads and text_scaling_efficiency_pct < 80 and text_scaling_efficiency_pct > 0:
                    logging.info("- (文本输出) 系统扩展性可能受限，增加更多连接可能无法线性提升性能。")
                elif current_threads > args.start_threads and text_scaling_efficiency_pct >= 95:
                    logging.info("- (文本输出) 系统具有良好的并发处理能力，扩展效率高。")
            
            if args.multimodal:
                if current_threads == args.start_threads and args.end_threads > args.start_threads:
                    logging.info("- (多模态) 建议继续增加并发连接数以探索系统最大吞吐量。")
                if multimodal_avg_rate > 0 and multimodal_std_dev / multimodal_avg_rate > 0.5 and current_threads > 1:
                    logging.info("- (多模态) 连接间性能差异较大，可能存在资源争用或网络抖动。")
                if current_threads > args.start_threads and multimodal_scaling_efficiency_pct < 80 and multimodal_scaling_efficiency_pct > 0:
                    logging.info("- (多模态) 系统扩展性可能受限，增加更多连接可能无法线性提升性能。")
                elif current_threads > args.start_threads and multimodal_scaling_efficiency_pct >= 95:
                    logging.info("- (多模态) 系统具有良好的并发处理能力，扩展效率高。")
                
            logging.info(f"{'='*50}\n")

if __name__ == "__main__":
    # 使用 asyncio.run() 运行顶层异步函数
    asyncio.run(async_main())

"""
运行注释和运行示例:

脚本说明:
这是一个用于测试兼容 OpenAI API 的本地大语言模型 (LLM) 并发性能的 Python 脚本。
它支持三种测试模式：
1.  **步进测试模式 (默认)**: 逐步增加并发连接数，测试在不同并发下的性能表现。
python llm-test.py -i -o -st 10 -et 300 -ts 10 -pp 500 -tg 500 -l custom_model_test2.log
2.  **压力测试模式**: 以固定的并发数持续运行指定时长，模拟长时间高负载情况，并提供 CPU 功耗和温度监控。
python llm-test.py -d 8 -st 300 -pp 500 -tg 500 -l custom_model_stress_test.log
3.  **多模态测试功能**: 增加了对支持图像输入的多模态模型的测试，通过将图片编码为 base64 字符串发送。
python llm-test.py -i -o -mm -st 10 -et 300 -ts 10 -pp 500 -tg 500 -ip testphoto.jpg -mtp "这张图片里有什么？请详细描述。" -l custom_model_full_test.log

多模态测试注意事项:
API兼容性: 请确保您测试的后端API (例如 http://ks.sligenai.cn:5004/v1) 支持 OpenAI 的 Vision API 消息格式，即 messages 字段中可以包含 { "type": "image_url", "image_url": { "url": "data:image/..." } }。
模型支持: DEFAULT_MODEL  需要是实际支持多模态输入（视觉能力）的模型。如果您的模型不支持，测试将失败或返回错误。
图片文件: 请确保 testphoto.jpg (或您通过 -ip 指定的文件) 存在于脚本运行的相同目录下，且是可读的图片文件 (JPG, PNG, GIF, WEBP等)。

主要功能:
-   **文本输入测试 (-i)**: 测量 API 处理纯文本输入提示的平均速度 (tokens/second)。
-   **文本输出测试 (-o)**: 测量 API 生成纯文本输出 token 的平均速度 (tokens/second/connection) 和总吞吐量 (tokens/second)。
-   **多模态测试 (-mm)**: 测量 API 处理图像和文本组合输入并生成输出 token 的平均速度和总吞吐量。
-   **并发控制**: 可配置起始、结束并发数和步进增量。
-   **压力测试**: 可设置测试持续时长（小时）。
-   **系统监控**: 在压力测试模式下，会实时监控并记录 CPU 使用率和 CPU 温度。
-   **日志输出**: 所有测试结果和监控数据都会同时输出到控制台和指定日志文件。

安装依赖:
```bash
pip install aiohttp numpy psutil
"""
