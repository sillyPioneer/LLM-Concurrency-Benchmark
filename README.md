# LLM-Concurrency-Benchmark

A Python script for benchmarking concurrent performance of local Large Language Models (LLMs) with an OpenAI-compatible API, supporting both text and multimodal (image+text) inputs.

## Features

- **Text Input Benchmark:** Measures average API input preprocessing speed (tokens/second).

- **Text Output Benchmark:** Measures concurrent token generation speed (tokens/second/connection) and total throughput.

- **Multimodal Benchmark:** Tests LLMs with image+text input; requires OpenAI Vision API compatibility.

- **Step Testing Mode:** Incrementally increases concurrency from a configured start to end value with custom steps.

- **Stress Testing Mode:** Continuously benchmarks the LLM with a fixed concurrency over a set period.

- **System Monitoring:** Monitors CPU utilization and temperature (if supported) during stress tests.

- **Configurable Logging:** Prints and saves results to a user-specified log file.

- **Easy Customization:** Flexible command-line arguments to tailor tests for any compatible LLM backend.

## Installation

bash

`pip install aiohttp numpy psutil`

## Usage

## 1. Step Testing Mode (default)

Run input and output tests, incrementing concurrency from 10 to 300, step 10:

bash

`python llm-test.py -i -o -st 10 -et 300 -ts 10 -pp 500 -tg 500 -l custom_model_test2.log`

## 2. Stress Testing Mode

Runs for 8 hours at fixed concurrency (e.g., 300 threads):

bash

`python llm-test.py -d 8 -st 300 -pp 500 -tg 500 -l custom_model_stress_test.log`

## 3. Multimodal Testing

Test with image and text input. Make sure `testphoto.jpg` (or your image file) exists in the script directory:

bash

`python llm-test.py -i -o -mm -st 10 -et 300 -ts 10 -pp 500 -tg 500 -ip testphoto.jpg -mtp "Describe the contents of this image." -l custom_model_full_test.log`

## CLI Arguments

| Argument                         | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-i, --input`                    | Run input (prompt ingestion) benchmark                       |
| `-o, --output`                   | Run output (token generation) benchmark                      |
| `-mm, --multimodal`              | Run multimodal (image+text) tests                            |
| `-st, --start_threads`           | Initial concurrency (threads)                                |
| `-et, --end_threads`             | End concurrency for step test                                |
| `-ts, --step_threads`            | Concurrency increment (for step mode)                        |
| `-pp, --prompt_length`           | Prompt length (characters, for input benchmark)              |
| `-tg, --tokens_to_generate`      | Tokens to generate (output & multimodal benchmarks)          |
| `-u, --url`                      | API base URL (OpenAI-compatible)                             |
| `-m, --model`                    | Model name (must match your backend API)                     |
| `-ip, --image_path`              | Path to test image (for multimodal)                          |
| `-mtp, --multimodal_text_prompt` | Text prompt for multimodal test                              |
| `-l, --log_file`                 | Output log file                                              |
| `-d, --duration_hours`           | Duration (hours) for stress testing; if 0, step mode is used |

Use `-h` for full options.

## Notes

- **Model Compatibility:** Set `DEFAULT_MODEL` (or pass `--model`) to a model actually deployed at your API endpoint.

- **Multimodal:** Your API backend must support OpenAI Vision API-style messages to use multimodal test.

- **Image Files:** Place your test images in the script directory, in a supported format (jpg, png, gif, webp).

- **API Token:** If authentication is needed, update `API_AUTH_TOKEN` in the script.

- **Logging:** Results print to both console and the log file you specify.

## Example Log Output

Output includes:

- Per-thread and total tokens/sec rates

- Throughput efficiency

- System CPU usage & temperature (stress mode)

- Recommendations based on scaling trends

## Troubleshooting

- Ensure your API endpoint is reachable and compatible with OpenAI API calls.

- Errors or warnings will be logged if images are missing or API is incompatible.

## License

MIT

Feel free to copy/modify above for your project, and update `DEFAULT_API_URL`, `DEFAULT_MODEL` and authentication if needed!

# LLM 并发性能测试脚本

本项目提供一个基于 Python 的测试脚本，可用于评估本地大语言模型（LLM，兼容 OpenAI API 接口）的并发处理能力，支持**文本**及多模态（图片+文本）输入，适配多种 API 后端与模型。

## 功能简介

- **文本输入测试**：测量 API 对文本输入处理的速度（tokens/秒）

- **文本输出测试**：测量并发环境下文本生成 token 的速度与吞吐量

- **多模态测试**：支持图片+文本输入（需后端兼容 OpenAI Vision API 消息格式）

- **步进测试模式**：并发数从起始逐步递增，便于评估可扩展性

- **压力测试模式**：固定并发数下长时间压测，适合高负载稳定性评估

- **系统性能监控**：在压力测试模式下监控 CPU 利用率和温度（如支持）

- **日志输出**：所有测试结果实时打印并同步写入指定日志文件

## 环境依赖

bash

`pip install aiohttp numpy psutil`

## 使用方法

## 1. 步进测试模式（默认）

并发数从 10 至 300 递增，每步 +10，测试输入与输出：

bash

`python llm-test.py -i -o -st 10 -et 300 -ts 10 -pp 500 -tg 500 -l custom_model_test2.log`

## 2. 压力测试模式

持续 8 小时，固定 300 个并发线程：

bash

`python llm-test.py -d 8 -st 300 -pp 500 -tg 500 -l custom_model_stress_test.log`

## 3. 多模态测试

图片 + 文本输入。请确保 testphoto.jpg（或指定图片）存在于本目录：

bash

`python llm-test.py -i -o -mm -st 10 -et 300 -ts 10 -pp 500 -tg 500 -ip testphoto.jpg -mtp "请描述图片内容。" -l custom_model_full_test.log`

## 常用参数说明

| 参数                               | 说明                |
| -------------------------------- | ----------------- |
| `-i, --input`                    | 启用文本输入测试          |
| `-o, --output`                   | 启用文本输出测试          |
| `-mm, --multimodal`              | 启用多模态测试（图片+文本）    |
| `-st, --start_threads`           | 起始并发数             |
| `-et, --end_threads`             | 末尾并发数（仅步进测试有效）    |
| `-ts, --step_threads`            | 并发数步进增量           |
| `-pp, --prompt_length`           | 输入文本长度（字符数）       |
| `-tg, --tokens_to_generate`      | 生成 token 数        |
| `-u, --url`                      | API 端点 URL        |
| `-m, --model`                    | 模型名称              |
| `-ip, --image_path`              | 多模态测试图片路径         |
| `-mtp, --multimodal_text_prompt` | 多模态文本提示           |
| `-l, --log_file`                 | 日志文件              |
| `-d, --duration_hours`           | 压力测试时长（小时，0为步进模式） |

运行 `-h` 可查看全部参数及帮助说明。

## 注意事项

- **API 兼容性**：请确保 API 后端兼容 OpenAI 格式，特别是进行多模态测试时需要 Vision API 消息格式支持

- **模型名称**：通过 `DEFAULT_MODEL` 或启动时 `--model` 指定实际部署的模型

- **图片支持**：多模态测试支持 jpg/png/gif/webp 等图片格式，需放在脚本目录下

- **API 鉴权**：如 API 需身份认证，请在脚本中更新 `API_AUTH_TOKEN`

- **日志记录**：结果会同时输出到终端和你指定的日志文件

## 输出指标

- 各请求 tokens/sec（单连接与总吞吐量）

- CPU 使用率与温度（压力测试模式）

- 扩展效率和吞吐建议

- 运行错误及警告自动记录

## 常见问题与排查

- 请确保 API 端点可访问，且参数正确

- 多模态测试如遇报错，多为图片路径、API 支持或模型不兼容问题

- 所有异常或警告均会在日志中注明

## 许可证

MIT 许可协议

如需更多使用方法或原理细节，欢迎阅读源码注释！此脚本可灵活适配不同环境，欢迎根据实际需求定制。

