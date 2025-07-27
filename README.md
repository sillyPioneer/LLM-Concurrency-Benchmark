

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

1. https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/78716966/80faccca-ed0c-421c-a01c-768af8345e4d/llm-test.py
