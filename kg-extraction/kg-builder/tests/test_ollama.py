import argparse
import http
import requests
import time
from pathlib import Path

# Try to import the ollama library, but don't fail if it's not installed,
# as the user might only want to use the HTTP API.
try:
    from ollama import Client
except ImportError:
    Client = None

def test_http_generate(base_url: str, model: str, prompt: str, text: str, times: int = 3):
    """Tests the /api/generate endpoint using the requests library."""
    print("--- Testing HTTP Generate Endpoint ---")
    payload = {
        "model": model,
        "prompt": f"{prompt}\n\n{text}",
        "stream": False,
    }
    
    durations = []
    for i in range(times):
        print(f"Run {i+1}/{times}:")
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )
            response.raise_for_status()
            end_time = time.time()
            duration = end_time - start_time
            
            if i == 0:  # Only show response for first run
                content = response.json().get('response', '')
                print("Response:\n", content)
            print(f"⏱️ HTTP Generate API call took: {duration:.2f} seconds")
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ HTTP Generate API call took: {duration:.2f} seconds")
            print(f"Error calling HTTP Generate API: {e}")
            print(f"Response body: {e.response.text if e.response else 'No response'}")
        
        durations.append(duration)
    
    avg_duration = sum(durations) / len(durations)
    print(f"📊 Average time over {times} runs: {avg_duration:.2f} seconds")

def test_http_chat(base_url: str, model: str, prompt: str, text: str, times: int = 3):
    """Tests the /api/chat endpoint using the requests library."""
    print("--- Testing HTTP Chat Endpoint ---")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        "stream": False,
    }
    
    durations = []
    for i in range(times):
        print(f"Run {i+1}/{times}:")
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                headers={'Content-Type': 'application/json; charset=utf-8'}
            )
            response.raise_for_status()
            end_time = time.time()
            duration = end_time - start_time
            
            if i == 0:  # Only show response for first run
                content = response.json().get('message', {}).get('content', '')
                print("Response:\n", content)
            print(f"⏱️ HTTP Chat API call took: {duration:.2f} seconds")
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ HTTP Chat API call took: {duration:.2f} seconds")
            print(f"Error calling HTTP Chat API: {e}")
            print(f"Response body: {e.response.text if e.response else 'No response'}")
        
        durations.append(duration)
    
    avg_duration = sum(durations) / len(durations)
    print(f"📊 Average time over {times} runs: {avg_duration:.2f} seconds")


def test_python_generate(host: str, model: str, prompt: str, text: str, times: int = 3):
    """Tests the generate endpoint using the ollama-python library."""
    if Client is None:
        print("Ollama python library not installed. Skipping.")
        return
    print("--- Testing Python SDK Generate Endpoint ---")
    
    durations = []
    for i in range(times):
        print(f"Run {i+1}/{times}:")
        start_time = time.time()
        try:
            client = Client(host=host)
            response = client.generate(
                model=model,
                prompt=f"{prompt}\n\n{text}",
                stream=False,
            )
            end_time = time.time()
            duration = end_time - start_time
            
            if i == 0:  # Only show response for first run
                content = response.get('response', '')
                print("Response:\n", content)
            print(f"⏱️ Python SDK Generate API call took: {duration:.2f} seconds")
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Python SDK Generate API call took: {duration:.2f} seconds")
            print(f"Error calling Python SDK Generate API: {e}")
        
        durations.append(duration)
    
    avg_duration = sum(durations) / len(durations)
    print(f"📊 Average time over {times} runs: {avg_duration:.2f} seconds")

def test_python_chat(host: str, model: str, prompt: str, text: str, times: int = 3):
    """Tests the chat endpoint using the ollama-python library."""
    if Client is None:
        print("Ollama python library not installed. Skipping.")
        return
    print("--- Testing Python SDK Chat Endpoint ---")
    
    durations = []
    for i in range(times):
        print(f"Run {i+1}/{times}:")
        start_time = time.time()
        try:
            client = Client(host=host)
            response = client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': text},
                ]
            )
            end_time = time.time()
            duration = end_time - start_time
            
            if i == 0:  # Only show response for first run
                content = response.get('message', {}).get('content', '')
                print("Response:\n", content)
            print(f"⏱️ Python SDK Chat API call took: {duration:.2f} seconds")
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ Python SDK Chat API call took: {duration:.2f} seconds")
            print(f"Error calling Python SDK Chat API: {e}")
        
        durations.append(duration)
    
    avg_duration = sum(durations) / len(durations)
    print(f"📊 Average time over {times} runs: {avg_duration:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Ollama Test Harness")
    parser.add_argument("--prompt-file", type=Path, help="Path to a single prompt file.")
    parser.add_argument("--prompt-dir", type=Path, help="Directory containing prompt files (.md files).")
    parser.add_argument("--text-file", type=Path, help="Path to a single text file.")
    parser.add_argument("--text-dir", type=Path, help="Directory containing text files (.txt files).")
    parser.add_argument("--host", type=str, default="http://localhost", help="Ollama server host.")
    parser.add_argument("--port", type=int, default=11434, help="Ollama server port.")
    parser.add_argument("--model", type=str, help="Single model to test.")
    parser.add_argument("--models", nargs='+', help="List of models to test.")
    parser.add_argument("--api", type=str, choices=['http', 'python', 'all'], default='all', help="API to test.")
    parser.add_argument("--endpoint", type=str, choices=['generate', 'chat', 'all'], default='all', help="Endpoint to test.")
    parser.add_argument("-t", "--times", type=int, default=3, help="Number of times to run each test (default: 3).")

    args = parser.parse_args()

    # Validate arguments
    if not args.prompt_file and not args.prompt_dir:
        parser.error("Either --prompt-file or --prompt-dir must be specified")
    if args.prompt_file and args.prompt_dir:
        parser.error("Cannot specify both --prompt-file and --prompt-dir")
    if not args.text_file and not args.text_dir:
        parser.error("Either --text-file or --text-dir must be specified")
    if args.text_file and args.text_dir:
        parser.error("Cannot specify both --text-file and --text-dir")
    if not args.model and not args.models:
        parser.error("Either --model or --models must be specified")
    if args.model and args.models:
        parser.error("Cannot specify both --model and --models")

    # Collect prompt files
    if args.prompt_file:
        prompt_files = [args.prompt_file]
    else:
        prompt_files = list(args.prompt_dir.glob("*.md"))
        if not prompt_files:
            parser.error(f"No .md files found in {args.prompt_dir}")

    # Collect text files
    if args.text_file:
        text_files = [args.text_file]
    else:
        text_files = list(args.text_dir.glob("*.txt"))
        if not text_files:
            parser.error(f"No .txt files found in {args.text_dir}")

    # Collect models
    models = args.models if args.models else [args.model]

    # Load all prompts and texts
    prompts = {}
    for prompt_file in prompt_files:
        try:
            prompts[prompt_file.name] = prompt_file.read_text(encoding='utf-8')
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check your prompt file paths.")
            return

    texts = {}
    for text_file in text_files:
        try:
            texts[text_file.stem] = text_file.read_text(encoding='utf-8')
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check your text file paths.")
            return

    base_url = f"{args.host}:{args.port}"
    python_host_url = f"{args.host}:{args.port}"

    # Override with hardcoded values if needed (for testing)
    base_url = "http://carz1.ornl.gov:11434"
    python_host_url = "http://carz1.ornl.gov:11434"

    apis_to_test = ['http', 'python'] if args.api == 'all' else [args.api]
    endpoints_to_test = ['generate', 'chat'] if args.endpoint == 'all' else [args.endpoint]

    # Reordered loop: model first, then prompt, then document (most efficient for Ollama)
    for model_name in models:
        print(f"🔄 Testing with model: {model_name}")
        
        for api in apis_to_test:
            for endpoint in endpoints_to_test:
                for prompt_name, prompt_content in prompts.items():
                    for text_name, text_content in texts.items():
                        print(f"  📝 Using prompt: {prompt_name} | 📄 Document: {text_name}")
                        
                        if api == 'http' and endpoint == 'generate':
                            test_http_generate(base_url, model_name, prompt_content, text_content, args.times)
                        elif api == 'http' and endpoint == 'chat':
                            test_http_chat(base_url, model_name, prompt_content, text_content, args.times)
                        elif api == 'python' and endpoint == 'generate':
                            test_python_generate(python_host_url, model_name, prompt_content, text_content, args.times)
                        elif api == 'python' and endpoint == 'chat':
                            test_python_chat(python_host_url, model_name, prompt_content, text_content, args.times)
                        
                        print("-" * 60)
        
        print(f"✅ Completed testing for model: {model_name}")
        print("=" * 80)


if __name__ == "__main__":
    main()
