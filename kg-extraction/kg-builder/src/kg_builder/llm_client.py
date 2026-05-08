from openai import OpenAI
from pathlib import Path
import logging
import requests
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import time
import re
import string
import uuid
from .io_utils import normalize_entity, normalize_relation, normalize_triple


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: Optional[int] = None, provider: str = "vllm", exclusions: list[str] = None, timeout: int = 600):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.timeout = timeout  # Timeout in seconds for requests
        if exclusions is None:
            exclusions = []
        self.exclusions = exclusions

        if provider == "vllm":
            self.client = OpenAI(base_url=base_url, api_key="dummy")  # No API key required for vLLM
        else:
            self.client = None  # For Ollama, we'll use direct HTTP requests

    def log_failure(self, document_id: str, prompt_path: Path, model_name: str, server_url: str, error_type: str, error_msg: str):
        """Log all failures that result in 0 triples to a dedicated failure log file."""
        import json
        from pathlib import Path
        
        failure_log_path = Path("logs_multiprompt/failures.jsonl")
        failure_log_path.parent.mkdir(exist_ok=True, parents=True)
        
        failure_record = {
            "timestamp": time.time(),
            "document_id": document_id,
            "prompt_filename": prompt_path.name,
            "model_name": model_name,
            "server_url": server_url,
            "error_type": error_type,
            "error_message": error_msg,
            "timeout_seconds": self.timeout
        }
        
        try:
            with open(failure_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failure_record, ensure_ascii=False) + '\n')
            logging.warning(f"Logged {error_type} failure for doc {document_id}, model {model_name} on server {server_url}")
        except Exception as e:
            logging.error(f"Failed to log failure: {e}")

    def extract_triples_with_regeneration(self, text: str, prompt_path: Path, regen_config=None) -> tuple[list, int]:
        """
        Extract triples with regeneration capability for empty responses.

        Returns:
            tuple: (triples_list, attempt_number)
        """
        if regen_config and regen_config.enabled:
            return self._extract_with_regeneration(text, prompt_path, regen_config)
        else:
            # Use original method without regeneration
            response = self.extract_triples(text, prompt_path)
            triples = self._parse_jsonl(response)
            return triples, 1

    def _extract_with_regeneration(self, text: str, prompt_path: Path, regen_config) -> tuple[list, int]:
        """
        Extract triples with regeneration attempts using different parameters.
        Only triggers regeneration for responses that appear to be actual failures.
        For timeouts, retries with same parameters since the model can complete successfully.
        """
        # Read the prompt once
        try:
            with open(prompt_path) as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {prompt_path}")
            raise

        current_temperature = self.temperature
        current_max_tokens = self.max_tokens

        for attempt in range(1, regen_config.max_attempts + 1):
            if attempt == 1:
                # First attempt - use original parameters
                pass
            else:
                # Log when regeneration is triggered
                server = self.base_url.replace('http://', '').replace('https://', '').split(':')[0]
                logging.info(f"Regeneration triggered for {server} - attempt {attempt}/{regen_config.max_attempts} "
                           f"(temp={current_temperature:.2f}, max_tokens={current_max_tokens or 'None'})")

            try:
                # Temporarily modify parameters for this attempt
                original_temp = self.temperature
                original_tokens = self.max_tokens

                self.temperature = current_temperature
                self.max_tokens = current_max_tokens

                # Extract triples with current parameters
                response = self.extract_triples(text, prompt_path)

                # Restore original parameters
                self.temperature = original_temp
                self.max_tokens = original_tokens

                # Parse the response
                triples = self._parse_jsonl(response)

                # Check if we got any triples
                if len(triples) > 0:
                    server = self.base_url.replace('http://', '').replace('https://', '').split(':')[0]
                    logging.info(f"Regeneration successful on attempt {attempt} for {server}: extracted {len(triples)} triples")
                    logging.info(f"Regeneration completed successfully for {server} after {attempt} attempts")
                    return triples, attempt

                # Check if this looks like a real failure that should trigger regeneration
                if self._should_trigger_regeneration(response):
                    logging.warning(f"Attempt {attempt} returned 0 triples, trying regeneration...")

                    # Prepare parameters for next attempt
                    if attempt < regen_config.max_attempts:
                        current_temperature = min(
                            current_temperature + regen_config.temperature_increment,
                            regen_config.max_temperature
                        )
                        # Slightly increase max tokens for more creative responses
                        if current_max_tokens is not None:
                            current_max_tokens = int(current_max_tokens * 1.1)  # Reduced from 1.2

                        # Add delay between attempts
                        if regen_config.delay_between_attempts > 0:
                            time.sleep(regen_config.delay_between_attempts)
                else:
                    # Response doesn't look like it should contain triples, don't regenerate
                    logging.info(f"Response appears valid but empty (no triples found), not regenerating")
                    return [], attempt

            except requests.exceptions.Timeout as e:
                # Handle timeouts specially - don't trigger regeneration, just retry with same parameters
                server = self.base_url.replace('http://', '').replace('https://', '').split(':')[0]
                if attempt < regen_config.max_attempts:
                    logging.warning(f"Timeout on attempt {attempt} for {server}, retrying with same parameters...")
                    # For timeouts, don't change parameters - the model can complete successfully
                    # Add a small delay to avoid overwhelming the server
                    time.sleep(1.0)
                else:
                    logging.error(f"All attempts timed out for {server}, giving up")
                    raise

            except Exception as e:
                server = self.base_url.replace('http://', '').replace('https://', '').split(':')[0]
                logging.error(f"Regeneration attempt {attempt} failed for {server}: {e}")
                if attempt == regen_config.max_attempts:
                    raise

        # All attempts failed, try alternative extraction method as last resort
        logging.info("Trying alternative extraction method as last resort...")
        try:
            # Use the appropriate extraction method for the provider
            if self.provider == "vllm":
                response = self._extract_openwebui_style(text, prompt)
            else:  # Ollama or other providers
                response = self._extract_ollama(text, prompt)
            triples = self._parse_jsonl(response)
            if len(triples) > 0:
                logging.info(f"Alternative extraction successful: extracted {len(triples)} triples")
                return triples, regen_config.max_attempts + 1
        except Exception as e:
            logging.error(f"Alternative extraction failed: {e}")

        # All attempts failed, return the last response
        server = self.base_url.replace('http://', '').replace('https://', '').split(':')[0]
        logging.warning(f"All regeneration attempts including OpenWebUI style failed for {server}, returning empty result")
        logging.warning(f"Regeneration failed completely for {server} after {regen_config.max_attempts} attempts")
        return [], regen_config.max_attempts + 1

    def _should_trigger_regeneration(self, response: str) -> bool:
        """
        Determine if regeneration should be triggered based on response analysis.
        Only trigger regeneration for responses that appear to be actual failures.
        """
        if not response or not response.strip():
            return True  # Empty response is a failure

        response_lower = response.lower().strip()

        # Don't regenerate if response indicates no triples found in valid text
        no_triple_indicators = [
            "no triples",
            "no relationships",
            "no relations",
            "cannot extract",
            "unable to find",
            "no factual information",
            "no verifiable information"
        ]

        for indicator in no_triple_indicators:
            if indicator in response_lower:
                return False  # Don't regenerate - model correctly identified no triples

        # Don't regenerate if response is very short (likely intentional)
        if len(response.strip()) < 50:
            return False  # Probably intentional empty response

        # Don't regenerate if response contains valid JSON but no triples
        if '"h"' in response or '"subject"' in response:
            # Contains JSON structure but no valid triples were parsed
            # This might be a parsing issue rather than generation issue
            return False

        # Trigger regeneration for other cases (malformed responses, etc.)
        return True

    def generate_embedding(self, text: str) -> list:
        """Generate embeddings for text using the configured model."""
        try:
            if self.provider == "vllm":
                # Use OpenAI-compatible embeddings API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    timeout=self.timeout
                )
                return response.data[0].embedding
            
            elif self.provider == "ollama":
                # Use Ollama embeddings API
                payload = {
                    "model": self.model,
                    "prompt": text
                }
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    headers={'Content-Type': 'application/json; charset=utf-8'},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('embedding', [])
                else:
                    logging.error(f"Ollama embeddings request failed: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama embeddings API error: {response.status_code}")
            
            else:
                raise ValueError(f"Embeddings not supported for provider: {self.provider}")
                
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logging.error(f"Embedding generation failed for model {self.model}: {e}")
            raise

    def extract_triples(self, text: str, prompt_path: Path, document_id: str = None) -> str:
        try:
            with open(prompt_path) as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {prompt_path}")
            raise

        if self.provider == "vllm":
            return self._extract_vllm(text, prompt, document_id, prompt_path)
        elif self.provider == "openwebui":
            return self._extract_openwebui_style(text, prompt, document_id, prompt_path)
        else:
            return self._extract_ollama(text, prompt, document_id, prompt_path)

    def _extract_openwebui_style(self, text: str, prompt: str, document_id: str = None, prompt_path: Path = None) -> str:
        """Extract triples using OpenWebUI-style message structure."""
        if self.provider == "vllm":
            # Use OpenAI client with separate messages for prompt and text
            messages = [
                {"role": "user", "content": prompt},
                {"role": "user", "content": text}
            ]
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # Ensure content is properly decoded
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                return content
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                logging.error(f"vLLM OpenWebUI style request failed: {e}")
                if document_id and prompt_path:
                    self.log_failure(document_id, prompt_path, self.model, self.base_url, "vllm_openwebui_error", error_msg)
                raise
        else:
            # Use HTTP request for other providers (OpenWebUI, Ollama with compatible endpoint)
            # Generate unique message IDs like OpenWebUI
            user_message_id = str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())
            
            # Create the message structure similar to OpenWebUI
            messages = {
                user_message_id: {
                    "id": user_message_id,
                    "parentId": None,
                    "childrenIds": [assistant_message_id],
                    "role": "user",
                    "content": f"{prompt}\n\n{text}",
                    "files": []  # Add file support if needed
                }
            }
            
            # Create the OpenWebUI-style payload with different structure
            payload = {
                "model": self.model,  # Try model at top level
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n{text}"
                    }
                ],
                "options": {
                    "temperature": self.temperature,
                    "stream": False
                }
            }
            if self.max_tokens is not None:
                payload["options"]["max_tokens"] = self.max_tokens

            try:
                # Send request to OpenWebUI-compatible endpoint
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers={'Content-Type': 'application/json; charset=utf-8'},
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    data = response.json()
                    
                    # Extract content from OpenWebUI response format
                    # The response structure may vary, adjust based on your actual OpenWebUI setup
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0].get('message', {}).get('content', '')
                    elif 'response' in data:
                        content = data['response']
                    elif 'content' in data:
                        content = data['content']
                    else:
                        # Try to find content in the response structure
                        content = self._extract_content_from_openwebui_response(data)
                    
                    # Ensure content is properly decoded
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                    return content
                else:
                    response.encoding = 'utf-8'
                    logging.error(f"OpenWebUI request failed: {response.status_code} - {response.text}")
                    raise Exception(f"OpenWebUI API error: {response.status_code}")
            except requests.exceptions.Timeout as e:
                error_msg = f"Request timed out after {self.timeout} seconds"
                logging.error(f"OpenWebUI request timeout for model {self.model}: {error_msg}")
                if document_id and prompt_path:
                    self.log_failure(document_id, prompt_path, self.model, self.base_url, "timeout", error_msg)
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                logging.error(f"OpenWebUI request failed: {e}")
                if document_id and prompt_path:
                    self.log_failure(document_id, prompt_path, self.model, self.base_url, "openwebui_error", error_msg)
                raise

    def _extract_content_from_openwebui_response(self, data: dict) -> str:
        """Helper to extract content from various OpenWebUI response formats."""
        # Try different possible response structures
        possible_paths = [
            ['message', 'content'],
            ['assistant_message', 'content'],
            ['result', 'content'],
            ['data', 'content'],
            ['output'],
            ['text']
        ]
        
        for path in possible_paths:
            current = data
            try:
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        break
                else:
                    # Successfully traversed the path
                    if isinstance(current, str):
                        return current
            except (KeyError, TypeError):
                continue
        
        # Fallback: return string representation of the entire response
        logging.warning("Could not find content in OpenWebUI response, returning full response")
        return str(data)

    def _extract_vllm(self, text: str, prompt: str, document_id: str = None, prompt_path: Path = None) -> str:
        """Extract triples using vLLM with Unicode support."""
        # Combine the prompt and the text into a single user message
        combined_content = f"{prompt}\n\n{text}"
        messages = [
            {"role": "user", "content": combined_content}
        ]

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "timeout": self.timeout  # Add timeout parameter
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            # Ensure content is properly decoded
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            return content
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logging.error(f"vLLM request failed: {e}")
            if document_id and prompt_path:
                self.log_failure(document_id, prompt_path, self.model, self.base_url, "vllm_error", error_msg)
            raise

    def _extract_ollama(self, text: str, prompt: str, document_id: str = None, prompt_path: Path = None) -> str:
        """Extract triples using Ollama with Unicode support."""
        payload = {
            "model": self.model,
            "prompt": f"{prompt}\n\n{text}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens

        try:
            # Send request with proper encoding and timeout
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={'Content-Type': 'application/json; charset=utf-8'},
                timeout=self.timeout
            )

            if response.status_code == 200:
                # Ensure response is properly decoded
                response.encoding = 'utf-8'
                data = response.json()
                content = data.get('response', '')
                # Ensure content is properly decoded
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                return content
            else:
                # Handle error response with proper encoding
                response.encoding = 'utf-8'
                logging.error(f"Ollama request failed: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timed out after {self.timeout} seconds"
            logging.error(f"Ollama request timeout for model {self.model}: {error_msg}")
            if document_id and prompt_path:
                self.log_failure(document_id, prompt_path, self.model, self.base_url, "timeout", error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logging.error(f"Ollama request failed: {e}")
            if document_id and prompt_path:
                self.log_failure(document_id, prompt_path, self.model, self.base_url, "ollama_error", error_msg)
            raise

    def _parse_jsonl(self, response: str) -> list:
        """Parse JSONL response into triples list with robust error handling and cleanup."""
        import json
        import re
        triples = []

        if not response or not response.strip():
            logging.warning("Empty or None response received")
            return triples

        # Log the raw response for debugging (first 200 chars)
        logging.debug(f"Raw LLM response (first 200 chars): {response[:200]}...")

        # First, try to extract all JSON objects from the response using regex
        # Handle both single and double quotes, and alternative key names
        json_patterns = [
            r'\{[^{}]*["\']h["\']\s*:\s*["\'][^"\']*["\']\s*,\s*["\']r["\']\s*:\s*["\'][^"\']*["\']\s*,\s*["\']t["\']\s*:\s*["\'][^"\']*["\'][^{}]*\}',  # h,r,t
            r'\{[^{}]*["\']subject["\']\s*:\s*["\'][^"\']*["\']\s*,\s*["\'](?:relation|predicate)["\']\s*:\s*["\'][^"\']*["\']\s*,\s*["\']object["\']\s*:\s*["\'][^"\']*["\'][^{}]*\}'  # subject,relation/object
        ]

        json_matches = []
        for pattern in json_patterns:
            json_matches.extend(re.findall(pattern, response, re.DOTALL))

        if json_matches:
            logging.debug(f"Found {len(json_matches)} potential JSON objects via regex")
            # Process regex-extracted JSON objects
            for match in json_matches:
                try:
                    # Clean up the JSON string
                    cleaned_json = self._clean_json_string(match)
                    logging.debug(f"Attempting to parse cleaned JSON: {cleaned_json}")
                    triple = json.loads(cleaned_json, strict=False)
                    # Map alternative key names to standard h,r,t
                    triple = self._map_triple_keys(triple)
                    if self._validate_triple(triple):
                        # Normalize the triple entities and relations
                        triple = self._normalize_triple(triple)
                        triples.append(triple)
                        logging.debug(f"Successfully parsed and normalized triple: {triple}")
                    else:
                        logging.warning(f"Triple failed validation: {triple}")
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"Failed to parse extracted JSON: {match[:100]}... - Error: {e}")

        # If regex didn't find anything, try line-by-line parsing as fallback
        if not triples:
            logging.debug("Regex extraction failed, trying line-by-line parsing")
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Try to extract JSON from the line
                json_in_line = self._extract_json_from_line(line)
                if json_in_line:
                    try:
                        cleaned_json = self._clean_json_string(json_in_line)
                        logging.debug(f"Attempting to parse line JSON: {cleaned_json}")
                        triple = json.loads(cleaned_json, strict=False)
                        # Map alternative key names to standard h,r,t
                        triple = self._map_triple_keys(triple)
                        if self._validate_triple(triple):
                            # Normalize the triple entities and relations
                            triple = self._normalize_triple(triple)
                            triples.append(triple)
                            logging.debug(f"Successfully parsed and normalized triple from line: {triple}")
                        else:
                            logging.warning(f"Line triple failed validation: {triple}")
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"Failed to parse JSON from line: {line[:100]}... - Error: {e}")

        # If still no triples found, try more aggressive parsing approaches
        if not triples:
            logging.debug("Line-by-line parsing failed, trying more aggressive approaches")

            # Try to find any JSON-like structures with more lenient patterns
            lenient_patterns = [
                r'\{[^}]*h[^}]*r[^}]*t[^}]*\}',  # Any object with h, r, t somewhere
                r'\{[^}]*subject[^}]*relation[^}]*object[^}]*\}',  # Alternative keys
                r'\{[^}]*head[^}]*relation[^}]*tail[^}]*\}',  # Alternative keys
            ]

            for pattern in lenient_patterns:
                matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    try:
                        cleaned_json = self._clean_json_string(match)
                        triple = json.loads(cleaned_json, strict=False)
                        triple = self._map_triple_keys(triple)
                        if self._validate_triple(triple):
                            triple = self._normalize_triple(triple)
                            triples.append(triple)
                            logging.debug(f"Successfully parsed triple with lenient pattern: {triple}")
                    except:
                        continue

                if triples:  # If we found some triples, stop trying other patterns
                    break

        # If still no triples found, try to fix the response with LLM
        if not triples and response.strip():
            # Check if LLM-based JSON fixing is enabled in config
            enable_llm_fix = True  # Default to enabled for backward compatibility
            if hasattr(self, 'config') and hasattr(self.config, 'parallel_execution'):
                enable_llm_fix = getattr(self.config.parallel_execution, 'enable_llm_json_fix', True)

            if enable_llm_fix:
                logging.info(f"No valid triples found in response ({len(response)} chars), attempting to fix malformed response...")
                try:
                    fixed_response = self._fix_malformed_response(response)
                    if fixed_response:
                        logging.debug("LLM fix attempt successful, recursively parsing fixed response")
                        # Recursively parse the fixed response
                        return self._parse_jsonl(fixed_response)
                    else:
                        logging.warning("LLM fix attempt returned empty or None response")
                except Exception as e:
                    logging.warning(f"Failed to fix malformed response: {e}")
            else:
                logging.info("LLM-based JSON fixing disabled in configuration")

        logging.info(f"JSON parsing completed: extracted {len(triples)} valid triples")
        return triples

    def _clean_json_string(self, json_str: str) -> str:
        """Clean up malformed JSON strings."""
        # Remove extra whitespace and newlines
        cleaned = json_str.strip()

        # Convert single quotes to double quotes for property names and string values
        # This is a common issue where LLMs use single quotes instead of double quotes
        cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)  # Convert 'value' to "value"
        cleaned = re.sub(r'{\s*', '{', cleaned)  # Remove spaces after {
        cleaned = re.sub(r'\s*}', '}', cleaned)  # Remove spaces before }
        cleaned = re.sub(r',\s*', ',', cleaned)  # Remove spaces after commas
        cleaned = re.sub(r'\s*:', ':', cleaned)  # Remove spaces before colons

        # Fix common issues
        # Remove trailing commas before closing braces
        cleaned = re.sub(r',\s*}', '}', cleaned)
        # Remove trailing commas before closing brackets
        cleaned = re.sub(r',\s*]', ']', cleaned)

        # Ensure proper Unicode handling
        try:
            # Try to encode/decode to handle any encoding issues
            cleaned = cleaned.encode('utf-8').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        return cleaned

    def _extract_json_from_line(self, line: str) -> str:
        """Extract JSON object from a line that might contain extra text."""
        # Find the first '{' and last '}' to extract JSON
        start_idx = line.find('{')
        end_idx = line.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return line[start_idx:end_idx + 1]

        return None

    def _validate_triple(self, triple: dict) -> bool:
        """Validate that a parsed object is a valid triple."""
        if not isinstance(triple, dict):
            return False

        required_keys = ['h', 'r', 't']
        if not all(key in triple for key in required_keys):
            return False

        # Ensure all values are strings or can be converted to strings
        for key in required_keys:
            if not isinstance(triple[key], (str, int, float)):
                return False

        # Convert non-string values to strings
        for key in required_keys:
            if not isinstance(triple[key], str):
                triple[key] = str(triple[key])

        return True

    def _fix_malformed_response(self, response: str) -> str:
        """Use LLM to fix a malformed JSON response."""
        try:
            # Create a simple fix prompt
            fix_prompt = f"""The following text contains JSON objects that need to be extracted and fixed.
Return only valid JSON lines, one per line, in this exact format:
{{"h": "head", "r": "relation", "t": "tail"}}

Original text:
{response}

Fixed JSON lines:"""

            # Use self to fix the response
            logging.info(f"Using {self.model} to fix malformed response...")

            # Make a simple API call to fix the response
            if self.provider == "vllm":
                response_obj = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": fix_prompt}],
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=1024
                )
                fixed_content = response_obj.choices[0].message.content.strip()

            elif self.provider == "ollama":
                payload = {
                    "model": self.model,
                    "prompt": fix_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024
                    }
                }
                ollama_response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=30
                )
                if ollama_response.status_code == 200:
                    data = ollama_response.json()
                    fixed_content = data.get('response', '').strip()
                else:
                    return None

            return fixed_content if fixed_content else None

        except Exception as e:
            logging.warning(f"Failed to fix malformed response with LLM: {e}")
            return None

        return None

    def _map_triple_keys(self, triple: dict) -> dict:
        """Map alternative key names to standard triple format (h, r, t)."""
        if not isinstance(triple, dict):
            return triple
        
        # Create a new dict to avoid modifying while iterating
        mapped_triple = {}
        
        for key, value in triple.items():
            key_lower = key.lower()
            if key_lower == 'subject':
                mapped_triple['h'] = value
            elif key_lower == 'object':
                mapped_triple['t'] = value
            elif key_lower in ['relation', 'predicate']:
                mapped_triple['r'] = value
            else:
                # Keep other keys as is
                mapped_triple[key] = value
        
        return mapped_triple

    def _normalize_triple(self, triple: dict) -> dict:
        """Normalize the entities and relation in a triple."""
        return normalize_triple(triple, self.exclusions)


class MultiLLMClient:
    def __init__(self, model_configs: list, exclusions: list[str] = None, default_timeout: int = 600):
        if exclusions is None:
            exclusions = []
        self.exclusions = exclusions
        self.clients = []
        for config in model_configs:
            # Use model-specific timeout if provided, otherwise use default
            timeout = config.timeout if config.timeout is not None else default_timeout
            client = LLMClient(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                provider=config.provider,
                exclusions=self.exclusions,
                timeout=timeout
            )
            self.clients.append({
                'client': client,
                'name': config.name
            })

    def extract_triples_multi(self, text: str, prompt_path: Path, config=None) -> dict:
        """
        Extract triples using all configured LLMs sequentially with optional regeneration.

        Args:
            text: The text to extract triples from
            prompt_path: Path to the prompt file
            config: Configuration object (optional, for regeneration settings)

        Returns:
            dict: {model_name: triples_list} for each model
        """
        sequential_start_time = time.time()
        logging.info(f"Starting sequential execution across {len(self.clients)} models")

        results = {}
        regen_enabled = config and hasattr(config, 'regeneration') and config.regeneration.enabled

        for client_info in self.clients:
            client = client_info['client']
            model_name = client_info['name']

            model_start_time = time.time()
            try:
                logging.info(f"Querying model: {model_name}")

                if regen_enabled:
                    # Use regeneration method
                    triples, attempts_used = client.extract_triples_with_regeneration(
                        text, prompt_path, config.regeneration
                    )
                    results[model_name] = triples

                    model_time = time.time() - model_start_time
                    if attempts_used > 1:
                        logging.info(f"Extracted {len(triples)} triples from {model_name} "
                                   f"in {model_time:.2f}s (used {attempts_used} attempts)")
                    else:
                        logging.info(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")
                else:
                    # Use original method
                    response = client.extract_triples(text, prompt_path)
                    triples = client._parse_jsonl(response)
                    results[model_name] = triples

                    model_time = time.time() - model_start_time
                    logging.info(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")

            except Exception as e:
                model_time = time.time() - model_start_time
                logging.error(f"Failed to extract triples from {model_name} after {model_time:.2f}s: {e}")
                results[model_name] = []

        sequential_time = time.time() - sequential_start_time
        successful_models = sum(1 for triples in results.values() if len(triples) > 0)
        total_triples = sum(len(triples) for triples in results.values())

        logging.info(f"Sequential execution completed: {successful_models}/{len(self.clients)} models successful, "
                    f"{total_triples} total triples in {sequential_time:.2f}s")

        return results

    def extract_triples_single_model(self, model_name: str, text: str, prompt_path: Path, config=None) -> list:
        """
        Extract triples using a single specified model.

        Args:
            model_name: Name of the model to use
            text: The text to extract triples from
            prompt_path: Path to the prompt file
            config: Configuration object (optional, for regeneration settings)

        Returns:
            list: List of triples extracted by the model
        """
        # Find the client for the specified model
        client_info = None
        for info in self.clients:
            if info['name'] == model_name:
                client_info = info
                break

        if client_info is None:
            logging.error(f"Model '{model_name}' not found in configured models")
            return []

        client = client_info['client']
        model_start_time = time.time()

        try:
            logging.debug(f"Extracting triples from model: {model_name}")

            regen_enabled = config and hasattr(config, 'regeneration') and config.regeneration.enabled

            if regen_enabled:
                # Use regeneration method
                triples, attempts_used = client.extract_triples_with_regeneration(
                    text, prompt_path, config.regeneration
                )
                model_time = time.time() - model_start_time
                if attempts_used > 1:
                    logging.debug(f"Extracted {len(triples)} triples from {model_name} "
                                f"in {model_time:.2f}s (used {attempts_used} attempts)")
                else:
                    logging.debug(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")
            else:
                # Use original method
                response = client.extract_triples(text, prompt_path)
                triples = client._parse_jsonl(response)
                model_time = time.time() - model_start_time
                logging.debug(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")

            return triples

        except Exception as e:
            model_time = time.time() - model_start_time
            logging.error(f"Failed to extract triples from {model_name} after {model_time:.2f}s: {e}")
            return []

    def extract_triples_multi_parallel(self, text: str, prompt_path: Path, config, document_id: str = None, models_to_run: list = None) -> dict:
        """
        Extract triples using configured LLMs with parallel execution across different servers.
        Supports regeneration for empty responses and selective model execution.

        Args:
            text: The text to extract triples from
            prompt_path: Path to the prompt file
            config: Configuration object with parallel_execution and regeneration settings
            document_id: Optional document ID for logging
            models_to_run: Optional list of model names to run (if None, runs all models)

        Returns:
            dict: {model_name: triples_list} for each model
        """
        # Check for parallel execution config in the LLM section
        parallel_config = getattr(config, 'parallel_execution', None)
        if not parallel_config or not parallel_config.enabled:
            logging.info("Parallel execution disabled, falling back to sequential execution")
            return self.extract_triples_multi(text, prompt_path, config)

        parallel_start_time = time.time()
        regen_enabled = hasattr(config, 'regeneration') and config.regeneration.enabled

        # Filter clients to only those specified in models_to_run
        if models_to_run:
            filtered_clients = [client_info for client_info in self.clients if client_info['name'] in models_to_run]
            if len(filtered_clients) != len(models_to_run):
                missing_models = set(models_to_run) - set(client_info['name'] for client_info in filtered_clients)
                logging.warning(f"Some requested models not found: {missing_models}")
        else:
            filtered_clients = self.clients

        if regen_enabled:
            logging.info(f"Starting parallel execution with regeneration across {len(filtered_clients)} models")
        else:
            logging.info(f"Starting parallel execution across {len(filtered_clients)} models")

        # Group clients by server (base_url)
        server_groups = {}
        for client_info in filtered_clients:
            client = client_info['client']
            model_name = client_info['name']
            base_url = client.base_url

            if base_url not in server_groups:
                server_groups[base_url] = []
            server_groups[base_url].append((model_name, client))

        logging.info(f"Grouped {len(filtered_clients)} models into {len(server_groups)} server groups")

        # Execute requests in parallel across different servers
        results = {}
        max_workers = min(len(server_groups), parallel_config.max_concurrent_servers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each server group
            future_to_server = {}
            for base_url, model_clients in server_groups.items():
                future = executor.submit(
                    self._process_server_group,
                    base_url,
                    model_clients,
                    text,
                    prompt_path,
                    parallel_config,
                    config.regeneration if regen_enabled else None,
                    document_id
                )
                future_to_server[future] = base_url

            # Collect results as they complete
            for future in as_completed(future_to_server):
                server_url = future_to_server[future]
                try:
                    server_results = future.result()
                    results.update(server_results)
                    logging.info(f"Collected results from server: {server_url}")
                except Exception as e:
                    logging.error(f"Failed to process server {server_url}: {e}")
                    # Add empty results for failed models
                    for model_name, _ in server_groups[server_url]:
                        results[model_name] = []

        parallel_time = time.time() - parallel_start_time
        successful_models = sum(1 for triples in results.values() if len(triples) > 0)
        total_triples = sum(len(triples) for triples in results.values())

        logging.info(f"Parallel execution completed: {successful_models}/{len(filtered_clients)} models successful, "
                    f"{total_triples} total triples in {parallel_time:.2f}s")

        return results

    def _process_server_group(self, server_url: str, model_clients: List[tuple], text: str,
                              prompt_path: Path, parallel_config, regen_config=None, document_id: str = None) -> Dict[str, List[Dict]]:
        """
        Process all models for a single server sequentially with optional regeneration.

        Args:
            server_url: The base URL of the server
            model_clients: List of (model_name, client) tuples for this server
            text: The text to extract triples from
            prompt_path: Path to the prompt file
            parallel_config: Parallel execution configuration
            regen_config: Regeneration configuration (optional)

        Returns:
            dict: {model_name: triples_list} for models on this server
        """
        results = {}
        server_start_time = time.time()

        regen_enabled = regen_config and regen_config.enabled
        if regen_enabled:
            logging.info(f"Processing {len(model_clients)} models on server: {server_url} (with regeneration)")
        else:
            logging.info(f"Processing {len(model_clients)} models on server: {server_url}")

        for model_name, client in model_clients:
            model_start_time = time.time()

            for attempt in range(parallel_config.retry_attempts + 1):
                try:
                    logging.info(f"Querying model: {model_name} (attempt {attempt + 1})")

                    if regen_enabled:
                        # Use regeneration method
                        triples, regen_attempts = client.extract_triples_with_regeneration(
                            text, prompt_path, regen_config
                        )
                        results[model_name] = triples

                        model_time = time.time() - model_start_time
                        if regen_attempts > 1:
                            logging.info(f"Extracted {len(triples)} triples from {model_name} "
                                       f"in {model_time:.2f}s (used {regen_attempts} regeneration attempts)")
                        else:
                            logging.info(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")
                    else:
                        # Use original method
                        response = client.extract_triples(text, prompt_path, document_id)
                        triples = client._parse_jsonl(response)
                        results[model_name] = triples

                        model_time = time.time() - model_start_time
                        logging.info(f"Extracted {len(triples)} triples from {model_name} in {model_time:.2f}s")

                    break  # Success, exit retry loop

                except Exception as e:
                    if attempt < parallel_config.retry_attempts:
                        delay = parallel_config.retry_delay * (2 ** attempt)  # Exponential backoff
                        logging.warning(f"Attempt {attempt + 1} failed for {model_name}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        model_time = time.time() - model_start_time
                        logging.error(f"All attempts failed for {model_name} after {model_time:.2f}s: {e}")
                        results[model_name] = []

        server_time = time.time() - server_start_time
        successful_models = sum(1 for triples in results.values() if len(triples) > 0)
        total_triples = sum(len(triples) for triples in results.values())

        logging.info(f"Completed server {server_url}: {successful_models}/{len(model_clients)} models successful, "
                    f"{total_triples} total triples in {server_time:.2f}s")

        return results