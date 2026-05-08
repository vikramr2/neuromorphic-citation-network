import re
import logging
from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
import json


class EntityPostprocessor:
    """Postprocessor for cleaning up entity names and relations after merging."""

    def __init__(self, config):
        self.config = config
        self.cleanup_rules = []
        self._compile_rules()

    def _compile_rules(self):
        """Compile regex patterns from config."""
        for rule in self.config.cleanup_rules:
            try:
                compiled_pattern = re.compile(rule['pattern'], re.IGNORECASE)
                self.cleanup_rules.append({
                    'name': rule['name'],
                    'pattern': compiled_pattern,
                    'replacement': rule['replacement'],
                    'description': rule.get('description', '')
                })
                logging.info(f"Compiled cleanup rule '{rule['name']}': {rule['pattern']}")
            except re.error as e:
                logging.error(f"Invalid regex pattern in rule '{rule['name']}': {rule['pattern']} - {e}")

    def clean_text(self, text: str) -> str:
        """Apply all cleanup rules to a text string."""
        if not isinstance(text, str) or not text.strip():
            return text

        original_text = text
        for rule in self.cleanup_rules:
            text = rule['pattern'].sub(rule['replacement'], text)

        if text != original_text:
            logging.debug(f"Cleaned '{original_text}' -> '{text}'")

        return text

    def postprocess_dataframe(self, df: pd.DataFrame, fields_to_clean: List[str]) -> pd.DataFrame:
        """Apply postprocessing cleanup to specified fields in a DataFrame."""
        if df.empty:
            return df

        df_cleaned = df.copy()
        total_changes = 0

        for field in fields_to_clean:
            if field in df_cleaned.columns:
                original_values = df_cleaned[field].copy()
                df_cleaned[field] = df_cleaned[field].apply(self.clean_text)

                # Count changes
                changes = (original_values != df_cleaned[field]).sum()
                total_changes += changes

                if changes > 0:
                    logging.info(f"Applied postprocessing to {changes} entries in field '{field}'")
            else:
                logging.warning(f"Field '{field}' not found in DataFrame columns: {list(df_cleaned.columns)}")

        logging.info(f"Postprocessing completed: {total_changes} total text modifications across all fields")
        return df_cleaned

    def postprocess_jsonl_file(self, jsonl_path: Path, fields_to_clean: List[str]) -> int:
        """Apply postprocessing cleanup to a JSONL file in-place."""
        if not jsonl_path.exists():
            logging.warning(f"JSONL file does not exist: {jsonl_path}")
            return 0

        temp_path = jsonl_path.with_suffix('.tmp')
        total_changes = 0

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as infile, \
                 open(temp_path, 'w', encoding='utf-8') as outfile:

                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        original_data = data.copy()

                        # Skip triples with 'nan' entities
                        import math
                        h_val = data.get('h')
                        t_val = data.get('t')
                        if (isinstance(h_val, float) and math.isnan(h_val)) or \
                           (isinstance(t_val, float) and math.isnan(t_val)) or \
                           h_val == 'nan' or t_val == 'nan':
                            logging.debug(f"Skipping triple with 'nan' entity on line {line_num}")
                            continue

                        # Apply cleanup to specified fields
                        for field in fields_to_clean:
                            if field in data and isinstance(data[field], str):
                                data[field] = self.clean_text(data[field])

                        # Check if any changes were made
                        if data != original_data:
                            total_changes += 1

                        # Write the (possibly modified) data back
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                    except json.JSONDecodeError as e:
                        logging.error(f"Invalid JSON on line {line_num} of {jsonl_path}: {e}")
                        # Write original line if JSON is invalid
                        outfile.write(line + '\n')

            # Replace original file with cleaned version
            temp_path.replace(jsonl_path)
            logging.info(f"Postprocessed JSONL file {jsonl_path}: {total_changes} lines modified")

        except Exception as e:
            logging.error(f"Error postprocessing JSONL file {jsonl_path}: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            return 0

        return total_changes

    def postprocess_merged_data(self, merged_dir: Path) -> Dict[str, int]:
        """Apply postprocessing to all merged data files in the merged directory."""
        if not self.config.enabled:
            logging.info("Postprocessing is disabled in config")
            return {}

        results = {}

        # Postprocess entities table
        entities_path = merged_dir / "merged_entities.txt"
        if entities_path.exists():
            try:
                df = pd.read_csv(entities_path, sep='|', encoding='utf-8', engine='python')
                df_cleaned = self.postprocess_dataframe(df, ['entity_name'])
                df_cleaned.to_csv(entities_path, sep='|', index=False, encoding='utf-8')
                results['entities'] = len(df_cleaned)
                logging.info(f"Postprocessed entities file: {entities_path}")
            except Exception as e:
                logging.error(f"Error postprocessing entities file: {e}")

        # Postprocess predications table
        predications_path = merged_dir / "merged_predications.txt"
        if predications_path.exists():
            try:
                df = pd.read_csv(predications_path, sep='|', encoding='utf-8', engine='python')
                df_cleaned = self.postprocess_dataframe(df, ['subject_entity_name', 'object_entity_name', 'predicate'])
                df_cleaned.to_csv(predications_path, sep='|', index=False, encoding='utf-8')
                results['predications'] = len(df_cleaned)
                logging.info(f"Postprocessed predications file: {predications_path}")
            except Exception as e:
                logging.error(f"Error postprocessing predications file: {e}")

        # Postprocess JSONL triples file
        jsonl_path = merged_dir / "all_triples.jsonl"
        if jsonl_path.exists():
            changes = self.postprocess_jsonl_file(jsonl_path, ['h', 'r', 't'])
            results['jsonl_changes'] = changes
            logging.info(f"Postprocessed JSONL file: {jsonl_path}")

        # Postprocess documents table (if it has entity names)
        documents_path = merged_dir / "merged_documents.txt"
        if documents_path.exists():
            try:
                df = pd.read_csv(documents_path, sep='|', encoding='utf-8', engine='python')
                # Check if there are any text fields that might need cleaning
                text_fields = [col for col in df.columns if df[col].dtype == 'object']
                if text_fields:
                    df_cleaned = self.postprocess_dataframe(df, text_fields)
                    df_cleaned.to_csv(documents_path, sep='|', index=False, encoding='utf-8')
                    results['documents'] = len(df_cleaned)
                    logging.info(f"Postprocessed documents file: {documents_path}")
            except Exception as e:
                logging.error(f"Error postprocessing documents file: {e}")

        total_processed = sum(results.values())
        logging.info(f"✅ Postprocessing completed: {total_processed} total records processed")
        return results


def run_postmerging(config, merged_dir: Path) -> Dict[str, int]:
    """Run postprocessing on merged data using the provided configuration."""
    postprocessor = EntityPostprocessor(config.postprocessing)
    return postprocessor.postprocess_merged_data(merged_dir)