import re
import unicodedata
import string


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_entity(entity: str, exclusions: list[str] = None) -> str:
    """Normalize entity to canonical singular form with individual word processing.
    
    Each word in multi-word entities is processed individually:
    - Plurals are converted to singular form
    - All words are kept in lowercase
    - Acronyms (2-5 uppercase letters) are preserved
    
    Examples:
        "Spikes Between Layers Of Neuron" -> "spike between layer of neuron"
        "Graph Neural Networks" -> "graph neural network"
        "CNN" -> "CNN" (preserved acronym)
    """
    if exclusions is None:
        exclusions = []
    
    if not entity or not isinstance(entity, str):
        return entity

    # Store original words to preserve acronyms
    original_words = entity.strip().split()

    # Process each word individually
    normalized_words = []
    for i, word in enumerate(original_words):
        # Convert to lowercase for processing
        word_lower = word.lower()

        # Skip normalization if word is in exclusions
        if word_lower in exclusions:
            normalized_words.append(word_lower)
            continue

        # Handle plural forms for individual words
        if word_lower.endswith('ies'):
            word_lower = word_lower[:-3] + 'y'  # studies -> study
        elif word_lower.endswith('es'):
            if word_lower.endswith(('ches', 'shes', 'sses', 'xes', 'zes')):
                word_lower = word_lower[:-2]  # boxes -> box
            else:
                word_lower = word_lower[:-1]  # processes -> process
        elif word_lower.endswith('s') and not word_lower.endswith(('ss', 'us', 'is')):
            word_lower = word_lower[:-1]  # networks -> network

        # Preserve original casing for acronyms (2-5 chars, all caps)
        if word.isupper() and 2 <= len(word) <= 5 and word.isalpha():
            normalized_words.append(word.upper())
        else:
            normalized_words.append(word_lower)

    return ' '.join(normalized_words)


def normalize_relation(relation: str, exclusions: list[str] = None) -> str:
    """Normalize relation to base verb form and snake_case."""
    if exclusions is None:
        exclusions = []
    
    if not relation or not isinstance(relation, str):
        return relation

    # Convert to lowercase first
    relation = relation.strip().lower()

    # Convert spaces and hyphens to underscores for snake_case first
    relation = re.sub(r'[\s\-]+', '_', relation)

    # Split by underscores to handle compound relations
    parts = relation.split('_')
    normalized_parts = []

    for part in parts:
        original_part = part

        # Skip normalization if part is in exclusions
        if part in exclusions:
            normalized_parts.append(part)
            continue

        # Handle verb endings to get base form - be more careful
        if part.endswith('ing') and len(part) > 5:  # using (5+ chars)
            part = part[:-3]
        elif part.endswith('ed') and len(part) > 5:  # used (4+ chars after removal)
            part = part[:-2]
        elif part.endswith('es') and len(part) > 5:  # uses (4+ chars after removal)
            part = part[:-2]
        elif part.endswith('s') and not part.endswith(('ss', 'us', 'is')) and len(part) > 4:  # uses (3+ chars after removal)
            part = part[:-1]

        # Handle specific verb mappings - only apply if the part matches exactly
        verb_mappings = {
            'use': 'use',
            'uses': 'use',
            'using': 'use',
            'used': 'use',
            'apply': 'apply',
            'applies': 'apply',
            'applying': 'apply',
            'applied': 'apply',
            'include': 'include',
            'includes': 'include',
            'including': 'include',
            'included': 'include',
            'contain': 'contain',
            'contains': 'contain',
            'containing': 'contain',
            'contained': 'contain',
            'belong': 'belong',
            'belongs': 'belong',
            'belonging': 'belong',
            'belonged': 'belong',
            'is': 'be',
            'are': 'be',
            'was': 'be',
            'were': 'be',
            'has': 'have',
            'have': 'have',
            'had': 'have',
            'been': 'be',
        }

        # Only apply mapping if the original part (before any transformations) is in the mapping
        if original_part in verb_mappings:
            part = verb_mappings[original_part]

        normalized_parts.append(part)

    # Join back with underscores
    relation = '_'.join(normalized_parts)

    # Remove multiple underscores
    relation = re.sub(r'_+', '_', relation)

    # Remove leading/trailing underscores
    relation = relation.strip('_')

    return relation


def normalize_triple(triple: dict, exclusions: list[str] = None) -> dict:
    """Normalize the entities and relation in a triple."""
    if exclusions is None:
        exclusions = []
    
    if not isinstance(triple, dict):
        return triple

    # Normalize entities to Title Case and singular form
    if 'h' in triple:
        triple['h'] = normalize_entity(triple['h'], exclusions)
    if 't' in triple:
        triple['t'] = normalize_entity(triple['t'], exclusions)
    if 'subject_entity_name' in triple:
        triple['subject_entity_name'] = normalize_entity(triple['subject_entity_name'], exclusions)
    if 'object_entity_name' in triple:
        triple['object_entity_name'] = normalize_entity(triple['object_entity_name'], exclusions)

    # Normalize relation to snake_case and base form
    if 'r' in triple:
        triple['r'] = normalize_relation(triple['r'], exclusions)
    if 'predicate' in triple:
        triple['predicate'] = normalize_relation(triple['predicate'], exclusions)

    return triple
