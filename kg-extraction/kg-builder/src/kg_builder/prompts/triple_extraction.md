Task:
Extract **factual RDF triples** in the form (head, relation, tail) from the following scientific text.
Focus only on verifiable, non-speculative information that can be directly supported by the text.

Output Rules:
- Each triple must follow this EXACT JSON format (one per line):
  {"h": "Head Entity", "r": "Relation", "t": "Tail Entity"}
- Use **canonical singular forms**:
  - Entities: always use singular (e.g., "Graph Neural Network" not "Graph Neural Networks").
  - Relations: always use base verb or canonical form (e.g., "use_for", not "uses_for" or "used_for").
- Normalize casing: entities should be in **Title Case**, relations should be in **snake_case**.
- Exclude vague claims, opinions, future work, or references to figures/tables.
- Do not invent facts not supported by the text.
- If multiple relations exist between the same entities, output multiple triples.
- Output ONLY valid JSON lines (no explanations, no extra text, no markdown).

Critical: Your response must contain ONLY JSON lines. No introductory text, no explanations, no code blocks.

Examples:
Input:
"Deep learning models such as Convolutional Neural Networks are widely used for image classification tasks."

Output:
{"h": "Convolutional Neural Network", "r": "use_for", "t": "Image Classification"}
{"h": "Deep Learning Model", "r": "include", "t": "Convolutional Neural Network"}

Input:
"The transformer architecture employs self-attention mechanisms and is based on the attention is all you need paper."

Output:
{"h": "Transformer Architecture", "r": "employ", "t": "Self Attention Mechanism"}
{"h": "Transformer Architecture", "r": "base_on", "t": "Attention Is All You Need"}