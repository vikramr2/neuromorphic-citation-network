import pandas as pd
from pathlib import Path
import owlready2
import logging


def create_ontology(triples: list, name: str, base_iri: str, ontology_path: Path):
    onto = owlready2.get_ontology(base_iri + name + "#")

    with onto:
        # Generic Entity class
        class Entity(owlready2.Thing):
            pass

        # Collect unique relations
        relations = set(t['r'] for t in triples)
        properties = {}
        for rel in relations:
            prop_name = rel.replace(' ', '_').replace('-', '_')
            # Create ObjectProperty class within ontology context
            prop_class = type(prop_name, (owlready2.ObjectProperty,), {})
            setattr(onto, prop_name, prop_class)
            properties[rel] = prop_class

        # Collect unique entities
        entities = set()
        for t in triples:
            entities.add(t['h'])
            entities.add(t['t'])

        individuals = {}
        for ent in entities:
            ind_name = ent.replace(' ', '_').replace('-', '_')
            ind = Entity(base_iri + ind_name)
            individuals[ent] = ind

        # Assert triples
        for t in triples:
            h_ind = individuals[t['h']]
            t_ind = individuals[t['t']]
            prop_name = t['r'].replace(' ', '_').replace('-', '_')
            # Skip built-in attributes
            if prop_name in ['is_a', 'namespace', 'storid', 'name']:
                logging.warning(f"Skipping built-in property: {prop_name}")
                continue
            # Use setattr to assign the relation
            if hasattr(h_ind, prop_name):
                current_list = getattr(h_ind, prop_name)
                if not isinstance(current_list, list):
                    current_list = [current_list]
                if t_ind not in current_list:
                    current_list.append(t_ind)
                    setattr(h_ind, prop_name, current_list)
            else:
                setattr(h_ind, prop_name, [t_ind])

def create_ontology(triples: list, name: str, base_iri: str, ontology_path: Path):
    onto = owlready2.get_ontology(base_iri + name + "#")

    with onto:
        # Generic Entity class
        class Entity(owlready2.Thing):
            pass

        # Collect unique relations - handle both old format (h,r,t) and new format (subject_entity_name, predicate, object_entity_name)
        relations = set()
        for t in triples:
            if 'r' in t and pd.notna(t['r']):
                relations.add(t['r'])
            elif 'predicate' in t and pd.notna(t['predicate']):
                relations.add(t['predicate'])

        properties = {}
        for rel in relations:
            prop_name = rel.replace(' ', '_').replace('-', '_')
            # Create ObjectProperty class within ontology context
            prop_class = type(prop_name, (owlready2.ObjectProperty,), {})
            setattr(onto, prop_name, prop_class)
            properties[rel] = prop_class

        # Collect unique entities - handle both formats
        entities = set()
        for t in triples:
            if 'h' in t and 't' in t and pd.notna(t['h']) and pd.notna(t['t']):
                entities.add(t['h'])
                entities.add(t['t'])
            elif 'subject_entity_name' in t and 'object_entity_name' in t and pd.notna(t['subject_entity_name']) and pd.notna(t['object_entity_name']):
                entities.add(t['subject_entity_name'])
                entities.add(t['object_entity_name'])

        individuals = {}
        for ent in entities:
            if pd.notna(ent):  # Additional check for NaN entities
                ind_name = ent.replace(' ', '_').replace('-', '_')
                ind = Entity(base_iri + ind_name)
                individuals[ent] = ind

        # Assert triples - handle both formats
        for t in triples:
            try:
                if 'h' in t and 'r' in t and 't' in t and pd.notna(t['h']) and pd.notna(t['r']) and pd.notna(t['t']):
                    if t['h'] not in individuals or t['t'] not in individuals:
                        logging.warning(f"Skipping triple with missing entities: {t}")
                        continue
                    h_ind = individuals[t['h']]
                    t_ind = individuals[t['t']]
                    prop_name = t['r'].replace(' ', '_').replace('-', '_')
                elif 'subject_entity_name' in t and 'predicate' in t and 'object_entity_name' in t and pd.notna(t['subject_entity_name']) and pd.notna(t['predicate']) and pd.notna(t['object_entity_name']):
                    if t['subject_entity_name'] not in individuals or t['object_entity_name'] not in individuals:
                        logging.warning(f"Skipping triple with missing entities: {t}")
                        continue
                    h_ind = individuals[t['subject_entity_name']]
                    t_ind = individuals[t['object_entity_name']]
                    prop_name = t['predicate'].replace(' ', '_').replace('-', '_')
                else:
                    logging.warning(f"Skipping triple with missing or NaN values: {t}")
                    continue

                # Skip built-in attributes
                if prop_name in ['is_a', 'namespace', 'storid', 'name', 'save']:
                    logging.warning(f"Skipping built-in property: {prop_name}")
                    continue
                # Use setattr to assign the relation
                if hasattr(h_ind, prop_name):
                    current_list = getattr(h_ind, prop_name)
                    if not isinstance(current_list, list):
                        current_list = [current_list]
                    if t_ind not in current_list:
                        current_list.append(t_ind)
                        setattr(h_ind, prop_name, current_list)
                else:
                    setattr(h_ind, prop_name, [t_ind])
            except Exception as e:
                logging.warning(f"Error processing triple {t}: {e}")
                continue

    # Ensure the parent directory exists
    ontology_path.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(ontology_path), format="rdfxml")
    logging.info(f"Saved ontology to {ontology_path}")


def generate_ontology(merged_dir: Path, ontology_path: Path, name: str, base_iri: str):
    deduped_txt = merged_dir / "deduped_predications.txt"
    if not deduped_txt.exists():
        logging.error(f"Deduped predications file not found: {deduped_txt}")
        return

    df = pd.read_csv(deduped_txt, sep='|', encoding='utf-8', engine='python')
    triples = df.to_dict('records')
    create_ontology(triples, name, base_iri, ontology_path)
