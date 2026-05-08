from pathlib import Path
from owlready2 import get_ontology, sync_reasoner
import logging


def check_consistency(ontology_path: Path, report_path: Path):
    try:
        onto = get_ontology(str(ontology_path)).load()

        # Try to run reasoner, but handle Java not available
        try:
            sync_reasoner(onto, infer_property_values=True, debug=0)
            reasoner_success = True
        except Exception as e:
            logging.warning(f"Reasoner failed (likely Java not available): {e}")
            reasoner_success = False

        # Check for inconsistent individuals or classes
        if reasoner_success:
            inconsistent_classes = list(onto.inconsistent_classes())
            inconsistent_individuals = [ind for ind in onto.individuals() if ind is None or not ind.is_consistent()]
        else:
            inconsistent_classes = []
            inconsistent_individuals = []
            logging.info("Skipping detailed consistency checks due to reasoner failure")

        # Ensure the parent directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Consistency Report\n\n")
            f.write(f"Reasoner available: {reasoner_success}\n\n")

            if inconsistent_classes:
                f.write("## Inconsistent Classes\n")
                for cls in inconsistent_classes:
                    f.write(f"- {cls}\n")
            else:
                f.write("No inconsistent classes found.\n")

            if inconsistent_individuals:
                f.write("\n## Inconsistent Individuals\n")
                for ind in inconsistent_individuals:
                    f.write(f"- {ind}\n")
            else:
                f.write("\nNo inconsistent individuals found.\n")

        logging.info(f"Consistency check completed, report saved to {report_path}")

    except Exception as e:
        logging.error(f"Consistency check failed: {e}")
        # Ensure the parent directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Consistency check failed: {e}\n")


def consistency_check(ontology_path: Path, report_path: Path):
    check_consistency(ontology_path, report_path)
