import pymongo
from pymongo.collection import Collection
from typing import List, Dict, Any
import logging


def connect_mongo(uri: str, database: str, collection: str) -> Collection:
    try:
        client = pymongo.MongoClient(uri)
        db = client[database]
        if collection not in db.list_collection_names():
            # Heuristic: find collections with 'pub', 'paper', 'doc', 'article'
            candidates = [c for c in db.list_collection_names() if any(keyword in c.lower() for keyword in ['pub', 'paper', 'doc', 'article'])]
            if candidates:
                collection = candidates[0]
                logging.warning(f"Collection '{collection}' not found, using '{collection}' as fallback")
            else:
                available = db.list_collection_names()
                raise ValueError(f"Collection '{collection}' not found in database '{database}'. Available collections: {available}")
        return db[collection]
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise


def get_documents(collection: Collection, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        if limit <= 0:
            # No limit - get all documents, sorted by _id for consistent ordering
            docs = list(collection.find().sort([('_id', 1)]))
            logging.info(f"Retrieved ALL {len(docs)} documents from collection (no limit applied, sorted by _id)")
        else:
            docs = list(collection.find().sort([('_id', 1)]).limit(limit))
            logging.info(f"Retrieved {len(docs)} documents from collection (limit: {limit}, sorted by _id)")
        return docs
    except Exception as e:
        logging.error(f"Failed to retrieve documents: {e}")
        raise


def extract_text(doc: Dict[str, Any]) -> str:
    text_parts = []
    # if 'title' in doc and doc['title']:
    #     text_parts.append(str(doc['title']))
    # if 'abstract' in doc and doc['abstract']:
    #     text_parts.append(str(doc['abstract']))
    # if 'body' in doc and doc['body']:
    #     text_parts.append(str(doc['body']))
    # if not text_parts:
    #     # Fallback to entire doc as string
    #     text_parts.append(str(doc))
    # return ' '.join(text_parts)
def extract_text(doc: Dict[str, Any]) -> str:
    """Extract text from MongoDB document with proper Unicode handling."""
    current_content = []
    sections = doc.get('sections')
    if sections is None:
        return ""  # Return empty string if sections is None

    for sec in sections:
        content = sec['content']
        # Ensure content is properly decoded as string
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        elif not isinstance(content, str):
            content = str(content)
        current_content.append(content)

    total_content = " ".join(current_content)
    total_content_length = len(total_content)
    print(f"Total content length: {total_content_length}")
    return total_content

