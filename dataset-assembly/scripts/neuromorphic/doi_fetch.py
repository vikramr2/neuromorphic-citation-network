from pymongo import MongoClient
import json
from bson import ObjectId
import requests
from tqdm import tqdm


def get_doi_from_title(title: str) -> str:
    """
    Get DOI from article title using CrossRef API.
    
    Args:
        title: The article title to search for
        
    Returns:
        DOI string if found, None otherwise
    """
    url = "https://api.crossref.org/works"
    params = {
        'query.title': title,
        'rows': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['message']['items']:
            doi = data['message']['items'][0].get('DOI')
            return doi
        else:
            print("Can't find doi for title: ", title)
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

if __name__ == '__main__':
    client = MongoClient("mongodb://carz1.ornl.gov:27017")
    db = client["knight"]
    collection = db["knight_documents"]

    print(collection.count_documents({}))  # Should print 2396

    # Iterate through all documents and fetch all data with DOIs
    print("\nFetching all documents and adding DOI fields...")
    total_docs = collection.count_documents({})
    documents_with_dois = []

    for doc in tqdm(collection.find(), total=total_docs, desc="Processing documents"):
        # Convert MongoDB document to dict and handle ObjectId
        doc_dict = json.loads(MongoJSONEncoder().encode(doc))

        # Get DOI from title using CrossRef API
        if "title" in doc and doc["title"]:
            doi = get_doi_from_title(doc["title"])
            doc_dict["doi"] = doi  # Add DOI field (will be None if not found)
        else:
            doc_dict["doi"] = None

        documents_with_dois.append(doc_dict)

    print(f"\nProcessed {len(documents_with_dois)} documents")
    print(f"Found DOIs for {sum(1 for d in documents_with_dois if d.get('doi'))} documents")

    # Save all document data with DOI fields
    with open("../data/mongo_data_with_dois.json", 'w') as f:
        json.dump(documents_with_dois, f, indent=2)

    print(f"\nSaved all document data to ../data/mongo_data_with_dois.json")
