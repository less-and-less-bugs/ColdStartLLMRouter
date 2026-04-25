import json
from typing import Dict, Any

def read_kg_data(kg_path: str) -> dict:
    with open(kg_path, 'r', encoding='utf-8') as f:
        nodes_data = json.load(f)["nodes"]
    return nodes_data
    
def extract_task_hierarchy(kg_path: str) -> Dict[str, Any]:
    """
    Extract task hierarchy from knowledge graph
    
    Returns:
        Dictionary containing domains, subcategories, and difficulty levels
    """
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    hierarchy = {
        "domains": [],
        "subcategories": [],
        "difficulty_levels": [],
        "preference_levels": [],
        "domain_to_subcats": {},
        "subcat_to_difficulties": {},
        "subcat_to_preferences": {}
    }
    
    # First pass: create domains with IDs
    domain_id_map = {}  # map domain name to id
    for domain in kg.get("nodes", []):
        domain_name = domain["name"]
        domain_id = len(hierarchy["domains"])  # ID is current length
        domain_id_map[domain_name] = domain_id
        hierarchy["domains"].append({
            "id": domain_id,
            "name": domain_name,
            "definition": domain["definition"],
            "example": domain["example"]
        })
        
    # Second pass: process subcategories and their children
    subcat_id_map = {}  # map subcat name to id
    
    for domain in kg.get("nodes", []):
        domain_name = domain["name"]
        domain_id = domain_id_map[domain_name]
        subcats = []
        
        for subcat in domain.get("subcategory_nodes", []):
            subcat_name = subcat["name"]
            subcat_id = len(hierarchy["subcategories"])  # ID is current length
            subcat_id_map[subcat_name] = subcat_id
            
            subcats.append(subcat_name)
            hierarchy["subcategories"].append({
                "id": subcat_id,
                "name": subcat_name,
                "definition": subcat["definition"],
                "example": subcat["example"],
                "domain": domain_name,
                "parent_id": domain_id
            })
            
            difficulties = []
            for diff in subcat.get("difficulty_nodes", []):
                diff_name = diff["name"]
                diff_id = len(hierarchy["difficulty_levels"])  # ID is current length
                
                difficulties.append(diff_name)
                hierarchy["difficulty_levels"].append({
                    "id": diff_id,
                    "name": diff_name,
                    "definition": diff["definition"],
                    "example": diff["example"],
                    "subcategory": subcat_name,
                    "domain": domain_name,
                    "parent_id": subcat_id
                })
            
            preferences = []
            for pref in subcat.get("preference_nodes", []):
                pref_name = pref["name"]
                pref_id = len(hierarchy["preference_levels"])  # ID is current length
                
                preferences.append(pref_name)
                hierarchy["preference_levels"].append({
                    "id": pref_id,
                    "name": pref_name,
                    "definition": pref["definition"],
                    "example": pref["example"],
                    "subcategory": subcat_name,
                    "domain": domain_name,
                    "parent_id": subcat_id
                })
            
            hierarchy["subcat_to_difficulties"][subcat_id] = difficulties
            hierarchy["subcat_to_preferences"][subcat_id] = preferences
        
        hierarchy["domain_to_subcats"][domain_id] = subcats
    
    return hierarchy
    