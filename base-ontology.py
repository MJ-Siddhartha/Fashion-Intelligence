import networkx as nx
import json
from typing import Dict, Any, List
import rdflib

class FashionOntology:
    def __init__(self):
        """
        Initialize Fashion Ontology with a graph-based structure
        """
        self.graph = nx.DiGraph()
        self.rdf_graph = rdflib.Graph()
        self._initialize_base_ontology()

    def _initialize_base_ontology(self):
        """
        Create base taxonomy of fashion features
        """
        base_categories = {
            'Apparel': ['Top', 'Bottom', 'Dress', 'Outerwear'],
            'Accessories': ['Jewelry', 'Bags', 'Footwear', 'Headwear'],
            'Materials': ['Cotton', 'Silk', 'Polyester', 'Wool'],
            'Styles': ['Casual', 'Formal', 'Vintage', 'Modern']
        }

        for category, subcategories in base_categories.items():
            self.graph.add_node(category, type='main_category')
            for subcategory in subcategories:
                self.graph.add_edge(category, subcategory)

    def add_feature(self, category: str, feature: Dict[str, Any]):
        """
        Add a new feature to the ontology
        
        Args:
            category (str): Product category
            feature (Dict): Feature details
        """
        feature_name = feature.get('name')
        self.graph.add_node(feature_name, **feature)
        self.graph.add_edge(category, feature_name)

        # RDF representation
        subject = rdflib.URIRef(f"fashion:{feature_name}")
        for key, value in feature.items():
            predicate = rdflib.URIRef(f"fashion:{key}")
            object_value = rdflib.Literal(value)
            self.rdf_graph.add((subject, predicate, object_value))

    def get_feature_relationships(self, feature: str) -> Dict[str, List[str]]:
        """
        Get relationships and context for a feature
        
        Args:
            feature (str): Feature to analyze
        
        Returns:
            Dict of feature relationships
        """
        predecessors = list(self.graph.predecessors(feature))
        successors = list(self.graph.successors(feature))
        
        return {
            'category': predecessors,
            'related_features': successors,
            'metadata': dict(self.graph.nodes[feature])
        }

    def export_ontology(self, format='json'):
        """
        Export ontology in specified format
        
        Args:
            format (str): Export format (json/rdf)
        
        Returns:
            Exported ontology representation
        """
        if format == 'json':
            return nx.node_link_data(self.graph)
        elif format == 'rdf':
            return self.rdf_graph.serialize(format='turtle')

    def merge_ontologies(self, other_ontology):
        """
        Merge another ontology with current ontology
        
        Args:
            other_ontology (FashionOntology): Ontology to merge
        """
        self.graph = nx.compose(self.graph, other_ontology.graph)
        self.rdf_graph += other_ontology.rdf_graph

# Usage example
if __name__ == "__main__":
    ontology = FashionOntology()
    ontology.add_feature('Dress', {
        'name': 'Floral Summer Dress',
        'pattern': 'Floral',
        'length': 'Knee',
        'sleeve_type': 'Sleeveless'
    })
    print(json.dumps(ontology.export_ontology(), indent=2))
