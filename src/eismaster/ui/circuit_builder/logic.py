from typing import List, Dict
import typing







from eismaster.ui.circuit_builder.graphics import ComponentNode, CircuitScene

def build_cdc_from_scene(scene: CircuitScene) -> str:
    """
    Experimental topology parser.
    Currently maps a simple series chain of components.
    """
    nodes = []
    for item in scene.items():
        if isinstance(item, ComponentNode):
            nodes.append(item)
            
    input_node = next((n for n in nodes if n.c_type == "INPUT"), None)
    output_node = next((n for n in nodes if n.c_type == "OUTPUT"), None)
    
    if not input_node or not output_node:
        return ""
        
    # Traverse from input to output
    # For now, simplistic straight-line traversal assumption
    cdc_parts = []
    
    current = input_node
    while current:
        if current.c_type not in ("INPUT", "OUTPUT"):
            cdc_parts.append(current.c_type)
            
        out_edges = current.out_socket.edges
        if not out_edges:
            break
            
        # Simplistic next node
        next_node = out_edges[0].dest.parent_node
        current = next_node
        
    return "".join(cdc_parts)