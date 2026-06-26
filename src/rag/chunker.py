from llama_index.core.node_parser import SentenceSplitter
from src.rag.loader import universal_doc_loader

def chunker(documents, chunk_size : int = 512, chunk_overlap:int = 50):
    """_summary_

    Args:
        documents (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 512.
        chunk_overlap (int, optional): _description_. Defaults to 50.
        
    Descriptions:
        Take documents and create a beautiful chunk for embedding layers
    """
    
    splitter = SentenceSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    return nodes

# docs = universal_doc_loader('src/uploads')
# chunks = chunker(docs)

# if chunks:
#     print(f'Successfully split {len(docs)} into {len(chunks)}')
#     print('----Sample of chunks----')
#     print(chunks[2].text[:400])

    
    