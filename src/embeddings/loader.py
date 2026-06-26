from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader , ImageReader
import os

def universal_doc_loader(path : str):
    
    if not os.path.exists(path):
        print('File directory is not exits.')
        return []
    
    file_extractors = {
        '.pdf' : PyMuPDFReader(),
        '.png' : ImageReader(keep_image=False),
        '.jpg' : ImageReader(keep_image=False),
        '.jpeg' : ImageReader(keep_image=False)
    }
    
    loader = SimpleDirectoryReader(
        input_dir=path,
        file_extractor=file_extractors,
    )
    
    return loader.load_data()

docs = universal_doc_loader('src/uploads')

if docs:
    print(f'Succefully loaded {len(docs)} page/documents')
    print('----Sample Text----')
    print(docs[0].text[500])
else:
    print('No pdf or file is found try again.')