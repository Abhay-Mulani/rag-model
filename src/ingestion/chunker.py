from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Document] = []
    step = chunk_size - chunk_overlap
    for document in documents:
        content = document.page_content
        for start in range(0, len(content), step):
            chunk = content[start : start + chunk_size].strip()
            if chunk:
                metadata = {**document.metadata, "chunk_start": start}
                chunks.append(Document(page_content=chunk, metadata=metadata))
            if start + chunk_size >= len(content):
                break
    return chunks
