# retrieval/metadata_extractor.py
import re

def extract_metadata_from_query(query: str) -> tuple[dict, list]:
    query_lower = query.lower()
    filters = {}
    general_keywords = []

    # Loại văn bản
    doc_type_map = {
        "luật": "Luật", "nghị định": "Nghị định", "thông tư": "Thông tư",
        "quyết định": "Quyết định", "công văn": "Công văn", "hướng dẫn": "Hướng dẫn",
        "nghị quyết": "Nghị quyết", "chỉ thị": "Chỉ thị",
    }
    for keyword, doc_type in doc_type_map.items():
        if re.search(rf'\b{keyword}\b', query_lower):
            filters["doc_type"] = doc_type
            break

    # Cơ quan ban hành
    issuer_map = {
        "quốc hội": "Quốc hội", "chính phủ": "Chính phủ", "thủ tướng": "Thủ tướng Chính phủ",
        "bộ tài chính": "Bộ Tài chính", "bộ y tế": "Bộ Y tế", "tổng cục thuế": "Tổng cục Thuế",
        "ubnd": "UBND", "ủy ban nhân dân": "UBND",
    }
    for keyword, issuer in issuer_map.items():
        if keyword in query_lower:
            filters["issuer"] = issuer
            break

    # Số hiệu văn bản
    patterns = [
        r'\b(\d{1,4}/\d{4}/[A-ZĐÂÊÔƯ\-]+)\b',
        r'\b(\d{1,4}/\d{4})\b',
        r'\b(?:số\s*)?(\d{1,3})(?=\s*(?:/|năm|nghị|thông|quyết|$))',
    ]
    for pattern in patterns:
        m = re.search(pattern, query_lower)
        if m:
            filters["document_number"] = m.group(1)
            break

    # Năm
    if m := re.search(r'\b(20\d{2})\b', query):
        filters["year"] = int(m.group(1))

    # Tiêu đề trong ngoặc kép hoặc sau "về"
    if m := re.search(r'"([^"]+)"', query):
        filters["title_contains"] = m.group(1).strip()
    elif m := re.search(r'(?:về|liên quan đến|quy định về)\s+([^,;.]+)', query_lower):
        filters["title_contains"] = m.group(1).strip()

    # Từ khóa chung (dùng để boost source_file)
    general_keywords = [kw for kw in re.findall(r'\b\w{5,}\b', query_lower)
                       if kw not in doc_type_map and kw not in issuer_map]

    return filters, general_keywords