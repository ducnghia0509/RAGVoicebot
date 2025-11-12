# prompt/builder.py
def build_context_and_prompt(chunks: list, user_question: str) -> tuple[str, list]:
    context = "\n\n".join([
        f"**Nguồn {i+1}:** {c['metadata'].get('doc_type','')} - {c['metadata'].get('document_number','')} - {c['metadata'].get('title','')}\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    system_prompt = "Bạn là trợ lý pháp lý chuyên nghiệp tại Việt Nam. Chỉ trả lời dựa hoàn toàn vào Context được cung cấp. Không suy diễn, không thêm thông tin ngoài Context. Có thể có nhiều context nhiễu, phải trả lời CẨN THẬN và đưa ra câu trả lời thống nhất sau cùng, bạn trả lời tự nhiên như đang nói chuyện với người!"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nCâu hỏi: {user_question}"}
    ]
    return context, messages