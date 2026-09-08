"""End-to-end testing script for AskMyPDF API."""

import json
import time
import urllib.request
import io
import httpx

BASE_URL = "http://127.0.0.1:8000"

def create_sample_pdf_bytes() -> bytes:
    """Create a minimal valid PDF with structured text."""
    # A standard valid minimal 2-page PDF
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        "4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        "5 0 obj << /Length 200 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(AskMyPDF is a modern modular RAG platform. The project architecture follows strict SOLID and OOP standards with hot-swappable interfaces for vector databases, LLMs, embeddings, and caches.) Tj\nET\n"
        "endstream\nendobj\n"
        "6 0 obj << /Length 180 >> stream\n"
        "BT\n/F1 12 Tf\n50 700 Td\n(The warranty period for all enterprise deployments is three years from installation. Technical support is available twenty-four hours a day.) Tj\nET\n"
        "endstream\nendobj\n"
        "7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        "xref\n0 8\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000240 00000 n \n0000000365 00000 n \n0000000615 00000 n \n0000000845 00000 n \ntrailer << /Size 8 /Root 1 0 R >>\nstartxref\n920\n%%EOF"
    )
    return pdf_content.encode("latin-1")

def run_e2e_tests():
    print("[1] Testing Health Endpoint...")
    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        res = client.get("/api/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        print("  -> Health Check Passed:", res.json())

        print("\n[2] Testing PDF Upload...")
        pdf_data = create_sample_pdf_bytes()
        files = {"file": ("enterprise_agreement.pdf", pdf_data, "application/pdf")}
        res = client.post("/api/documents/upload", files=files)
        assert res.status_code == 200, f"Upload failed: {res.text}"
        upload_json = res.json()
        doc_id = upload_json["doc_id"]
        print(f"  -> Uploaded successfully: doc_id={doc_id}, pages={upload_json['total_pages']}, chunks={upload_json['total_chunks']}")

        print("\n[3] Testing Document List...")
        res = client.get("/api/documents")
        assert res.status_code == 200
        docs = res.json()
        assert len(docs) >= 1
        print(f"  -> Total documents listed: {len(docs)}")

        print("\n[4] Testing Synchronous Query (First run - Cache Miss)...")
        query_payload = {"question": "What is the warranty period for enterprise deployments?", "doc_id": doc_id}
        res = client.post("/api/query", json=query_payload)
        assert res.status_code == 200, f"Query failed: {res.text}"
        q_data = res.json()
        print("  -> Answer:", q_data["answer"])
        print(f"  -> Cached: {q_data['is_cached']}, Execution Time: {q_data['execution_time_ms']}ms, Citations: {len(q_data['citations'])}")
        assert "three years" in q_data["answer"].lower() or len(q_data["citations"]) > 0

        print("\n[5] Testing Synchronous Query (Second run - Expect Cache HIT)...")
        res2 = client.post("/api/query", json=query_payload)
        assert res2.status_code == 200
        q_data2 = res2.json()
        print(f"  -> Cached: {q_data2['is_cached']}, Execution Time: {q_data2['execution_time_ms']}ms")
        assert q_data2["is_cached"] is True, "Second identical query should be a cache hit!"

        print("\n[6] Testing SSE Streaming Query...")
        stream_payload = {"question": "What standards does the project architecture follow?", "doc_id": doc_id}
        tokens = []
        with client.stream("POST", "/api/query/stream", json=stream_payload) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line.replace("data: ", ""))
                    if event["type"] == "token":
                        tokens.append(event["content"])
        full_stream_answer = "".join(tokens)
        print("  -> Streamed Answer:", full_stream_answer)
        assert len(full_stream_answer) > 0

        print("\n[7] Testing Cache Stats...")
        res = client.get("/api/cache/stats")
        assert res.status_code == 200
        stats = res.json()
        print(f"  -> Cache Stats: {stats['hits']} hits, {stats['misses']} misses, Hit rate: {stats['hit_rate_percentage']}%")
        assert stats["hits"] > 0

        print("\n[8] Testing Delete Document...")
        res = client.delete(f"/api/documents/{doc_id}")
        assert res.status_code == 200
        print(f"  -> Successfully deleted document {doc_id}")

        print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY! [SUCCESS]")

if __name__ == "__main__":
    run_e2e_tests()
