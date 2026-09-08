/**
 * AskMyPDF - Modern Frontend Logic
 * Supports SSE streaming, document management, drag-and-drop, and caching stats.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let isQuerying = false;

    // Elements
    const dropzone = document.getElementById("pdf-dropzone");
    const fileInput = document.getElementById("file-input");
    const progressBar = document.getElementById("upload-progress-bar");
    const progressFill = document.getElementById("progress-fill");
    const docCountEl = document.getElementById("doc-count");
    const documentsListEl = document.getElementById("documents-list");
    const emptyDocsMsg = document.getElementById("empty-docs-msg");
    const chatFeed = document.getElementById("chat-feed");
    const welcomeCard = document.getElementById("welcome-card");
    const chatForm = document.getElementById("chat-form");
    const queryInput = document.getElementById("query-input");
    const btnSend = document.getElementById("btn-send");
    const btnClearChat = document.getElementById("btn-clear-chat");
    const btnClearCache = document.getElementById("btn-clear-cache");
    const valCacheRate = document.getElementById("val-cache-rate");
    const toast = document.getElementById("toast");

    // Initialize
    loadDocuments();
    loadCacheStats();

    // Auto-resize input
    queryInput.addEventListener("input", () => {
        queryInput.style.height = "auto";
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
    });

    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event("submit"));
        }
    });

    // Starter Prompt Chips
    document.querySelectorAll(".prompt-chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const prompt = chip.getAttribute("data-prompt");
            if (prompt && !isQuerying) {
                queryInput.value = prompt;
                chatForm.dispatchEvent(new Event("submit"));
            }
        });
    });

    // Drag & Drop Handling
    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showToast("Only PDF files are supported.", "error");
            return;
        }

        const currentCount = parseInt(docCountEl.textContent || "0", 10);
        if (currentCount >= 10) {
            showToast("Document limit reached (maximum 10). Please delete a document first.", "error");
            return;
        }

        if (file.size > 25 * 1024 * 1024) {
            showToast("File exceeds the 25 MB cumulative storage limit.", "error");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        progressFill.style.width = "40%";
        dropzone.style.pointerEvents = "none";

        try {
            const resp = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
            });

            progressFill.style.width = "80%";
            const data = await resp.json();

            if (!resp.ok) {
                throw new Error(data.detail || "Upload failed");
            }

            progressFill.style.width = "100%";
            setTimeout(() => {
                progressFill.style.width = "0%";
                dropzone.style.pointerEvents = "auto";
            }, 500);

            showToast(`Indexed '${data.filename}' (${data.total_chunks} chunks)`, "success");
            fileInput.value = "";
            await loadDocuments();
        } catch (err) {
            progressFill.style.width = "0%";
            dropzone.style.pointerEvents = "auto";
            showToast(err.message, "error");
        }
    }

    // Document Management
    async function loadDocuments() {
        try {
            const resp = await fetch("/api/documents");
            const docs = await resp.json();
            renderDocumentsList(docs);
        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    function renderDocumentsList(docs) {
        docCountEl.textContent = docs.length;

        if (docs.length === 0) {
            emptyDocsMsg.style.display = "block";
            // Remove existing cards
            document.querySelectorAll(".doc-card").forEach((c) => c.remove());
            return;
        }

        emptyDocsMsg.style.display = "none";
        // Remove existing cards
        document.querySelectorAll(".doc-card").forEach((c) => c.remove());

        docs.forEach((doc) => {
            const card = document.createElement("div");
            card.className = "doc-card";
            card.dataset.docId = doc.doc_id;

            card.innerHTML = `
                <div class="doc-card-header">
                    <span class="doc-title" title="${doc.filename}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
                        ${escapeHtml(doc.filename)}
                    </span>
                    <button class="btn-delete-doc" data-doc-id="${doc.doc_id}" title="Delete document">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
                <div class="doc-meta">
                    <span>${doc.total_pages} pages</span>
                    <span>•</span>
                    <span>${doc.total_chunks} chunks</span>
                </div>
            `;

            card.querySelector(".btn-delete-doc").addEventListener("click", (e) => {
                e.stopPropagation();
                deleteDocument(doc.doc_id, doc.filename);
            });

            documentsListEl.appendChild(card);
        });
    }

    async function deleteDocument(docId, filename) {
        if (!confirm(`Are you sure you want to delete '${filename}'?`)) return;

        try {
            const resp = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
            if (resp.ok) {
                showToast(`Deleted '${filename}'`, "success");
                await loadDocuments();
            } else {
                throw new Error("Failed to delete document");
            }
        } catch (err) {
            showToast(err.message, "error");
        }
    }

    // Cache Stats & Flush
    async function loadCacheStats() {
        if (!valCacheRate) return;
        try {
            const resp = await fetch("/api/cache/stats");
            const data = await resp.json();
            valCacheRate.textContent = `${data.hit_rate_percentage}% (${data.hits}H/${data.misses}M)`;
        } catch (err) {
            console.error("Failed to load cache stats:", err);
        }
    }

    if (btnClearCache) {
        btnClearCache.addEventListener("click", async () => {
            try {
                const resp = await fetch("/api/cache/clear", { method: "POST" });
                if (resp.ok) {
                    showToast("SQLite cache flushed successfully", "success");
                    loadCacheStats();
                }
            } catch (err) {
                showToast("Failed to flush cache", "error");
            }
        });
    }

    // Clear Chat
    if (btnClearChat) {
        btnClearChat.addEventListener("click", () => {
            const messages = chatFeed.querySelectorAll(".chat-message");
            messages.forEach((m) => m.remove());
            if (welcomeCard) welcomeCard.style.display = "block";
        });
    }

    // Chat Submission & Streaming
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query || isQuerying) return;

        if (welcomeCard) welcomeCard.style.display = "none";

        // Append user bubble
        appendUserMessage(query);
        queryInput.value = "";
        queryInput.style.height = "auto";

        // Create placeholder assistant bubble
        const { bubbleEl, contentEl, metaEl, citationsBoxEl } = createAssistantMessage();

        isQuerying = true;
        btnSend.disabled = true;

        try {
            const resp = await fetch("/api/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: query,
                    doc_id: null,
                }),
            });

            if (!resp.ok) {
                throw new Error(`Server returned error: ${resp.statusText}`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let answerText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // Keep incomplete piece in buffer

                for (const block of lines) {
                    if (block.startsWith("data: ")) {
                        const jsonStr = block.replace("data: ", "").trim();
                        if (!jsonStr) continue;

                        try {
                            const event = JSON.parse(jsonStr);

                            if (event.type === "sources") {
                                renderCitations(citationsBoxEl, event.citations);
                                if (event.is_cached) {
                                    metaEl.innerHTML = `<span class="cache-tag">⚡ Cache Hit</span>`;
                                }
                            } else if (event.type === "token") {
                                answerText += event.content;
                                contentEl.textContent = answerText;
                                chatFeed.scrollTop = chatFeed.scrollHeight;
                            } else if (event.type === "done") {
                                const cacheTag = event.is_cached ? `<span class="cache-tag">⚡ Cache Hit</span>` : "";
                                metaEl.innerHTML = `${cacheTag} <span>${event.execution_time_ms}ms • ${event.llm_provider}</span>`;
                                loadCacheStats();
                            } else if (event.type === "error") {
                                contentEl.textContent = `Error: ${event.message}`;
                            }
                        } catch (parseErr) {
                            console.error("Failed to parse SSE line:", block, parseErr);
                        }
                    }
                }
            }

            // Remove typing cursor indicator
            const cursor = bubbleEl.querySelector(".typing-cursor");
            if (cursor) cursor.remove();

        } catch (err) {
            contentEl.textContent = `Error: ${err.message}`;
        } finally {
            isQuerying = false;
            btnSend.disabled = false;
            chatFeed.scrollTop = chatFeed.scrollHeight;
        }
    });

    function appendUserMessage(text) {
        const msg = document.createElement("div");
        msg.className = "chat-message user";
        msg.innerHTML = `
            <div class="msg-bubble">
                <div class="msg-content">${escapeHtml(text)}</div>
            </div>
            <div class="msg-avatar">You</div>
        `;
        chatFeed.appendChild(msg);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function createAssistantMessage() {
        const msg = document.createElement("div");
        msg.className = "chat-message assistant";
        msg.innerHTML = `
            <div class="msg-avatar">AI</div>
            <div class="msg-bubble">
                <div class="msg-header">
                    <span class="msg-sender">AskYourDoc Assistant</span>
                    <div class="msg-meta-badge">Thinking...</div>
                </div>
                <div class="msg-content"><span class="typing-cursor"></span></div>
                <div class="citations-box" style="display: none;"></div>
            </div>
        `;
        chatFeed.appendChild(msg);
        chatFeed.scrollTop = chatFeed.scrollHeight;

        return {
            bubbleEl: msg.querySelector(".msg-bubble"),
            contentEl: msg.querySelector(".msg-content"),
            metaEl: msg.querySelector(".msg-meta-badge"),
            citationsBoxEl: msg.querySelector(".citations-box"),
        };
    }

    function renderCitations(container, citations) {
        if (!citations || citations.length === 0) return;

        container.style.display = "block";
        const avgScore = Math.round(
            (citations.reduce((sum, c) => sum + (c.similarity_score || 0), 0) / citations.length) * 100
        );

        container.innerHTML = `
            <div class="citations-header">
                <span>📚 Grounded Sources (${citations.length}) • ${avgScore}% Avg Match</span>
                <span class="accordion-arrow">▼</span>
            </div>
            <div class="citations-list">
                ${citations
                    .map(
                        (c) => `
                    <div class="citation-item">
                        <div class="citation-meta">
                            <span class="citation-page">${escapeHtml(c.filename)} (Page ${c.page_number})</span>
                            <span class="citation-score">Relevance: ${Math.round((c.similarity_score || 0) * 100)}%</span>
                        </div>
                        <p class="citation-text">"${escapeHtml(c.excerpt)}"</p>
                    </div>
                `
                    )
                    .join("")}
            </div>
        `;

        const header = container.querySelector(".citations-header");
        const list = container.querySelector(".citations-list");
        header.addEventListener("click", () => {
            const isHidden = list.style.display === "none";
            list.style.display = isHidden ? "flex" : "none";
            header.querySelector(".accordion-arrow").textContent = isHidden ? "▼" : "▶";
        });
    }

    // Helper Toast
    function showToast(message, type = "info") {
        toast.textContent = message;
        toast.className = `toast ${type}`;
        setTimeout(() => {
            toast.className = "toast hidden";
        }, 3500);
    }

    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
