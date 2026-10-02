document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chat-history");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const sampleQueryList = document.getElementById("sample-query-list");
    const clearChatBtn = document.getElementById("clear-chat-btn");
    const dbTypeElem = document.getElementById("db-type");
    const chunkCountElem = document.getElementById("chunk-count");
    const llmNameElem = document.getElementById("llm-name");
    const welcomeBanner = document.getElementById("welcome-banner");

    let isProcessing = false;

    // 1. Fetch Health and Metadata
    async function loadHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                dbTypeElem.textContent = (data.vector_store || "ChromaDB").toUpperCase();
                chunkCountElem.textContent = `${data.indexed_vectors || 123} chunks`;
                llmNameElem.textContent = `${data.llm_provider || 'LangGraph'}`;
            }
        } catch (e) {
            console.warn("Could not load health endpoint:", e);
            dbTypeElem.textContent = "CHROMA";
            chunkCountElem.textContent = "123 chunks";
        }
    }

    // 2. Fetch Sample Queries
    async function loadSampleQueries() {
        try {
            const res = await fetch("/api/sample-queries");
            if (res.ok) {
                const queries = await res.json();
                sampleQueryList.innerHTML = "";
                queries.slice(0, 6).forEach((item) => {
                    const btn = document.createElement("button");
                    btn.className = "query-btn";
                    btn.innerHTML = `<strong>#${item.id}</strong> ${escapeHtml(item.question)}`;
                    btn.addEventListener("click", () => {
                        userInput.value = item.question;
                        chatForm.dispatchEvent(new Event("submit"));
                    });
                    sampleQueryList.appendChild(btn);
                });
            }
        } catch (e) {
            console.warn("Could not load sample queries:", e);
        }
    }

    // 3. Clear Chat History
    clearChatBtn.addEventListener("click", () => {
        chatHistory.innerHTML = "";
        if (welcomeBanner) {
            chatHistory.appendChild(welcomeBanner);
        }
    });

    // 4. Submit Chat
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text || isProcessing) return;

        // Hide welcome banner if visible
        if (welcomeBanner && welcomeBanner.parentNode) {
            welcomeBanner.remove();
        }

        // Append User Message
        appendMessage("user", text);
        userInput.value = "";
        userInput.focus();

        // Append Loading Message
        const loadingId = appendLoading();
        setProcessing(true);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question: text })
            });

            removeLoading(loadingId);

            if (!res.ok) {
                const err = await res.json();
                appendMessage("bot", `⚠️ Error: ${err.detail || "Server error occurred."}`);
                return;
            }

            const data = await res.json();
            appendBotResponse(data);
        } catch (err) {
            removeLoading(loadingId);
            appendMessage("bot", `⚠️ Failed to connect to server: ${err.message}`);
        } finally {
            setProcessing(false);
            scrollToBottom();
        }
    });

    function setProcessing(loading) {
        isProcessing = loading;
        sendBtn.disabled = loading;
        userInput.disabled = loading;
    }

    function appendMessage(role, content) {
        const row = document.createElement("div");
        row.className = `message-row ${role}-row`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = role === "user" ? "👤" : "🤖";

        const wrapper = document.createElement("div");
        wrapper.className = "msg-content-wrapper";

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.textContent = content;

        wrapper.appendChild(bubble);
        row.appendChild(avatar);
        row.appendChild(wrapper);
        chatHistory.appendChild(row);
        scrollToBottom();
        return row;
    }

    function appendLoading() {
        const id = "loading-" + Date.now();
        const row = document.createElement("div");
        row.id = id;
        row.className = "message-row bot-row";

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = "🤖";

        const wrapper = document.createElement("div");
        wrapper.className = "msg-content-wrapper";

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.innerHTML = `
            <div class="typing-dots">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;

        wrapper.appendChild(bubble);
        row.appendChild(avatar);
        row.appendChild(wrapper);
        chatHistory.appendChild(row);
        scrollToBottom();
        return id;
    }

    function removeLoading(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendBotResponse(data) {
        const row = document.createElement("div");
        row.className = "message-row bot-row";

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = "🤖";

        const wrapper = document.createElement("div");
        wrapper.className = "msg-content-wrapper";

        // Answer Bubble
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        bubble.textContent = data.final_answer;
        wrapper.appendChild(bubble);

        // Response Meta: Confidence & Chunks Accordion Toggle
        const metaBar = document.createElement("div");
        metaBar.className = "response-meta";

        const confClass = (data.confidence_level || "low").toLowerCase();
        const confPill = document.createElement("span");
        confPill.className = `confidence-pill ${confClass}`;
        confPill.innerHTML = `<span>🎯 Confidence: ${data.confidence_score} (${data.confidence_level})</span>`;
        metaBar.appendChild(confPill);

        if (data.is_grounded) {
            const groundPill = document.createElement("span");
            groundPill.className = "confidence-pill high";
            groundPill.innerHTML = "<span>🛡️ Strictly Grounded</span>";
            metaBar.appendChild(groundPill);
        }

        const chunks = data.retrieved_chunks || [];
        if (chunks.length > 0) {
            const toggleBtn = document.createElement("button");
            toggleBtn.className = "chunks-toggle-btn";
            toggleBtn.textContent = `📚 View ${chunks.length} Retrieved Chunks ▼`;
            
            const chunksContainer = document.createElement("div");
            chunksContainer.className = "chunks-container";
            chunksContainer.style.display = "none";

            chunks.forEach((c, idx) => {
                const chunkItem = document.createElement("div");
                chunkItem.className = "chunk-item";
                chunkItem.innerHTML = `
                    <div class="chunk-header">
                        <span class="chunk-page-badge">Page ${c.page || 'N/A'}</span>
                        <span class="chunk-score">Score: ${c.relevance_score || '0.0'}</span>
                    </div>
                    <div class="chunk-chapter">${escapeHtml(c.chapter || 'Excerpt')}</div>
                    <div class="chunk-text">${escapeHtml(c.text)}</div>
                `;
                chunksContainer.appendChild(chunkItem);
            });

            toggleBtn.addEventListener("click", () => {
                const isOpen = chunksContainer.style.display === "flex";
                chunksContainer.style.display = isOpen ? "none" : "flex";
                toggleBtn.textContent = isOpen 
                    ? `📚 View ${chunks.length} Retrieved Chunks ▼`
                    : `📚 Hide ${chunks.length} Retrieved Chunks ▲`;
                scrollToBottom();
            });

            metaBar.appendChild(toggleBtn);
            wrapper.appendChild(metaBar);
            wrapper.appendChild(chunksContainer);
        } else {
            wrapper.appendChild(metaBar);
        }

        row.appendChild(avatar);
        row.appendChild(wrapper);
        chatHistory.appendChild(row);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function escapeHtml(text) {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initialize
    loadHealth();
    loadSampleQueries();
});
