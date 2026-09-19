/**
 * Applied AI Document QA & RAG System - Frontend Application Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");
  const documentList = document.getElementById("documentList");
  const docCountBadge = document.getElementById("docCountBadge");
  const providerSelect = document.getElementById("providerSelect");
  const topKSlider = document.getElementById("topKSlider");
  const topKValue = document.getElementById("topKValue");
  const queryForm = document.getElementById("queryForm");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeCard = document.getElementById("welcomeCard");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const systemStatusText = document.getElementById("systemStatusText");
  const latencySummary = document.getElementById("latencySummary");

  // State
  let documents = [];

  // Init
  init();

  function init() {
    setupEventListeners();
    fetchHealth();
    fetchDocuments();
  }

  function setupEventListeners() {
    // Slider
    topKSlider.addEventListener("input", (e) => {
      topKValue.textContent = e.target.value;
    });

    // Dropzone
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });

    // Query Form
    queryForm.addEventListener("submit", handleQuerySubmit);

    // Auto-resize textarea
    questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        queryForm.dispatchEvent(new Event("submit"));
      }
    });

    // Quick Prompts
    document.querySelectorAll(".qp-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const q = btn.getAttribute("data-q");
        questionInput.value = q;
        queryForm.dispatchEvent(new Event("submit"));
      });
    });

    // Clear Chat
    clearChatBtn.addEventListener("click", () => {
      messagesContainer.innerHTML = "";
      messagesContainer.appendChild(welcomeCard);
      welcomeCard.classList.remove("hidden");
      latencySummary.textContent = "";
    });
  }

  // Health check
  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        systemStatusText.textContent = `Online • ${data.total_chunks} chunks`;
      }
    } catch (e) {
      systemStatusText.textContent = "Offline";
    }
  }

  // Fetch Documents
  async function fetchDocuments() {
    try {
      const res = await fetch("/api/documents");
      if (res.ok) {
        documents = await res.json();
        renderDocuments();
      }
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  }

  function renderDocuments() {
    docCountBadge.textContent = documents.length;
    if (documents.length === 0) {
      documentList.innerHTML = `<div class="empty-docs">No documents indexed yet. Upload a PDF/TXT to get started.</div>`;
      return;
    }

    documentList.innerHTML = documents
      .map(
        (doc) => `
      <div class="doc-item" data-id="${doc.doc_id}">
        <div class="doc-meta">
          <span class="doc-name" title="${doc.filename}">${escapeHtml(doc.filename)}</span>
          <span class="doc-sub">${doc.chunk_count} chunks • ${formatBytes(doc.char_count)} chars</span>
        </div>
        <button class="btn-icon-danger" title="Delete document" onclick="window.deleteDoc('${doc.doc_id}')">
          &times;
        </button>
      </div>
    `
      )
      .join("");
  }

  window.deleteDoc = async function (docId) {
    if (!confirm("Are you sure you want to remove this document from the vector index?")) return;
    try {
      const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
      if (res.ok) {
        await fetchDocuments();
        await fetchHealth();
      }
    } catch (e) {
      alert("Failed to delete document.");
    }
  };

  // Upload handler
  async function handleFileUpload(file) {
    uploadStatus.classList.remove("hidden", "success", "error");
    uploadStatus.textContent = `Ingesting & indexing "${file.name}"...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }

      uploadStatus.classList.add("success");
      uploadStatus.textContent = `Indexed ${data.document.chunk_count} chunks from ${file.name}`;
      fileInput.value = "";
      await fetchDocuments();
      await fetchHealth();

      setTimeout(() => {
        uploadStatus.classList.add("hidden");
      }, 4000);
    } catch (err) {
      uploadStatus.classList.add("error");
      uploadStatus.textContent = `Error: ${err.message}`;
    }
  }

  // Handle Query
  async function handleQuerySubmit(e) {
    e.preventDefault();
    const query = questionInput.value.trim();
    if (!query) return;

    if (documents.length === 0) {
      alert("Please index at least one document before asking questions.");
      return;
    }

    // Hide welcome card
    welcomeCard.classList.add("hidden");

    // Append User Message
    appendUserMessage(query);
    questionInput.value = "";
    sendBtn.disabled = true;

    // Append Assistant Loading Skeleton
    const loadingCardId = "loading-" + Date.now();
    appendLoadingMessage(loadingCardId);

    try {
      const payload = {
        question: query,
        top_k: parseInt(topKSlider.value, 10),
        provider: providerSelect.value,
      };

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      removeElement(loadingCardId);

      if (!res.ok) {
        appendAssistantMessage({
          answer: `Error: ${data.detail || "Query processing failed."}`,
          model_name: "Error",
          retrieval_latency_ms: 0,
          generation_latency_ms: 0,
          total_latency_ms: 0,
          citations: [],
        });
        return;
      }

      appendAssistantMessage(data);
      latencySummary.textContent = `Last query: ${data.total_latency_ms}ms (${data.model_name})`;
    } catch (err) {
      removeElement(loadingCardId);
      appendAssistantMessage({
        answer: `Network Error: ${err.message}`,
        model_name: "Error",
        retrieval_latency_ms: 0,
        generation_latency_ms: 0,
        total_latency_ms: 0,
        citations: [],
      });
    } finally {
      sendBtn.disabled = false;
      questionInput.focus();
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function appendLoadingMessage(id) {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.id = id;
    row.innerHTML = `
      <div class="assistant-card">
        <div class="assistant-header">
          <span class="assistant-tag">&#9881; Retrieving & Synthesizing...</span>
        </div>
        <div class="assistant-body" style="color:#888;">
          Searching vector database and generating source-grounded response...
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function appendAssistantMessage(data) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const citationsHtml =
      data.citations && data.citations.length > 0
        ? `
      <div class="citations-box">
        <button class="citations-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
          &#128269; View ${data.citations.length} Grounded Context Source(s) &darr;
        </button>
        <div class="citations-list hidden">
          ${data.citations
            .map(
              (c) => `
            <div class="citation-card">
              <div class="citation-header">
                <span>${escapeHtml(c.source_name)} (Page ${c.page_number}, Chunk #${c.chunk_index})</span>
                <span class="similarity-badge">${(c.similarity_score * 100).toFixed(1)}% Match</span>
              </div>
              <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    `
        : "";

    row.innerHTML = `
      <div class="assistant-card">
        <div class="assistant-header">
          <span class="assistant-tag">&#10024; ${escapeHtml(data.model_name)}</span>
          <div class="latency-pills">
            <span class="pill" title="Vector retrieval time">Retrieval: ${data.retrieval_latency_ms}ms</span>
            <span class="pill" title="Generation time">Gen: ${data.generation_latency_ms}ms</span>
            <span class="pill success" title="Total round-trip time">Total: ${data.total_latency_ms}ms</span>
          </div>
        </div>
        <div class="assistant-body">
          ${formatAnswer(data.answer)}
        </div>
        ${citationsHtml}
      </div>
    `;

    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function formatAnswer(text) {
    // Converts [Source: ...] citations to badges
    const escaped = escapeHtml(text);
    return escaped.replace(
      /\[Source:\s*([^,\]]+)(?:,\s*Page:\s*(\d+))?\]/gi,
      '<span class="badge" style="background:#e0f2fe; color:#0369a1; font-weight:600;">[Source: $1$2 ? ", Page: " + $2 : ""]</span>'
    );
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  function removeElement(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
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

  function formatBytes(chars) {
    if (chars < 1000) return chars + " chars";
    return (chars / 1000).toFixed(1) + "k chars";
  }
});
