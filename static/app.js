/**
 * DocuMind - frontend application logic.
 *
 * Design notes:
 * - All server-supplied strings are inserted with `textContent` or via explicit
 *   `createElement` calls. No `innerHTML` is used with untrusted data, so the
 *   stored-XSS class of bug cannot occur here even if backend sanitisation regresses.
 * - Requests check `res.ok` before parsing, and parse defensively, so a proxy 502
 *   surfaces its real status instead of a JSON syntax error.
 * - In-flight queries are guarded and cancellable; a hung request cannot leave the
 *   UI permanently stuck.
 */

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);

  const els = {
    dropZone: $("dropZone"),
    fileInput: $("fileInput"),
    uploadStatus: $("uploadStatus"),
    documentList: $("documentList"),
    docCountBadge: $("docCountBadge"),
    providerSelect: $("providerSelect"),
    topKSlider: $("topKSlider"),
    topKValue: $("topKValue"),
    queryForm: $("queryForm"),
    questionInput: $("questionInput"),
    sendBtn: $("sendBtn"),
    cancelBtn: $("cancelBtn"),
    messages: $("messagesContainer"),
    welcome: $("welcomeCard"),
    clearChatBtn: $("clearChatBtn"),
    systemStatusText: $("systemStatusText"),
    statusDot: $("statusDot"),
    latencySummary: $("latencySummary"),
  };

  const API_BASE = document.body.dataset.apiBase || "/api";

  const state = {
    documents: [],
    inFlight: null, // AbortController for the active query, if any
    uploadTimer: null,
    dragDepth: 0,
  };

  init();

  // ------------------------------------------------------------------ helpers

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  /** Reads a JSON body, tolerating non-JSON error pages from proxies. */
  async function readJson(res) {
    const text = await res.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return null;
    }
  }

  async function apiFetch(path, options = {}) {
    const res = await fetch(`${API_BASE}${path}`, options);
    const data = await readJson(res);
    if (!res.ok) {
      const detail = (data && data.detail) || `HTTP ${res.status}`;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function formatCharCount(chars) {
    if (chars === null || chars === undefined) return "-";
    if (chars < 1000) return `${chars} chars`;
    return `${(chars / 1000).toFixed(1)}k chars`;
  }

  function scrollToBottom() {
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function setStatus(text, tone) {
    els.uploadStatus.textContent = text;
    els.uploadStatus.className = tone ? tone : "";
  }

  // --------------------------------------------------------------------- init

  function init() {
    setupEventListeners();
    fetchHealth();
    fetchDocuments();
  }

  function setupEventListeners() {
    els.topKSlider.addEventListener("input", (e) => {
      els.topKValue.textContent = e.target.value;
    });

    // Drop zone. It is a <button> in the markup so it is focusable and operable
    // from the keyboard; click and keydown would both be redundant if it were a div.
    els.dropZone.addEventListener("click", () => els.fileInput.click());

    // dragenter/dragleave fire for every descendant, so a depth counter is needed
    // or the highlight flickers whenever the pointer crosses a child element.
    els.dropZone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      state.dragDepth += 1;
      els.dropZone.classList.add("drag-over");
    });
    els.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
    });
    els.dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      state.dragDepth = Math.max(0, state.dragDepth - 1);
      if (state.dragDepth === 0) els.dropZone.classList.remove("drag-over");
    });
    els.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      state.dragDepth = 0;
      els.dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
    });

    // Prevent the browser from navigating away when a file is dropped outside
    // the drop zone, which would destroy all UI state.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

    els.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
    });

    els.queryForm.addEventListener("submit", handleQuerySubmit);

    els.questionInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        // requestSubmit runs constraint validation; dispatchEvent does not, which
        // would silently bypass the `required` attribute.
        els.queryForm.requestSubmit();
      }
    });

    document.querySelectorAll(".qp-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        els.questionInput.value = btn.dataset.q || "";
        els.queryForm.requestSubmit();
      });
    });

    els.cancelBtn.addEventListener("click", cancelQuery);

    els.clearChatBtn.addEventListener("click", () => {
      els.messages.replaceChildren(els.welcome);
      els.welcome.classList.remove("hidden");
      els.latencySummary.textContent = "";
    });
  }

  // ------------------------------------------------------------------- health

  async function fetchHealth() {
    try {
      const data = await apiFetch("/health");
      els.systemStatusText.textContent = `Online • ${data.total_chunks} chunks indexed`;
      els.statusDot.classList.remove("offline");

      // Reconcile the controls with server truth instead of assuming defaults.
      if (data.active_provider) els.providerSelect.value = data.active_provider;
      els.dropZone.dataset.embedding = data.embedding_backend || "";
    } catch (e) {
      els.systemStatusText.textContent = "Offline";
      els.statusDot.classList.add("offline");
    }
  }

  // ---------------------------------------------------------------- documents

  async function fetchDocuments() {
    try {
      state.documents = (await apiFetch("/documents")) || [];
      renderDocuments();
    } catch (err) {
      console.error("Failed to fetch documents:", err);
      els.documentList.replaceChildren(
        el("div", "empty-docs", "Could not reach the API. Is the server running?")
      );
    }
  }

  function renderDocuments() {
    els.docCountBadge.textContent = state.documents.length;
    els.documentList.replaceChildren();

    if (state.documents.length === 0) {
      els.documentList.appendChild(
        el("div", "empty-docs", "No documents indexed yet. Upload a PDF, TXT, MD or CSV to begin.")
      );
      return;
    }

    for (const doc of state.documents) {
      const item = el("div", "doc-item");
      item.dataset.id = doc.doc_id;

      const meta = el("div", "doc-meta");

      const name = el("span", "doc-name", doc.filename);
      // setAttribute, not an interpolated string: this is the sink that made the
      // previous version vulnerable to attribute injection via the filename.
      name.title = doc.filename;
      meta.appendChild(name);

      const chunkCount = Number(doc.chunk_count) || 0;
      const sub = el(
        "span",
        "doc-sub",
        `${chunkCount} chunk${chunkCount === 1 ? "" : "s"} • ${formatCharCount(doc.char_count)}`
      );
      meta.appendChild(sub);
      item.appendChild(meta);

      const del = el("button", "btn-icon-danger", "×");
      del.type = "button";
      del.title = `Delete ${doc.filename}`;
      del.setAttribute("aria-label", `Delete document ${doc.filename}`);
      del.addEventListener("click", () => handleDelete(doc));
      item.appendChild(del);

      els.documentList.appendChild(item);
    }
  }

  async function handleDelete(doc) {
    const ok = window.confirm(`Remove "${doc.filename}" and its embeddings from the index?`);
    if (!ok) return;
    try {
      await apiFetch(`/documents/${encodeURIComponent(doc.doc_id)}`, { method: "DELETE" });
      await fetchDocuments();
      await fetchHealth();
    } catch (e) {
      setStatus(`Could not delete ${doc.filename}: ${e.message}`, "error");
    }
  }

  // ------------------------------------------------------------------ uploads

  function scheduleStatusClear() {
    if (state.uploadTimer) clearTimeout(state.uploadTimer);
    state.uploadTimer = setTimeout(() => {
      els.uploadStatus.className = "hidden";
      state.uploadTimer = null;
    }, 4000);
  }

  async function handleFileUpload(file) {
    if (state.uploadTimer) {
      clearTimeout(state.uploadTimer);
      state.uploadTimer = null;
    }
    setStatus(`Ingesting and indexing "${file.name}"…`, "info");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const data = await apiFetch("/upload", { method: "POST", body: formData });
      setStatus(
        `Indexed ${data.document.chunk_count} chunks from ${file.name}`,
        "success"
      );
      scheduleStatusClear();
      await Promise.all([fetchDocuments(), fetchHealth()]);
    } catch (err) {
      setStatus(`Upload failed: ${err.message}`, "error");
    } finally {
      // Reset on both paths. Leaving a stale value means re-picking the same
      // failed file fires no change event and appears to do nothing.
      els.fileInput.value = "";
    }
  }

  // ------------------------------------------------------------------- queries

  function setBusy(busy) {
    els.sendBtn.disabled = busy;
    els.cancelBtn.hidden = !busy;
    els.sendBtn.setAttribute("aria-busy", String(busy));
  }

  function cancelQuery() {
    if (state.inFlight) {
      state.inFlight.abort();
      state.inFlight = null;
    }
  }

  async function handleQuerySubmit(e) {
    e.preventDefault();
    if (state.inFlight) return; // guard: Enter, quick prompts and the button can all fire

    const query = els.questionInput.value.trim();
    if (!query) return;

    if (state.documents.length === 0) {
      setStatus("Index at least one document before asking questions.", "error");
      els.questionInput.focus();
      return;
    }

    els.welcome.classList.add("hidden");
    appendUserMessage(query);
    els.questionInput.value = "";

    const controller = new AbortController();
    state.inFlight = controller;
    setBusy(true);

    const loadingRow = appendLoadingMessage();

    try {
      const data = await apiFetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          top_k: Number(els.topKSlider.value),
          provider: els.providerSelect.value,
        }),
        signal: controller.signal,
      });
      loadingRow.remove();
      appendAssistantMessage(data);
    } catch (err) {
      loadingRow.remove();
      const aborted = err.name === "AbortError";
      appendAssistantMessage({
        answer: aborted
          ? "Query cancelled."
          : `Request failed: ${err.message}`,
        model_name: aborted ? "Cancelled" : "Error",
        is_grounded: false,
        citations: [],
        embedding_latency_ms: 0,
        search_latency_ms: 0,
        generation_latency_ms: 0,
        total_latency_ms: 0,
      });
    } finally {
      state.inFlight = null;
      setBusy(false);
      els.questionInput.focus();
    }
  }

  function appendUserMessage(text) {
    const row = el("div", "message-row user");
    row.appendChild(el("div", "user-bubble", text));
    els.messages.appendChild(row);
    scrollToBottom();
  }

  function appendLoadingMessage() {
    const row = el("div", "message-row assistant");
    row.setAttribute("aria-hidden", "true");

    const card = el("div", "assistant-card");
    const header = el("div", "assistant-header");
    header.appendChild(el("span", "assistant-tag", "Retrieving and synthesising…"));
    card.appendChild(header);
    card.appendChild(
      el("div", "assistant-body muted", "Searching the vector index and building a grounded response.")
    );
    row.appendChild(card);
    els.messages.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendAssistantMessage(data) {
    const row = el("div", "message-row assistant");
    const card = el("div", "assistant-card");

    // --- header: model tag + latency pills
    const header = el("div", "assistant-header");
    header.appendChild(el("span", "assistant-tag", data.model_name || "Response"));

    const pills = el("div", "latency-pills");
    const addPill = (label, value, title, cls) => {
      const p = el("span", cls ? `pill ${cls}` : "pill", `${label} ${value}ms`);
      p.title = title;
      pills.appendChild(p);
    };
    addPill("Embed", data.embedding_latency_ms, "Query embedding (transformer inference)", "");
    addPill("Search", data.search_latency_ms, "Vector similarity search only", "");
    addPill("Gen", data.generation_latency_ms, "Generation time", "");
    addPill("Total", data.total_latency_ms, "Total round-trip time", "success");
    header.appendChild(pills);
    card.appendChild(header);

    // --- body
    const grounded = data.is_grounded !== false;
    const body = el("div", "assistant-body");
    if (!grounded) body.appendChild(el("p", "refusal-note", "No sufficiently relevant context found."));
    body.appendChild(renderAnswer(data.answer || ""));
    card.appendChild(body);

    // --- citations
    if (Array.isArray(data.citations) && data.citations.length > 0) {
      card.appendChild(buildCitations(data.citations));
    }

    row.appendChild(card);
    els.messages.appendChild(row);
    scrollToBottom();
  }

  function buildCitations(citations) {
    const box = el("div", "citations-box");

    const toggle = el(
      "button",
      "citations-toggle",
      `View ${citations.length} grounded source${citations.length === 1 ? "" : "s"}`
    );
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");

    const list = el("div", "citations-list hidden");
    for (const c of citations) {
      const card = el("div", "citation-card");
      const head = el("div", "citation-header");
      head.appendChild(
        el("span", null, `${c.source_name} (Page ${c.page_number}, Chunk #${c.chunk_index})`)
      );
      head.appendChild(
        el("span", "similarity-badge", `${(c.similarity_score * 100).toFixed(1)}% match`)
      );
      card.appendChild(head);
      card.appendChild(el("blockquote", "citation-snippet", c.snippet));
      list.appendChild(card);
    }

    toggle.addEventListener("click", () => {
      const expanded = toggle.getAttribute("aria-expanded") === "true";
      toggle.setAttribute("aria-expanded", String(!expanded));
      list.classList.toggle("hidden", expanded);
    });

    box.appendChild(toggle);
    box.appendChild(list);
    return box;
  }

  /**
   * Renders an answer, turning `[Source: name, Page: N]` markers into badges.
   *
   * Built as DOM nodes rather than a regex replacement over an HTML string: a
   * replacement string only interprets `$$`/`$&`/`$1`, so a conditional expression
   * pasted into one is emitted literally instead of evaluated.
   */
  function renderAnswer(text) {
    const frag = document.createDocumentFragment();
    const pattern = /\[Source:\s*([^,\]]+?)(?:,\s*Page:\s*(\d+))?\]/gi;
    let last = 0;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const label = match[2] ? `${match[1]}, Page ${match[2]}` : match[1];
      frag.appendChild(el("span", "badge", `[Source: ${label}]`));
      last = pattern.lastIndex;
    }
    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }
    return frag;
  }
});
