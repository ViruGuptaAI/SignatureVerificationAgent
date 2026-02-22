/* ===================================================================
   Signature Verification Agent — Frontend Logic
   =================================================================== */

(() => {
    "use strict";

    // ---- Tab switching ----
    const tabs = document.querySelectorAll(".tab");
    const modes = document.querySelectorAll(".mode");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            modes.forEach(m => m.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(`mode-${tab.dataset.mode}`).classList.add("active");
        });
    });

    // ================================================================
    //  SINGLE COMPARE
    // ================================================================
    const singleFiles = { img1: null, img2: null };

    setupUploadZone("zone-img1", "input-img1", "preview-img1", "remove-img1", f => {
        singleFiles.img1 = f;
        updateSingleBtn();
    });
    setupUploadZone("zone-img2", "input-img2", "preview-img2", "remove-img2", f => {
        singleFiles.img2 = f;
        updateSingleBtn();
    });

    const btnCompare = document.getElementById("btn-compare");
    function updateSingleBtn() {
        btnCompare.disabled = !(singleFiles.img1 && singleFiles.img2);
    }

    btnCompare.addEventListener("click", async () => {
        btnCompare.disabled = true;
        btnCompare.classList.add("loading");
        hideResult("single-result");
        showWaitingNote("single-result");

        const form = new FormData();
        form.append("image1", singleFiles.img1);
        form.append("image2", singleFiles.img2);

        const model = document.getElementById("single-model").value;
        const reasoning = document.getElementById("single-reasoning").value;
        const preprocess = document.getElementById("single-preprocess").checked;

        const url = `/api/VerifySignature?model=${model}&reasoning_effort=${reasoning}&preprocess=${preprocess}`;

        try {
            const resp = await fetch(url, { method: "POST", body: form });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            renderSingleResult(data);
        } catch (err) {
            renderError("single-result", err.message);
        } finally {
            btnCompare.classList.remove("loading");
            updateSingleBtn();
        }
    });

    function renderSingleResult(data) {
        const r = data.result;
        const card = document.getElementById("single-result");
        const confClass = r.confidence_score >= 0.85 ? "high" : r.confidence_score >= 0.5 ? "mid" : "low";
        const verdictClass = r.signature_matched ? "matched" : "unmatched";
        const verdictLabel = r.signature_matched ? "Signatures Match" : "Signatures Do Not Match";
        const verdictIcon = r.signature_matched ? "✓" : "✗";

        card.innerHTML = `
            <div class="verdict-banner ${verdictClass}">
                <div class="verdict-icon">${verdictIcon}</div>
                <div class="verdict-text">
                    <h2>${verdictLabel}</h2>
                    <div class="verdict-meta">
                        <span>${data.image1} vs ${data.image2}</span>
                    </div>
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">
                    <span>Confidence</span>
                    <span>${(r.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill ${confClass}" style="width:${r.confidence_score * 100}%"></div>
                </div>
            </div>
            <div class="reasoning-section">
                <h3>Analysis</h3>
                <div class="md-content">${mdToHtml(r.reasoning)}</div>
            </div>
            <div class="metrics-row">
                ${metricHtml("TTFB", `${data.timing.ttfb_ms.toFixed(0)} ms`)}
                ${metricHtml("TTFT", `${data.timing.ttft_ms.toFixed(0)} ms`)}
                ${metricHtml("Total", `${data.elapsed_ms.toFixed(0)} ms`)}
                ${tokenMetric("Tokens", data.usage)}
                ${costMetric(data.cost_inr)}
            </div>
        `;
        card.classList.remove("hidden");
    }

    // ================================================================
    //  BATCH VERIFY
    // ================================================================
    let testFile = null;
    const refFiles = [];           // { file, url }
    const MAX_REFS = 10;

    setupUploadZone("zone-test", "input-test", "preview-test", "remove-test", f => {
        testFile = f;
        updateBatchBtn();
    });

    const btnAddRef = document.getElementById("btn-add-ref");
    const inputRefs = document.getElementById("input-refs");
    const refsGrid = document.getElementById("refs-grid");
    const refCount = document.getElementById("ref-count");
    const btnBatch = document.getElementById("btn-batch");

    btnAddRef.addEventListener("click", () => {
        if (refFiles.length >= MAX_REFS) return;
        inputRefs.click();
    });

    inputRefs.addEventListener("change", () => {
        for (const file of inputRefs.files) {
            if (refFiles.length >= MAX_REFS) break;
            addRefFile(file);
        }
        inputRefs.value = "";
        updateBatchBtn();
    });

    function addRefFile(file) {
        const url = URL.createObjectURL(file);
        const idx = refFiles.length;
        refFiles.push({ file, url });

        const thumb = document.createElement("div");
        thumb.className = "ref-thumb";
        thumb.innerHTML = `
            <img src="${url}" alt="${escHtml(file.name)}" />
            <button class="remove-btn" title="Remove">&times;</button>
        `;
        thumb.querySelector(".remove-btn").addEventListener("click", e => {
            e.stopPropagation();
            URL.revokeObjectURL(url);
            const i = refFiles.findIndex(r => r.url === url);
            if (i !== -1) refFiles.splice(i, 1);
            thumb.remove();
            updateRefCount();
            updateBatchBtn();
        });
        refsGrid.appendChild(thumb);
        updateRefCount();
    }

    function updateRefCount() {
        refCount.textContent = `(${refFiles.length}/${MAX_REFS})`;
        btnAddRef.disabled = refFiles.length >= MAX_REFS;
    }

    function updateBatchBtn() {
        btnBatch.disabled = !(testFile && refFiles.length >= 2);
    }

    // Drag-and-drop on refs grid
    refsGrid.addEventListener("dragover", e => { e.preventDefault(); refsGrid.style.outline = "2px dashed var(--accent)"; });
    refsGrid.addEventListener("dragleave", () => { refsGrid.style.outline = ""; });
    refsGrid.addEventListener("drop", e => {
        e.preventDefault();
        refsGrid.style.outline = "";
        for (const file of e.dataTransfer.files) {
            if (refFiles.length >= MAX_REFS) break;
            if (file.type.startsWith("image/")) addRefFile(file);
        }
        updateBatchBtn();
    });

    btnBatch.addEventListener("click", async () => {
        btnBatch.disabled = true;
        btnBatch.classList.add("loading");
        hideResult("batch-result");
        showWaitingNote("batch-result");

        const form = new FormData();
        form.append("test_image", testFile);
        refFiles.forEach((r, i) => form.append(`ref_${i + 1}`, r.file));

        const model = document.getElementById("batch-model").value;
        const reasoning = document.getElementById("batch-reasoning").value;
        const preprocess = document.getElementById("batch-preprocess").checked;

        const url = `/api/VerifySignatureBatch?model=${model}&reasoning_effort=${reasoning}&preprocess=${preprocess}`;

        try {
            const resp = await fetch(url, { method: "POST", body: form });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            renderBatchResult(data);
        } catch (err) {
            renderError("batch-result", err.message);
        } finally {
            btnBatch.classList.remove("loading");
            updateBatchBtn();
        }
    });

    function renderBatchResult(data) {
        const v = data.verdict;
        const confClass = v.avg_confidence >= 0.85 ? "high" : v.avg_confidence >= 0.5 ? "mid" : "low";
        const verdictClass = v.inconclusive ? "inconclusive" : v.signature_matched ? "matched" : "unmatched";
        const verdictLabel = v.inconclusive ? "Inconclusive" : v.signature_matched ? "Signatures Match" : "Signatures Do Not Match";
        const verdictIcon = v.inconclusive ? "?" : v.signature_matched ? "✓" : "✗";

        let individualsHtml = "";
        if (data.individual_results && data.individual_results.length) {
            individualsHtml = `
            <div class="individual-results">
                <h3>Individual Comparisons</h3>
                ${data.individual_results.map((ir, i) => {
                    const statusClass = ir.error ? "error" : ir.signature_matched ? "match" : "no-match";
                    const label = ir.error ? "Error" : ir.signature_matched ? "Match" : "No Match";
                    return `
                    <div class="individual-item-wrap" onclick="this.classList.toggle('expanded')">
                        <div class="individual-item">
                            <div class="ind-status ${statusClass}"></div>
                            <span class="ind-name">${escHtml(ir.reference_filename)}</span>
                            <span class="ind-confidence">${ir.error ? "—" : (ir.confidence_score * 100).toFixed(1) + "%"}</span>
                            <span class="ind-time">${ir.error ? "" : ir.elapsed_ms.toFixed(0) + " ms"}</span>
                            <span class="ind-expand">▼</span>
                        </div>
                        <div class="ind-reasoning md-content">${ir.error ? escHtml(ir.error) : mdToHtml(ir.reasoning)}</div>
                    </div>`;
                }).join("")}
            </div>`;
        }

        const card = document.getElementById("batch-result");
        card.innerHTML = `
            <div class="verdict-banner ${verdictClass}">
                <div class="verdict-icon">${verdictIcon}</div>
                <div class="verdict-text">
                    <h2>${verdictLabel}</h2>
                    <div class="verdict-meta">
                        <span>Match ratio: ${v.match_ratio}</span>
                        <span>Avg confidence: ${(v.avg_confidence * 100).toFixed(1)}%</span>
                        <span class="id-with-info">ID: ${data.request_id}
                            <span class="info-btn" tabindex="0">&#9432;<span class="info-tooltip">Save this ID to look up your result later in the Audit Log tab. It is the unique identifier for this batch comparison.</span></span>
                        </span>
                    </div>
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">
                    <span>Average Confidence</span>
                    <span>${(v.avg_confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill ${confClass}" style="width:${v.avg_confidence * 100}%"></div>
                </div>
            </div>
            <div class="reasoning-section">
                <h3>Summary</h3>
                <div class="md-content">${mdToHtml(v.reasoning)}</div>
            </div>
            <div class="metrics-row">
                ${metricHtml("Comparisons", data.individual_results.length)}
                ${metricHtml("Total Time", `${data.elapsed_ms.toFixed(0)} ms`)}
                ${tokenMetric("Total Tokens", data.total_usage)}
                ${costMetric(data.total_cost_inr)}
                ${metricHtml("Method", v.decision_method)}
            </div>
            ${individualsHtml}
        `;
        card.classList.remove("hidden");
    }

    // ================================================================
    //  AUDIT LOG LOOKUP
    // ================================================================

    const auditInput = document.getElementById("audit-id");
    const btnAudit = document.getElementById("btn-audit");

    auditInput.addEventListener("input", () => {
        btnAudit.disabled = !auditInput.value.trim();
    });

    // Allow Enter key to trigger lookup
    auditInput.addEventListener("keydown", e => {
        if (e.key === "Enter" && !btnAudit.disabled) btnAudit.click();
    });

    btnAudit.addEventListener("click", async () => {
        const requestId = auditInput.value.trim();
        if (!requestId) return;

        btnAudit.disabled = true;
        btnAudit.classList.add("loading");
        hideResult("audit-result");

        try {
            const resp = await fetch(`/api/logs/${encodeURIComponent(requestId)}`);
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            renderAuditResult(data);
        } catch (err) {
            renderError("audit-result", err.message);
        } finally {
            btnAudit.classList.remove("loading");
            btnAudit.disabled = !auditInput.value.trim();
        }
    });

    function renderAuditResult(data) {
        const card = document.getElementById("audit-result");

        // Detect whether this is a batch log (has verdict) or single log (has result)
        if (data.verdict) {
            renderAuditBatch(card, data);
        } else if (data.result) {
            renderAuditSingle(card, data);
        } else {
            card.innerHTML = `<div class="error-banner">Unrecognised log format.</div>`;
            card.classList.remove("hidden");
        }
    }

    function renderAuditBatch(card, data) {
        const v = data.verdict;
        const confClass = v.avg_confidence >= 0.85 ? "high" : v.avg_confidence >= 0.5 ? "mid" : "low";
        const verdictClass = v.inconclusive ? "inconclusive" : v.signature_matched ? "matched" : "unmatched";
        const verdictLabel = v.inconclusive ? "Inconclusive" : v.signature_matched ? "Signatures Match" : "Signatures Do Not Match";
        const verdictIcon = v.inconclusive ? "?" : v.signature_matched ? "✓" : "✗";

        let individualsHtml = "";
        if (data.individual_results && data.individual_results.length) {
            individualsHtml = `
            <div class="individual-results">
                <h3>Individual Comparisons</h3>
                ${data.individual_results.map((ir) => {
                    const statusClass = ir.error ? "error" : ir.signature_matched ? "match" : "no-match";
                    return `
                    <div class="individual-item-wrap" onclick="this.classList.toggle('expanded')">
                        <div class="individual-item">
                            <div class="ind-status ${statusClass}"></div>
                            <span class="ind-name">${escHtml(ir.reference_filename)}</span>
                            <span class="ind-confidence">${ir.error ? "—" : (ir.confidence_score * 100).toFixed(1) + "%"}</span>
                            <span class="ind-time">${ir.error ? "" : ir.elapsed_ms.toFixed(0) + " ms"}</span>
                            <span class="ind-expand">▼</span>
                        </div>
                        <div class="ind-reasoning md-content">${ir.error ? escHtml(ir.error) : mdToHtml(ir.reasoning)}</div>
                    </div>`;
                }).join("")}
            </div>`;
        }

        card.innerHTML = `
            <div class="audit-badge">Batch Result — ${escHtml(data.request_id)}</div>
            <div class="verdict-banner ${verdictClass}">
                <div class="verdict-icon">${verdictIcon}</div>
                <div class="verdict-text">
                    <h2>${verdictLabel}</h2>
                    <div class="verdict-meta">
                        <span>Match ratio: ${v.match_ratio}</span>
                        <span>Avg confidence: ${(v.avg_confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">
                    <span>Average Confidence</span>
                    <span>${(v.avg_confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill ${confClass}" style="width:${v.avg_confidence * 100}%"></div>
                </div>
            </div>
            <div class="reasoning-section">
                <h3>Summary</h3>
                <div class="md-content">${mdToHtml(v.reasoning)}</div>
            </div>
            <div class="metrics-row">
                ${metricHtml("Comparisons", data.individual_results ? data.individual_results.length : "—")}
                ${metricHtml("Total Time", `${data.elapsed_ms.toFixed(0)} ms`)}
                ${tokenMetric("Total Tokens", data.total_usage)}
                ${costMetric(data.total_cost_inr)}
                ${metricHtml("Method", v.decision_method)}
            </div>
            ${individualsHtml}
        `;
        card.classList.remove("hidden");
    }

    function renderAuditSingle(card, data) {
        const r = data.result;
        const confClass = r.confidence_score >= 0.85 ? "high" : r.confidence_score >= 0.5 ? "mid" : "low";
        const verdictClass = r.signature_matched ? "matched" : "unmatched";
        const verdictLabel = r.signature_matched ? "Signatures Match" : "Signatures Do Not Match";
        const verdictIcon = r.signature_matched ? "✓" : "✗";

        card.innerHTML = `
            <div class="audit-badge">Single Compare Result</div>
            <div class="verdict-banner ${verdictClass}">
                <div class="verdict-icon">${verdictIcon}</div>
                <div class="verdict-text">
                    <h2>${verdictLabel}</h2>
                    <div class="verdict-meta">
                        <span>${escHtml(data.image1 || "")} vs ${escHtml(data.image2 || "")}</span>
                    </div>
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">
                    <span>Confidence</span>
                    <span>${(r.confidence_score * 100).toFixed(1)}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill ${confClass}" style="width:${r.confidence_score * 100}%"></div>
                </div>
            </div>
            <div class="reasoning-section">
                <h3>Analysis</h3>
                <div class="md-content">${mdToHtml(r.reasoning)}</div>
            </div>
            <div class="metrics-row">
                ${data.elapsed_ms ? metricHtml("Total", `${data.elapsed_ms.toFixed(0)} ms`) : ""}
                ${tokenMetric("Tokens", data.usage)}
                ${costMetric(data.cost_inr)}
            </div>
        `;
        card.classList.remove("hidden");
    }

    // ================================================================
    //  UTILITIES
    // ================================================================

    function setupUploadZone(zoneId, inputId, previewId, removeId, onFile) {
        const zone = document.getElementById(zoneId);
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        const removeBtn = document.getElementById(removeId);

        zone.addEventListener("click", e => {
            if (e.target === removeBtn || removeBtn.contains(e.target)) return;
            input.click();
        });
        zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
        zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
        zone.addEventListener("drop", e => {
            e.preventDefault();
            zone.classList.remove("dragover");
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) setFile(file);
        });
        input.addEventListener("change", () => {
            if (input.files[0]) setFile(input.files[0]);
        });
        removeBtn.addEventListener("click", e => {
            e.stopPropagation();
            clearFile();
        });

        function setFile(file) {
            preview.src = URL.createObjectURL(file);
            zone.classList.add("has-file");
            onFile(file);
        }
        function clearFile() {
            if (preview.src) URL.revokeObjectURL(preview.src);
            preview.src = "";
            zone.classList.remove("has-file");
            input.value = "";
            onFile(null);
        }
    }

    function hideResult(id) {
        const el = document.getElementById(id);
        el.classList.add("hidden");
        el.innerHTML = "";
    }

    function showWaitingNote(id) {
        const card = document.getElementById(id);
        card.innerHTML = `<div class="waiting-note">
            <div class="waiting-spinner"></div>
            <div class="waiting-text">Analyzing signatures — avg wait time is ~20 seconds.<br>Hold tight, stay tuned!</div>
        </div>`;
        card.classList.remove("hidden");
    }

    function renderError(id, msg) {
        const card = document.getElementById(id);
        card.innerHTML = `<div class="error-banner">Error: ${escHtml(msg)}</div>`;
        card.classList.remove("hidden");
    }

    function metricHtml(label, value) {
        return `<div class="metric"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></div>`;
    }

    /** Format cost in INR for display — returns metric HTML or empty string if null/zero. */
    function costMetric(cost) {
        if (cost == null) return "";
        const formatted = cost < 0.01 ? `₹${cost.toFixed(4)}` : `₹${cost.toFixed(2)}`;
        return metricHtml("Est. Cost", formatted);
    }

    /** Render a Tokens metric with an ⓘ tooltip showing the input/output/cached/reasoning split. */
    function tokenMetric(label, usage) {
        if (!usage) return "";
        const total = (usage.total_tokens || 0).toLocaleString();
        const inp   = (usage.input_tokens || 0).toLocaleString();
        const out   = (usage.output_tokens || 0).toLocaleString();
        const cached = (usage.cached_tokens || 0).toLocaleString();
        const reason = (usage.reasoning_tokens || 0).toLocaleString();
        return `<div class="metric"><div class="metric-value">${total}<span class="info-btn" tabindex="0">&#9432;<span class="info-tooltip token-tip">Input: ${inp}<br>Output: ${out}<br>Cached: ${cached}<br>Reasoning: ${reason}</span></span></div><div class="metric-label">${label}</div></div>`;
    }

    function escHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Convert lightweight Markdown (from LLM output) into safe HTML.
     * Supports: headings (# – ###), **bold**, *italic*, `code`,
     * unordered lists (- / *), ordered lists (1.), line breaks.
     */
    function mdToHtml(raw) {
        if (!raw) return "";
        // 1. Escape HTML entities first for safety
        let text = escHtml(raw);

        // 2. Headings (must be at start of a line)
        text = text.replace(/^### (.+)$/gm, "<h5>$1</h5>");
        text = text.replace(/^## (.+)$/gm, "<h4>$1</h4>");
        text = text.replace(/^# (.+)$/gm, "<h3>$1</h3>");

        // 3. Bold & italic (handle bold first to avoid conflicts)
        text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        text = text.replace(/\*(.+?)\*/g, "<em>$1</em>");

        // 4. Inline code
        text = text.replace(/`(.+?)`/g, "<code>$1</code>");

        // 5. Unordered lists (lines starting with - or * followed by space)
        text = text.replace(/^(?:[-*]) (.+)$/gm, "<li>$1</li>");
        text = text.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

        // 6. Ordered lists (lines starting with number.)
        text = text.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
        // wrap consecutive <li> that aren't already inside <ul>
        text = text.replace(/(?<!<\/ul>)((?:<li>.*<\/li>\n?)+)/g, (m) => {
            return m.includes("<ul>") ? m : "<ol>" + m + "</ol>";
        });

        // 7. Paragraphs — convert double newlines to paragraph breaks
        text = text.replace(/\n{2,}/g, "</p><p>");

        // 8. Single newlines to <br> (but not inside block elements)
        text = text.replace(/\n/g, "<br>");

        return "<p>" + text + "</p>";
    }
})();
