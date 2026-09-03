// Cliente mínimo da API. Só usa fetch relativo (/api/ask); nenhuma base de conhecimento embutida.
(function () {
  const form = document.getElementById("ask-form");
  const questionEl = document.getElementById("question");
  const submitEl = document.getElementById("submit");
  const statusEl = document.getElementById("status");
  const resultEl = document.getElementById("result");
  const errorEl = document.getElementById("error");

  function text(id, value) { document.getElementById(id).textContent = value == null ? "" : String(value); }

  function locator(source) {
    const parts = [];
    if (source.page != null) parts.push("p. " + source.page);
    if (source.row != null) parts.push("linha " + source.row);
    if (source.section) parts.push(source.section);
    return parts.join(" · ");
  }

  function renderSources(sources) {
    const list = document.getElementById("sources");
    list.replaceChildren();
    text("sources-count", sources.length);
    for (const source of sources) {
      const li = document.createElement("li");
      const head = document.createElement("div");
      head.className = "source-head";
      head.textContent = source.document;
      const extra = document.createElement("span");
      extra.className = "muted";
      const bits = [locator(source)];
      if (typeof source.score === "number") bits.push("score " + source.score.toFixed(2));
      if (source.inferred) bits.push("inferida");
      extra.textContent = " — " + bits.filter(Boolean).join(" · ");
      head.appendChild(extra);
      li.appendChild(head);
      if (source.excerpt) {
        const quote = document.createElement("blockquote");
        quote.textContent = source.excerpt;
        li.appendChild(quote);
      }
      list.appendChild(li);
    }
    document.getElementById("sources-wrap").open = sources.length > 0;
  }

  function renderAnswer(data) {
    errorEl.hidden = true;
    resultEl.hidden = false;
    const refused = String(data.status || "").startsWith("refused");
    const badgeStatus = document.getElementById("badge-status");
    badgeStatus.textContent = refused ? "recusa (" + (data.refusal_reason || data.status) + ")" : "respondida";
    badgeStatus.className = "badge " + (refused ? "refused" : "answered");
    const badgeConf = document.getElementById("badge-confidence");
    badgeConf.textContent = "confiança: " + data.confidence;
    badgeConf.className = "badge " + data.confidence;
    text("badge-mode", "modo: " + data.mode);
    text("answer", data.answer);
    renderSources(Array.isArray(data.sources) ? data.sources : []);
    const timings = data.timings_ms || {};
    text("timings", Object.keys(timings).map((k) => k + " " + Number(timings[k]).toFixed(1) + " ms").join(" · "));
    text("request-id", data.request_id || "");
  }

  function renderError(status, data, requestId) {
    resultEl.hidden = true;
    errorEl.hidden = false;
    let detail = "Erro " + status;
    if (data && typeof data.detail === "string") detail = data.detail;
    else if (data && Array.isArray(data.detail)) detail = data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    const code = data && data.error_code ? " [" + data.error_code + "]" : "";
    const rid = requestId ? " (request_id " + requestId + ")" : "";
    errorEl.textContent = detail + code + rid;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionEl.value.trim();
    if (question.length < 2) return;
    submitEl.disabled = true;
    statusEl.textContent = "Consultando…";
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const requestId = response.headers.get("X-Request-ID");
      let data = null;
      try { data = await response.json(); } catch (_) { data = null; }
      if (!response.ok) renderError(response.status, data, requestId);
      else renderAnswer(data);
    } catch (err) {
      renderError("de rede", { detail: String(err) }, null);
    } finally {
      submitEl.disabled = false;
      statusEl.textContent = "";
    }
  });
})();
