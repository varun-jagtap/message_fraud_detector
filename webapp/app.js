const DEFAULT_CONFIG = {
  retrieval_weight: 0.55,
  rule_weight: 0.45,
  fraud_threshold: 0.48,
  suspicious_threshold: 0.28,
  top_k: 5,
  retrieval_min_score: 0.08,
  rule_cap: 1,
  lower_retrieve: true,
  strip_accents: true,
  max_features: 5000,
  ngram_min: 1,
  ngram_max: 2,
  sublinear_tf: true,
  model_version: "1.0"
};

let batchData = [];
let sampleData = [];

async function api(path, payload) {
  const options = { headers: { "Content-Type": "application/json" } };
  if (payload !== undefined) {
    options.method = "POST";
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function labelBadge(label) {
  const clean = String(label).toLowerCase();
  return `<span class="badge ${clean}">${escapeHtml(clean)}</span>`;
}

function setNotice(element, message, error = false) {
  element.textContent = message;
  element.className = `notice visible${error ? " error" : ""}`;
}

async function refreshModels() {
  const data = await api("/api/models");
  const selectors = ["score-model", "batch-model", "inspect-model"];
  for (const id of selectors) {
    const select = document.getElementById(id);
    const previous = select.value;
    select.innerHTML = data.models.length
      ? data.models.map(name => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")
      : '<option value="web_data/detector.json">web_data/detector.json</option>';
    if (data.models.includes(previous)) select.value = previous;
  }
  if (!data.models.length) {
    document.getElementById("score-result").innerHTML =
      '<p class="empty-title">No model available</p><p class="muted">Generate samples and train a model first.</p>';
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === name);
  });
}

function renderConfigFields() {
  const host = document.getElementById("config-fields");
  host.innerHTML = Object.entries(DEFAULT_CONFIG).map(([key, value]) => `
    <div>
      <label for="config-${key}">${escapeHtml(key)}</label>
      <input id="config-${key}" data-config-key="${escapeHtml(key)}"
        type="${typeof value === "boolean" ? "checkbox" : typeof value === "number" ? "number" : "text"}"
        ${typeof value === "boolean" ? (value ? "checked" : "") : `value="${escapeHtml(value)}"`}
        ${typeof value === "number" ? 'step="0.01"' : ""}>
    </div>
  `).join("");
  document.getElementById("config-json").value =
    JSON.stringify(DEFAULT_CONFIG, null, 2);
}

function collectConfig() {
  const config = {};
  for (const input of document.querySelectorAll("[data-config-key]")) {
    const key = input.dataset.configKey;
    if (input.type === "checkbox") config[key] = input.checked;
    else if (input.type === "number") config[key] = Number(input.value);
    else config[key] = input.value;
  }
  return config;
}

async function scoreMessage() {
  const button = document.getElementById("score-button");
  const text = document.getElementById("message-text").value;
  if (!text.trim()) {
    document.getElementById("score-result").innerHTML =
      '<p class="empty-title">Message required</p><p class="muted">Paste a message before checking it.</p>';
    return;
  }
  button.disabled = true;
  button.textContent = "Checking...";
  try {
    const result = await api("/api/score", {
      model: document.getElementById("score-model").value,
      message_id: document.getElementById("message-id").value,
      text
    });
    document.getElementById("score-result").innerHTML = `
      <p class="eyebrow">Analysis result</p>
      <span class="decision ${escapeHtml(result.label)}">${escapeHtml(result.label)}</span>
      <div class="score-row">
        <strong>${(result.score * 100).toFixed(1)}%</strong>
        <span>${escapeHtml(result.risk_level)} risk</span>
      </div>
      <p><strong>Key reasons</strong></p>
      <ul class="reason-list">
        ${(result.reasons || []).slice(0, 3).map(reason => `<li>${escapeHtml(reason)}</li>`).join("")}
      </ul>
    `;
  } catch (error) {
    document.getElementById("score-result").innerHTML =
      `<p class="empty-title">Unable to check</p><p class="muted">${escapeHtml(error.message)}</p>`;
  } finally {
    button.disabled = false;
    button.textContent = "Check for fraud";
  }
}

async function runBatch() {
  const button = document.getElementById("batch-button");
  const input = document.getElementById("batch-input").value;
  button.disabled = true;
  try {
    const data = await api("/api/batch", {
      model: document.getElementById("batch-model").value,
      text: input
    });
    batchData = data.results;
    const rows = data.results.map((result, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${labelBadge(result.label)}</td>
        <td><strong>${(result.score * 100).toFixed(1)}%</strong></td>
        <td>${escapeHtml(result.risk_level)}</td>
        <td>${escapeHtml((result.reasons || []).slice(0, 2).join("; "))}</td>
      </tr>
    `).join("");
    document.getElementById("batch-result").innerHTML = `
      <table>
        <thead><tr><th>#</th><th>Decision</th><th>Score</th><th>Risk</th><th>Reasons</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
    document.getElementById("download-batch").disabled = false;
  } catch (error) {
    document.getElementById("batch-result").innerHTML =
      `<p class="muted">${escapeHtml(error.message)}</p>`;
  } finally {
    button.disabled = false;
  }
}

async function trainModel() {
  const button = document.getElementById("train-button");
  const notice = document.getElementById("train-result");
  const input = document.getElementById("train-data").value;
  button.disabled = true;
  try {
    const data = await api("/api/train", {
      model: document.getElementById("train-name").value,
      config: collectConfig(),
      text: input,
      format: document.getElementById("train-format").value
    });
    setNotice(notice, `Saved ${data.path} with ${data.example_count} examples.`);
    await refreshModels();
  } catch (error) {
    setNotice(notice, error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function generateSamples() {
  try {
    const data = await api("/api/generate", {
      count: Number(document.getElementById("generate-count").value),
      seed: Number(document.getElementById("generate-seed").value)
    });
    sampleData = data.records;
    document.getElementById("generate-preview").textContent =
      JSON.stringify(data.records.slice(0, 12), null, 2);
    document.getElementById("download-samples").disabled = false;
  } catch (error) {
    document.getElementById("generate-preview").textContent = error.message;
  }
}

async function inspectModel() {
  try {
    const data = await api("/api/inspect", {
      model: document.getElementById("inspect-model").value
    });
    document.getElementById("inspect-result").textContent =
      JSON.stringify(data, null, 2);
  } catch (error) {
    document.getElementById("inspect-result").textContent = error.message;
  }
}

function downloadJson(filename, value) {
  const blob = new Blob([JSON.stringify(value, null, 2)], {
    type: "application/json"
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function validateConfig() {
  try {
    const config = JSON.parse(document.getElementById("config-json").value);
    const data = await api("/api/config/validate", { config });
    setNotice(document.getElementById("config-result"), "Configuration is valid.");
  } catch (error) {
    setNotice(document.getElementById("config-result"), error.message, true);
  }
}

async function createConfig() {
  try {
    const config = JSON.parse(document.getElementById("config-json").value);
    const data = await api("/api/config/create", {
      name: "detector",
      config
    });
    setNotice(document.getElementById("config-result"), `Saved ${data.path}.`);
  } catch (error) {
    setNotice(document.getElementById("config-result"), error.message, true);
  }
}

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

document.getElementById("score-button").addEventListener("click", scoreMessage);
document.getElementById("batch-button").addEventListener("click", runBatch);
document.getElementById("train-button").addEventListener("click", trainModel);
document.getElementById("generate-button").addEventListener("click", generateSamples);
document.getElementById("inspect-button").addEventListener("click", inspectModel);
document.getElementById("refresh-models").addEventListener("click", refreshModels);
document.getElementById("validate-config").addEventListener("click", validateConfig);
document.getElementById("create-config").addEventListener("click", createConfig);
document.getElementById("download-batch").addEventListener("click", () => {
  downloadJson("batch-results.json", batchData);
});
document.getElementById("download-samples").addEventListener("click", () => {
  const jsonl = sampleData.map(record => JSON.stringify(record)).join("\n") + "\n";
  const blob = new Blob([jsonl], { type: "application/x-ndjson" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "samples.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
});

renderConfigFields();
refreshModels().catch(() => {
  const status = document.getElementById("server-status");
  status.className = "status offline";
  status.innerHTML = "<span></span>Offline";
});

api("/api/health").then(() => {
  const status = document.getElementById("server-status");
  status.className = "status online";
  status.innerHTML = "<span></span>Online";
}).catch(() => {});
