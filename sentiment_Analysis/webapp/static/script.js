const LABELS = ["POS","NEG","NEU","MIX","UNSURE"];
const LABEL_NAMES = {POS:"Positive", NEG:"Negative", NEU:"Neutral", MIX:"Mixed", UNSURE:"Unsure"};
const CHIP_VAR = {POS:"--chip-pos", NEG:"--chip-neg", NEU:"--chip-neu", MIX:"--chip-mix", UNSURE:"--chip-unsure"};
const KEYS = {"1":"POS","2":"NEG","3":"NEU","4":"MIX","5":"UNSURE"};

let disagreements = [];
let labeled = [];
let reviews = {};     // id -> {manual_label}
let corrections = {}; // id -> {corrected_label, original_label}
let dqIndex = 0;

function esc(s){ const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }
function chip(label, extra){
  return `<span class="chip" style="background:var(${CHIP_VAR[label]})">${label}${extra ? " " + esc(extra) : ""}</span>`;
}

async function loadData(){
  const r = await fetch("/api/data");
  const d = await r.json();
  disagreements = d.disagreements;
  labeled = d.labeled;
  reviews = d.reviews;
  corrections = d.corrections;
}

function renderStats(){
  const total = disagreements.length;
  const done = Object.keys(reviews).length;
  const corrN = Object.keys(corrections).length;
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="n">${labeled.length.toLocaleString()}</div><div class="l">Labeled items</div></div>
    <div class="stat"><div class="n">${done}/${total}</div><div class="l">Disagreements resolved</div></div>
    <div class="stat"><div class="n">${corrN}</div><div class="l">Corrections made</div></div>
  `;
}

/* ---------------- Disagreements tab ---------------- */

function nextUnresolvedIndex(from){
  for (let i = from; i < disagreements.length; i++){
    if (!reviews[disagreements[i].id]) return i;
  }
  for (let i = 0; i < from; i++){
    if (!reviews[disagreements[i].id]) return i;
  }
  return -1;
}

function renderDisagreeTab(){
  const panel = document.getElementById("panel-disagree");
  const total = disagreements.length;
  const done = Object.keys(reviews).length;

  let html = `
    <div class="progress-row">
      <div class="progress-track"><div class="progress-fill" style="width:${total ? (done/total*100) : 0}%"></div></div>
      <div class="progress-label">${done} / ${total} resolved</div>
    </div>
  `;

  if (done >= total){
    html += `<div class="done-banner"><strong>All disagreements resolved</strong>Every item below can still be re-labeled.</div>`;
  } else {
    if (!disagreements[dqIndex] || reviews[disagreements[dqIndex].id]) {
      const nxt = nextUnresolvedIndex(0);
      if (nxt >= 0) dqIndex = nxt;
    }
    const item = disagreements[dqIndex];
    if (item){
      const votes = item.votes || {};
      const voteChips = Object.entries(votes).sort((a,b) => b[1]-a[1])
        .map(([lab, n]) => chip(lab, `×${n}`)).join(" ");
      const current = reviews[item.id]?.manual_label;
      html += `
        <div class="card">
          <div class="card-text" dir="auto">${esc(item.text)}</div>
          <div class="votes-row">${voteChips}</div>
          <div class="label-btns">
            ${LABELS.map((lab, i) => `
              <button class="label-btn ${lab === current ? "selected" : ""}" data-label="${lab}">
                <kbd>${i+1}</kbd> ${LABEL_NAMES[lab]}
              </button>`).join("")}
          </div>
          <div class="nav-row">
            <button class="ghost-btn" id="dq-prev">&larr; Previous</button>
            <span style="font-size:0.78rem;color:var(--muted);align-self:center;">${item.sample_group}</span>
            <button class="ghost-btn" id="dq-skip">Skip &rarr;</button>
          </div>
        </div>
      `;
    }
  }

  const reviewedItems = disagreements.filter(d => reviews[d.id]);
  if (reviewedItems.length){
    html += `<div class="reviewed-list"><h3>Reviewed</h3>`;
    for (const it of reviewedItems){
      const lab = reviews[it.id].manual_label;
      html += `<div class="reviewed-row">
        <span class="lab" style="color:var(${CHIP_VAR[lab]})">${lab}</span>
        <span class="txt" dir="auto">${esc(it.text)}</span>
        <button data-goto="${it.id}">edit</button>
      </div>`;
    }
    html += `</div>`;
  }

  panel.innerHTML = html;

  panel.querySelectorAll(".label-btn").forEach(btn => {
    btn.addEventListener("click", () => setDisagreementLabel(disagreements[dqIndex].id, btn.dataset.label));
  });
  const prevBtn = document.getElementById("dq-prev");
  if (prevBtn) prevBtn.addEventListener("click", () => { dqIndex = (dqIndex - 1 + total) % total; renderDisagreeTab(); });
  const skipBtn = document.getElementById("dq-skip");
  if (skipBtn) skipBtn.addEventListener("click", () => { dqIndex = (dqIndex + 1) % total; renderDisagreeTab(); });
  panel.querySelectorAll("[data-goto]").forEach(b => {
    b.addEventListener("click", () => {
      dqIndex = disagreements.findIndex(d => d.id === b.dataset.goto);
      renderDisagreeTab();
    });
  });
}

async function setDisagreementLabel(id, label){
  reviews[id] = { manual_label: label };
  renderDisagreeTab();
  renderStats();
  try {
    await fetch("/api/review", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id, label}),
    });
  } catch (e) { /* kept in memory for this session even if the save failed */ }
  setTimeout(() => {
    const nxt = nextUnresolvedIndex(dqIndex + 1);
    if (nxt >= 0) { dqIndex = nxt; renderDisagreeTab(); }
  }, 250);
}

document.addEventListener("keydown", (e) => {
  if (document.getElementById("panel-disagree").hidden) return;
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  const item = disagreements[dqIndex];
  if (!item) return;
  if (KEYS[e.key]) setDisagreementLabel(item.id, KEYS[e.key]);
  else if (e.key === "ArrowRight") { dqIndex = (dqIndex + 1) % disagreements.length; renderDisagreeTab(); }
  else if (e.key === "ArrowLeft") { dqIndex = (dqIndex - 1 + disagreements.length) % disagreements.length; renderDisagreeTab(); }
});

document.addEventListener("keydown", (e) => {
  if (document.getElementById("panel-browse").hidden || !unsureQueueMode) return;
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  const items = unsureItems();
  const item = items[unsureQueueIndex];
  if (!item || corrections[item.id]) return;
  const UQ_KEYS = {"1":"POS","2":"NEG","3":"NEU","4":"MIX"};
  if (UQ_KEYS[e.key]) {
    applyCorrection(item.id, UQ_KEYS[e.key]).then(() => {
      const nxt = unsureNextUnresolvedIndex(items, unsureQueueIndex + 1);
      if (nxt >= 0) unsureQueueIndex = nxt;
      renderUnsureQueue();
    });
  } else if (e.key === "ArrowRight") { unsureQueueIndex = (unsureQueueIndex + 1) % items.length; renderUnsureQueue(); }
});

document.addEventListener("keydown", (e) => {
  if (document.getElementById("panel-humaneval").hidden) return;
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  const item = heItems[heIndex];
  if (!item || heRatings[item.id]) return;
  if (KEYS[e.key]) submitHumanEvalRating(item.id, KEYS[e.key]);
  else if (e.key === "ArrowRight") { heIndex = (heIndex + 1) % heItems.length; heLastReveal = null; renderHumanEvalTab(); }
});

/* ---------------- Browse tab ---------------- */

let browsePage = 0;
const PAGE_SIZE = 40;

function filteredLabeled(){
  const labelSel = document.getElementById("f-label")?.value || "all";
  const groupSel = document.getElementById("f-group")?.value || "all";
  const sortSel = document.getElementById("f-sort")?.value || "default";
  let rows = labeled.filter(r =>
    (labelSel === "all" || r.label === labelSel) &&
    (groupSel === "all" || r.sample_group === groupSel)
  );
  if (sortSel === "shakiest"){
    rows = rows.filter(r => r.votes).sort((a, b) => {
      const shareA = Math.max(...Object.values(a.votes)) / 5;
      const shareB = Math.max(...Object.values(b.votes)) / 5;
      return shareA - shareB;
    });
  }
  return rows;
}

function renderBrowseTab(reset){
  if (reset) browsePage = 0;
  const panel = document.getElementById("panel-browse");
  const groups = [...new Set(labeled.map(r => r.sample_group))].sort();
  const unsureLeft = labeled.filter(r => r.label === "UNSURE" && !corrections[r.id]).length;

  if (reset || !document.getElementById("f-label")){
    panel.innerHTML = `
      <div class="filters">
        <select id="f-label">
          <option value="all">All labels</option>
          ${LABELS.map(l => `<option value="${l}">${LABEL_NAMES[l]} (${l})</option>`).join("")}
        </select>
        <select id="f-group">
          <option value="all">All sources</option>
          ${groups.map(g => `<option value="${g}">${g}</option>`).join("")}
        </select>
        <select id="f-sort">
          <option value="default">Order: as labeled</option>
          <option value="shakiest">Order: shakiest vote first</option>
        </select>
        <button class="ghost-btn" id="resolve-unsure-btn">Resolve UNSURE one by one (${unsureLeft} left)</button>
      </div>
      <div id="unsure-queue"></div>
      <div id="browse-list"></div>
      <button class="load-more" id="load-more">Load more</button>
      <div class="empty-note" id="browse-empty" hidden>No items match these filters.</div>
    `;
    document.getElementById("f-label").addEventListener("change", () => renderBrowseList(true));
    document.getElementById("f-group").addEventListener("change", () => renderBrowseList(true));
    document.getElementById("f-sort").addEventListener("change", () => renderBrowseList(true));
    document.getElementById("load-more").addEventListener("click", () => { browsePage++; renderBrowseList(false); });
    document.getElementById("resolve-unsure-btn").addEventListener("click", () => {
      unsureQueueMode = !unsureQueueMode;
      renderUnsureQueue();
    });
  }
  renderUnsureQueue();
  renderBrowseList(true);
}

/* ---- Resolve UNSURE one by one: a dedicated queue, same shape as the Disagreements tab, driven
   from labeled.json's UNSURE items instead of disagreements.json. Writes to the same `corrections`
   store the "Flag/correct" action already uses, so there's exactly one place a label change lives,
   not two competing mechanisms. ---- */
let unsureQueueMode = false;
let unsureQueueIndex = 0;

function unsureItems(){
  return labeled.filter(r => r.label === "UNSURE");
}

function unsureNextUnresolvedIndex(items, from){
  for (let i = from; i < items.length; i++){
    if (!corrections[items[i].id]) return i;
  }
  for (let i = 0; i < from; i++){
    if (!corrections[items[i].id]) return i;
  }
  return -1;
}

function renderUnsureQueue(){
  const box = document.getElementById("unsure-queue");
  const btn = document.getElementById("resolve-unsure-btn");
  if (!box) return;
  if (!unsureQueueMode){ box.innerHTML = ""; if (btn) btn.textContent = `Resolve UNSURE one by one (${labeled.filter(r => r.label === "UNSURE" && !corrections[r.id]).length} left)`; return; }

  const items = unsureItems();
  const total = items.length;
  const done = items.filter(r => corrections[r.id]).length;
  if (btn) btn.textContent = "Back to list";

  let html = `
    <div class="progress-row">
      <div class="progress-track"><div class="progress-fill" style="width:${total ? (done/total*100) : 0}%"></div></div>
      <div class="progress-label">${done} / ${total} resolved</div>
    </div>
  `;

  if (done >= total){
    html += `<div class="done-banner"><strong>Every UNSURE item has a new label</strong>Click "Back to list" to browse, or any row's "Edit correction" to change one again.</div>`;
  } else {
    if (!items[unsureQueueIndex] || corrections[items[unsureQueueIndex].id]){
      const nxt = unsureNextUnresolvedIndex(items, 0);
      if (nxt >= 0) unsureQueueIndex = nxt;
    }
    const item = items[unsureQueueIndex];
    if (item){
      html += `
        <div class="card">
          <div class="card-text" dir="auto">${esc(item.text)}</div>
          <div class="label-btns">
            ${LABELS.filter(l => l !== "UNSURE").map(lab => `
              <button class="label-btn" data-unsurelabel="${lab}">${LABEL_NAMES[lab]}</button>`).join("")}
          </div>
          <div class="nav-row">
            <span></span>
            <span style="font-size:0.78rem;color:var(--muted);align-self:center;">${item.sample_group}</span>
            <button class="ghost-btn" id="uq-skip">Skip &rarr;</button>
          </div>
        </div>
      `;
    }
  }

  box.innerHTML = html;
  box.querySelectorAll("[data-unsurelabel]").forEach(b => {
    b.addEventListener("click", async () => {
      const item = items[unsureQueueIndex];
      await applyCorrection(item.id, b.dataset.unsurelabel);
      const nxt = unsureNextUnresolvedIndex(items, unsureQueueIndex + 1);
      if (nxt >= 0) unsureQueueIndex = nxt;
      renderUnsureQueue();
    });
  });
  const skipBtn = document.getElementById("uq-skip");
  if (skipBtn) skipBtn.addEventListener("click", () => { unsureQueueIndex = (unsureQueueIndex + 1) % total; renderUnsureQueue(); });
}

function renderBrowseList(reset){
  if (reset) browsePage = 0;
  const rows = filteredLabeled();
  const shown = rows.slice(0, (browsePage + 1) * PAGE_SIZE);
  const list = document.getElementById("browse-list");
  document.getElementById("browse-empty").hidden = rows.length > 0;

  list.innerHTML = shown.map(r => {
    const corr = corrections[r.id];
    const votesNote = r.votes ? Object.entries(r.votes).map(([l,n]) => `${l}×${n}`).join(" ") : "single annotator";
    return `
      <div class="browse-row" data-id="${r.id}">
        <div class="top">
          ${chip(r.label)}
          ${corr ? `<span class="corrected-note">&rarr; corrected to ${corr.corrected_label}</span>` : ""}
        </div>
        <div class="txt" dir="auto">${esc(r.text)}</div>
        <div class="meta-row">
          <span class="tag">${r.sample_group}</span>
          <span class="tag">${votesNote}</span>
          <button class="correct-btn" data-correct="${r.id}">${corr ? "Edit correction" : "Flag / correct"}</button>
        </div>
        <div class="correction-panel" id="corr-${r.id}" hidden></div>
      </div>
    `;
  }).join("");

  document.getElementById("load-more").hidden = shown.length >= rows.length;

  list.querySelectorAll("[data-correct]").forEach(btn => {
    btn.addEventListener("click", () => toggleCorrectionPanel(btn.dataset.correct));
  });
}

function toggleCorrectionPanel(id){
  const panel = document.getElementById("corr-" + id);
  if (!panel.hidden){ panel.hidden = true; return; }
  const row = labeled.find(r => r.id === id);
  panel.hidden = false;
  panel.innerHTML = LABELS.filter(l => l !== row.label).map(l =>
    `<button class="label-btn" data-apply="${id}" data-newlabel="${l}">${LABEL_NAMES[l]}</button>`
  ).join("") + `<button class="ghost-btn" data-clear="${id}">Clear correction</button>`;
  panel.querySelectorAll("[data-apply]").forEach(b => {
    b.addEventListener("click", () => applyCorrection(b.dataset.apply, b.dataset.newlabel));
  });
  const clearBtn = panel.querySelector("[data-clear]");
  if (clearBtn) clearBtn.addEventListener("click", () => clearCorrection(id));
}

async function applyCorrection(id, newLabel){
  const row = labeled.find(r => r.id === id);
  corrections[id] = { corrected_label: newLabel, original_label: row.label };
  renderBrowseList(false);
  renderStats();
  const btn = document.getElementById("resolve-unsure-btn");
  if (btn && !unsureQueueMode) btn.textContent = `Resolve UNSURE one by one (${labeled.filter(r => r.label === "UNSURE" && !corrections[r.id]).length} left)`;
  try {
    await fetch("/api/correction", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id, label: newLabel, original_label: row.label}),
    });
  } catch (e) { /* kept locally for this session */ }
}

async function clearCorrection(id){
  delete corrections[id];
  renderBrowseList(false);
  renderStats();
  const btn = document.getElementById("resolve-unsure-btn");
  if (btn && !unsureQueueMode) btn.textContent = `Resolve UNSURE one by one (${labeled.filter(r => r.label === "UNSURE" && !corrections[r.id]).length} left)`;
  try {
    await fetch("/api/correction", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id, label: null}),
    });
  } catch (e) { /* local state already cleared */ }
}

/* ---------------- Human eval tab ---------------- */

let heItems = [];
let heRatings = {};  // id -> {human_label}  (no agent label here -- that only arrives post-submit)
let heIndex = 0;
let heLastReveal = null; // {id, agent_label, human_label} for the item just rated, shown once

async function loadHumanEval(){
  const r = await fetch("/api/human_eval");
  const d = await r.json();
  heItems = d.items;
  heRatings = d.ratings;
}

function heNextUnratedIndex(from){
  for (let i = from; i < heItems.length; i++){
    if (!heRatings[heItems[i].id]) return i;
  }
  for (let i = 0; i < from; i++){
    if (!heRatings[heItems[i].id]) return i;
  }
  return -1;
}

function renderHumanEvalTab(){
  const panel = document.getElementById("panel-humaneval");
  const total = heItems.length;
  const done = Object.keys(heRatings).length;

  let html = `
    <p style="font-size:0.82rem;color:var(--muted);margin-top:0;">
      Blind check on the <em>agreed-upon</em> labels (not the 29 disagreements, which already get
      separate review) -- the agents' label is hidden until after you submit your own.
    </p>
    <div class="progress-row">
      <div class="progress-track"><div class="progress-fill" style="width:${total ? (done/total*100) : 0}%"></div></div>
      <div class="progress-label">${done} / ${total} rated</div>
    </div>
  `;

  if (heLastReveal){
    const agree = heLastReveal.agent_label === heLastReveal.human_label;
    html += `<div class="reveal-banner ${agree ? "agree" : "disagree"}">
      ${agree ? "✓ Agent agreed" : "✗ Agent said"}: <strong>${heLastReveal.agent_label}</strong>
      ${agree ? "" : `(you said ${heLastReveal.human_label})`}
    </div>`;
  }

  if (done >= total && total > 0){
    html += `<div class="done-banner"><strong>All ${total} items rated</strong>See the results below.</div>`;
  } else {
    if (!heItems[heIndex] || heRatings[heItems[heIndex].id]) {
      const nxt = heNextUnratedIndex(0);
      if (nxt >= 0) heIndex = nxt;
    }
    const item = heItems[heIndex];
    if (item){
      html += `
        <div class="card">
          <div class="card-text" dir="auto">${esc(item.text)}</div>
          <div class="label-btns">
            ${LABELS.map((lab, i) => `
              <button class="label-btn" data-helabel="${lab}">
                <kbd>${i+1}</kbd> ${LABEL_NAMES[lab]}
              </button>`).join("")}
          </div>
          <div class="nav-row">
            <span></span>
            <span style="font-size:0.78rem;color:var(--muted);align-self:center;">${item.sample_group}</span>
            <button class="ghost-btn" id="he-skip">Skip &rarr;</button>
          </div>
        </div>
      `;
    }
  }

  html += `<div id="he-results"></div>`;
  panel.innerHTML = html;

  panel.querySelectorAll("[data-helabel]").forEach(btn => {
    btn.addEventListener("click", () => submitHumanEvalRating(heItems[heIndex].id, btn.dataset.helabel));
  });
  const skipBtn = document.getElementById("he-skip");
  if (skipBtn) skipBtn.addEventListener("click", () => { heIndex = (heIndex + 1) % total; heLastReveal = null; renderHumanEvalTab(); });

  if (done > 0) loadHumanEvalResults();
}

async function submitHumanEvalRating(id, label){
  heRatings[id] = { human_label: label };
  try {
    const r = await fetch("/api/human_eval/rate", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({id, label}),
    });
    const d = await r.json();
    heLastReveal = { id, agent_label: d.agent_label, human_label: label };
  } catch (e) { heLastReveal = null; }
  const nxt = heNextUnratedIndex(heIndex + 1);
  if (nxt >= 0) heIndex = nxt;
  renderHumanEvalTab();
}

async function loadHumanEvalResults(){
  const r = await fetch("/api/human_eval/results");
  const d = await r.json();
  const box = document.getElementById("he-results");
  if (!box || !d.n) return;

  let html = `
    <h3 style="font-size:0.82rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:22px 0 8px;">
      Results so far (n=${d.n}/${d.total})
    </h3>
    <div class="results-grid">
      <div class="stat"><div class="n">${(d.accuracy*100).toFixed(1)}%</div><div class="l">Agreement rate</div></div>
      <div class="stat"><div class="n">${d.kappa.toFixed(3)}</div><div class="l">Cohen's kappa</div></div>
    </div>
  `;

  html += `<div style="margin-bottom:18px;">`;
  for (const lab of LABELS){
    const pc = d.per_class[lab];
    const rate = pc.agent_agreement_rate === null ? "n/a" : (pc.agent_agreement_rate*100).toFixed(0) + "%";
    html += `<div class="per-class-row">
      <span>${chip(lab)} you rated this ${pc.n} time(s)</span>
      <span>agent agreed ${rate}</span>
    </div>`;
  }
  html += `</div>`;

  html += `<table class="cm"><tr><th>human \\ agent</th>${LABELS.map(l => `<th>${l}</th>`).join("")}</tr>`;
  d.confusion_matrix.forEach((row, i) => {
    html += `<tr><th>${LABELS[i]}</th>${row.map((v, j) => `<td class="${i===j ? "diag" : ""}">${v}</td>`).join("")}</tr>`;
  });
  html += `</table>`;

  box.innerHTML = html;
}

/* ---------------- Tabs + boot ---------------- */

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-disagree").hidden = btn.dataset.tab !== "disagree";
    document.getElementById("panel-browse").hidden = btn.dataset.tab !== "browse";
    document.getElementById("panel-humaneval").hidden = btn.dataset.tab !== "humaneval";
    if (btn.dataset.tab === "browse") renderBrowseTab(false);
    if (btn.dataset.tab === "humaneval") { heLastReveal = null; loadHumanEval().then(renderHumanEvalTab); }
  });
});

async function boot(){
  await loadData();
  const firstUnresolved = nextUnresolvedIndex(0);
  dqIndex = firstUnresolved >= 0 ? firstUnresolved : 0;
  renderStats();
  renderDisagreeTab();
}

boot();
