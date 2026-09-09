const textEl = document.getElementById("text");
const topkEl = document.getElementById("topk");
const goEl = document.getElementById("predictBtn");
const noteEl = document.getElementById("note");
const resultsEl = document.getElementById("results");
const emptyState = document.getElementById("emptyState");

function probColor(frac) {
  // same low/high convention as the embedding-eval app's scoreColor:
  // low -> red, high -> green, gold in between.
  const r = frac < 0.5 ? 200 : Math.round(200 - (frac - 0.5) * 2 * 176);
  const g = frac < 0.5 ? Math.round(16 + frac * 2 * 86) : 102 + Math.round((frac - 0.5) * 2 * 18);
  const b = frac < 0.5 ? 46 : Math.round(46 - (frac - 0.5) * 2 * 22);
  return `rgb(${r}, ${g}, ${b})`;
}

async function predict() {
  const text = textEl.value.trim();
  noteEl.innerHTML = "";
  resultsEl.innerHTML = "";
  emptyState.style.display = "none";

  if (!text) {
    noteEl.innerHTML = '<span class="err">Type something first.</span>';
    return;
  }

  goEl.disabled = true;
  goEl.textContent = "...";
  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: parseInt(topkEl.value, 10) || 20 }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      noteEl.innerHTML = '<span class="err">' + (data.error || "request failed") + "</span>";
      return;
    }

    const notes = [];
    if (data.truncated) {
      notes.push("input truncated to the model's max input length");
    }
    if (data.oov_words.length) {
      notes.push('<span class="oov">unknown to the vocab: ' + data.oov_words.join(", ") + "</span>");
    }
    noteEl.innerHTML = notes.join(" &middot; ");

    if (!data.predictions.length) {
      emptyState.style.display = "block";
      return;
    }

    const maxProb = data.predictions[0].prob;
    data.predictions.forEach((pred, idx) => {
      const card = document.createElement("div");
      card.className = "result-card";

      const rank = document.createElement("div");
      rank.className = "rank";
      rank.textContent = (idx + 1) + ".";

      const word = document.createElement("div");
      word.className = "word";
      word.textContent = pred.word;

      const track = document.createElement("div");
      track.className = "score-bar-track";
      const fill = document.createElement("div");
      fill.className = "score-bar-fill";
      fill.style.width = (100 * pred.prob / maxProb) + "%";
      fill.style.background = probColor(pred.prob / maxProb);
      track.appendChild(fill);

      const pct = document.createElement("div");
      pct.className = "pct";
      pct.textContent = (100 * pred.prob).toFixed(2) + "%";

      card.appendChild(rank);
      card.appendChild(word);
      card.appendChild(track);
      card.appendChild(pct);
      resultsEl.appendChild(card);
    });
  } catch (e) {
    noteEl.innerHTML = '<span class="err">' + e + "</span>";
  } finally {
    goEl.disabled = false;
    goEl.textContent = "Predict next word";
  }
}

goEl.addEventListener("click", predict);
textEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) predict();
});
