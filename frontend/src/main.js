const STAGE_LABELS = {
  uploaded: "Upload",
  parsing: "Reading paper",
  indexing: "Building knowledge base",
  summarizing: "Summarizing",
  planning: "Planning episode",
  awaiting_approval: "Awaiting your approval",
  scripting: "Writing dialogue",
  voice: "Creating audio",
  completed: "Ready",
  failed: "Failed",
};

const app = document.getElementById("app");
let pollTimer = null;
let paperId = null;
let audioReady = false;
let cues = [];
let activeIndex = -1;
let generating = false;
let planRendered = false;

app.innerHTML = `
  <div class="container">
    <section class="hero">
      <h1>Agentic Podcast Demo</h1>
      <p>Upload a research PDF. Two hosts walk you through it, grounded in the paper — and you can jump in and ask.</p>
    </section>

    <section class="card upload-card">
      <h2>Upload your paper</h2>
      <div class="upload-row">
        <label class="file-label">
          Choose PDF
          <input id="pdf-input" type="file" accept="application/pdf" />
        </label>
        <button id="upload-btn" disabled>Upload &amp; plan</button>
        <span id="file-name" class="mono empty">No file selected</span>
      </div>
      <p id="upload-error" class="error"></p>
    </section>

    <section id="plan-section" class="card plan-card hidden">
      <div class="plan-head">
        <p class="eyebrow">Before we record</p>
        <h2 id="plan-title">Episode plan</h2>
        <p class="plan-sub">This is what the hosts will cover. Remove or refocus anything before we generate the audio — so you get the episode you actually want.</p>
      </div>
      <ol id="plan-segments" class="plan-segments"></ol>
      <div class="plan-directive">
        <label for="plan-instruction">Want changes? Tell the agent (optional)</label>
        <textarea id="plan-instruction" rows="3" placeholder="e.g. Skip the history. Focus on the case study and limitations. Explain human iPSCs before organoids. Keep a technical, no-hype tone."></textarea>
      </div>
      <div class="plan-actions">
        <button id="revise-btn" type="button" class="secondary">Revise plan</button>
        <button id="generate-btn" type="button">Generate podcast</button>
      </div>
      <p id="plan-status" class="plan-status"></p>
    </section>

    <section id="player-section" class="card player-card hidden">
      <div class="player-header">
        <div>
          <p class="eyebrow">Your podcast <span id="mode-pill" class="mode-pill episode hidden">Episode</span></p>
          <h2 id="episode-title">Generating…</h2>
          <p id="player-status" class="player-status">Starting pipeline…</p>
        </div>
        <a id="download-link" class="download-link hidden" href="#" download>Download audio</a>
      </div>
      <audio id="podcast-player" controls class="hidden"></audio>
      <div id="progress-bar" class="progress-bar"><div id="progress-fill" class="progress-fill"></div></div>

      <div id="join-bar" class="join-bar hidden">
        <button id="join-btn" type="button">Raise hand &amp; ask</button>
        <span id="join-hint" class="join-hint">Tap to pause the hosts and ask a question by voice.</span>
      </div>

      <div id="transcript" class="transcript hidden"></div>
    </section>

    <section class="card details-card">
      <button id="toggle-details" class="details-toggle" type="button">Show pipeline details</button>
      <div id="details-panel" class="details-panel hidden">
        <p id="paper-id" class="mono empty">paper_id: —</p>
        <div id="stages" class="stages"></div>
      </div>
    </section>
  </div>
`;

const pdfInput = document.getElementById("pdf-input");
const uploadBtn = document.getElementById("upload-btn");
const fileName = document.getElementById("file-name");
const uploadError = document.getElementById("upload-error");
const playerSection = document.getElementById("player-section");
const episodeTitle = document.getElementById("episode-title");
const playerStatus = document.getElementById("player-status");
const podcastPlayer = document.getElementById("podcast-player");
const downloadLink = document.getElementById("download-link");
const progressFill = document.getElementById("progress-fill");
const joinBar = document.getElementById("join-bar");
const joinBtn = document.getElementById("join-btn");
const joinHint = document.getElementById("join-hint");
const transcriptEl = document.getElementById("transcript");
const toggleDetails = document.getElementById("toggle-details");
const detailsPanel = document.getElementById("details-panel");
const paperIdEl = document.getElementById("paper-id");
const stagesEl = document.getElementById("stages");
const planSection = document.getElementById("plan-section");
const planTitle = document.getElementById("plan-title");
const planSegments = document.getElementById("plan-segments");
const planInstruction = document.getElementById("plan-instruction");
const reviseBtn = document.getElementById("revise-btn");
const generateBtn = document.getElementById("generate-btn");
const planStatus = document.getElementById("plan-status");

let selectedFile = null;

toggleDetails.addEventListener("click", () => {
  const hidden = detailsPanel.classList.toggle("hidden");
  toggleDetails.textContent = hidden ? "Show pipeline details" : "Hide pipeline details";
});

pdfInput.addEventListener("change", () => {
  selectedFile = pdfInput.files?.[0] || null;
  uploadBtn.disabled = !selectedFile;
  fileName.textContent = selectedFile ? selectedFile.name : "No file selected";
  fileName.classList.toggle("empty", !selectedFile);
});

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  uploadError.textContent = "";
  uploadBtn.disabled = true;
  resetPlayer();
  playerSection.classList.remove("hidden");
  episodeTitle.textContent = selectedFile.name.replace(/\.pdf$/i, "");
  playerStatus.textContent = "Uploading and starting pipeline…";
  progressFill.style.width = "8%";

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res = await fetch("/api/podcast/upload-pdf", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    paperId = data.paper_id;
    paperIdEl.textContent = `paper_id: ${paperId}`;
    paperIdEl.classList.remove("empty");
    startPolling();
  } catch (err) {
    uploadError.textContent = `Upload failed: ${err.message}`;
    uploadBtn.disabled = false;
    playerStatus.textContent = "Upload failed.";
  }
});

function resetPlayer() {
  audioReady = false;
  generating = false;
  planRendered = false;
  cues = [];
  activeIndex = -1;
  podcastPlayer.classList.add("hidden");
  downloadLink.classList.add("hidden");
  joinBar.classList.add("hidden");
  transcriptEl.classList.add("hidden");
  transcriptEl.innerHTML = "";
  planSection.classList.add("hidden");
  planSegments.innerHTML = "";
  planInstruction.value = "";
  planStatus.textContent = "";
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refreshAll, 2000);
  refreshAll();
}

function planReviewOpen(status) {
  const approval = status.stages?.awaiting_approval;
  const scripting = status.stages?.scripting;
  return (
    approval?.status === "running" &&
    (!scripting || scripting.status === "pending") &&
    !generating
  );
}

async function refreshAll() {
  if (!paperId) return;
  try {
    const statusRes = await fetch(`/api/podcast/${paperId}/status`);
    if (statusRes.ok) {
      const status = await statusRes.json();
      renderStages(status);
      if (status.current_stage === "failed") {
        stopPolling();
        uploadBtn.disabled = false;
        playerStatus.textContent = status.error || "Pipeline failed.";
        return;
      }

      // Pause at the plan: show the proposed episode and wait for the user to
      // approve or steer it before any audio is generated.
      if (planReviewOpen(status)) {
        if (!planRendered) await loadAndRenderPlan();
        progressFill.style.width = "55%";
        playerStatus.textContent = "Episode plan ready — review it below.";
        return;
      }

      updateProgress(status);
      if (!audioReady) playerStatus.textContent = stageMessage(status);
    }

    const outlineRes = await fetch(`/api/podcast/${paperId}/outline`);
    if (outlineRes.ok) {
      const outline = await outlineRes.json();
      episodeTitle.textContent = outline.episode_title || episodeTitle.textContent;
    }

    const voiceRes = await fetch(`/api/podcast/${paperId}/voice`);
    if (voiceRes.ok) {
      const voice = await voiceRes.json();
      if (voice.status === "running" && !audioReady) {
        playerStatus.textContent = voice.message || "Creating podcast audio…";
      }
      if (voice.status === "completed" && voice.audio_url && !audioReady) {
        await onAudioReady(voice.audio_url);
      }
    }
  } catch (err) {
    console.error(err);
  }
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

async function onAudioReady(audioUrl) {
  audioReady = true;
  stopPolling();
  uploadBtn.disabled = false;
  podcastPlayer.src = audioUrl;
  podcastPlayer.classList.remove("hidden");
  downloadLink.href = `/api/podcast/${paperId}/audio/download`;
  downloadLink.classList.remove("hidden");
  joinBar.classList.remove("hidden");
  playerStatus.textContent = "Your podcast is ready. Press play — and raise your hand anytime.";
  progressFill.style.width = "100%";

  try {
    const cuesRes = await fetch(`/api/podcast/${paperId}/cues`);
    if (cuesRes.ok) {
      const track = await cuesRes.json();
      cues = track.cues || [];
      renderTranscript();
    }
  } catch (_) {}
}

function stageMessage(status) {
  const stage = status.current_stage;
  const info = status.stages?.[stage];
  if (info?.message) return info.message;
  return `Working on: ${STAGE_LABELS[stage] || stage}…`;
}

function updateProgress(status) {
  if (audioReady) return;
  const order = ["uploaded", "parsing", "indexing", "summarizing", "planning", "awaiting_approval", "scripting", "voice", "completed"];
  const idx = order.indexOf(status.current_stage);
  const pct = idx >= 0 ? Math.round(((idx + 1) / order.length) * 100) : 10;
  progressFill.style.width = `${pct}%`;
}

function renderStages(status) {
  const order = ["uploaded", "parsing", "indexing", "summarizing", "planning", "awaiting_approval", "scripting", "voice", "completed"];
  stagesEl.innerHTML = order
    .map((key) => {
      const stage = status.stages?.[key] || { status: "pending", message: "" };
      return `
        <div class="stage">
          <div>
            <div class="stage-name">${STAGE_LABELS[key] || key}</div>
            <div class="stage-msg">${escapeHtml(stage.message || "")}</div>
          </div>
          <span class="badge ${stage.status}">${stage.status}</span>
        </div>`;
    })
    .join("");
}

async function loadAndRenderPlan() {
  const res = await fetch(`/api/podcast/${paperId}/outline`);
  if (!res.ok) return;
  const outline = await res.json();
  renderPlan(outline);
  planRendered = true;
  planSection.classList.remove("hidden");
  uploadBtn.disabled = false;
}

function renderPlan(outline) {
  planTitle.textContent = outline.episode_title || "Episode plan";
  episodeTitle.textContent = outline.episode_title || episodeTitle.textContent;
  const segs = outline.segments || [];
  planSegments.innerHTML = segs
    .map(
      (seg) => `
      <li class="plan-seg">
        <div class="plan-seg-title">${escapeHtml(seg.title || "")}</div>
        <div class="plan-seg-goal">${escapeHtml(seg.goal || "")}</div>
      </li>`
    )
    .join("");
}

reviseBtn.addEventListener("click", async () => {
  const instruction = planInstruction.value.trim();
  if (!instruction) {
    planStatus.textContent = "Type what you'd like changed, then Revise.";
    return;
  }
  reviseBtn.disabled = true;
  generateBtn.disabled = true;
  planStatus.textContent = "Revising the plan…";
  try {
    const res = await fetch(`/api/podcast/${paperId}/outline/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction }),
    });
    if (!res.ok) throw new Error(await res.text());
    const outline = await res.json();
    renderPlan(outline);
    planInstruction.value = "";
    planStatus.textContent = "Updated. Review again, revise more, or generate.";
  } catch (err) {
    planStatus.textContent = `Couldn't revise: ${err.message}`;
  } finally {
    reviseBtn.disabled = false;
    generateBtn.disabled = false;
  }
});

generateBtn.addEventListener("click", async () => {
  const instruction = planInstruction.value.trim();
  reviseBtn.disabled = true;
  generateBtn.disabled = true;
  planStatus.textContent = "Approving plan…";
  try {
    const res = await fetch(`/api/podcast/${paperId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction }),
    });
    if (!res.ok) throw new Error(await res.text());
    generating = true;
    planSection.classList.add("hidden");
    playerStatus.textContent = "Generating your podcast…";
    progressFill.style.width = "70%";
    if (!pollTimer) startPolling();
    else refreshAll();
  } catch (err) {
    planStatus.textContent = `Couldn't start: ${err.message}`;
    reviseBtn.disabled = false;
    generateBtn.disabled = false;
  }
});

function renderTranscript() {
  transcriptEl.classList.remove("hidden");
  transcriptEl.innerHTML = cues
    .map(
      (cue) => `
      <div class="t-turn" data-index="${cue.index}">
        <span class="t-speaker ${cue.speaker}">${cue.speaker}</span>
        <span class="t-text">${escapeHtml(cue.text)}</span>
      </div>`
    )
    .join("");
}

podcastPlayer.addEventListener("timeupdate", () => {
  if (!cues.length) return;
  const ms = podcastPlayer.currentTime * 1000;
  const current = cues.find((c) => ms >= c.start_ms && ms < c.end_ms);
  const idx = current ? current.index : -1;
  if (idx !== activeIndex) {
    activeIndex = idx;
    transcriptEl.querySelectorAll(".t-turn").forEach((el) => {
      el.classList.toggle("active", Number(el.dataset.index) === idx);
    });
    const activeEl = transcriptEl.querySelector(".t-turn.active");
    if (activeEl) activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
});

// ---- Join the conversation (voice) ----
let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let pausedAtMs = 0;
let pausedIndex = -1;

joinBtn.addEventListener("click", async () => {
  if (!audioReady) return;
  if (!recording) {
    await startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    pausedAtMs = Math.round(podcastPlayer.currentTime * 1000);
    pausedIndex = activeIndex;
    podcastPlayer.pause();

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      sendQuestion();
    };
    mediaRecorder.start();
    recording = true;
    joinBtn.textContent = "Stop & send";
    joinBtn.classList.add("recording");
    joinHint.textContent = "Listening… ask your question, then tap Stop & send.";
  } catch (err) {
    joinHint.textContent = `Mic error: ${err.message}`;
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  recording = false;
  joinBtn.textContent = "Raise hand & ask";
  joinBtn.classList.remove("recording");
}

async function sendQuestion() {
  joinHint.textContent = "The hosts are thinking…";
  joinBtn.disabled = true;

  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  const form = new FormData();
  form.append("file", blob, "question.webm");
  form.append("current_index", String(pausedIndex));
  form.append("resume_ms", String(pausedAtMs));

  try {
    const res = await fetch(`/api/podcast/${paperId}/ask-voice`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    showInterjection(data);
  } catch (err) {
    joinHint.textContent = `Could not get an answer: ${err.message}`;
    joinBtn.disabled = false;
  }
}

function showInterjection(data) {
  const block = document.createElement("div");
  block.className = "interjection answering";
  const turnsHtml = (data.answer_turns || [])
    .map(
      (t) => `<div class="t-turn"><span class="t-speaker ${t.speaker}">${t.speaker}</span><span class="t-text">${escapeHtml(t.text)}</span></div>`
    )
    .join("");
  block.innerHTML = `
    <div class="ij-label answering-label">Answering your question</div>
    <div class="you">You asked: ${escapeHtml(data.question)}${data.grounded ? "" : ' <em class="ungrounded">(not covered in the paper)</em>'}</div>
    ${turnsHtml}
    ${renderTrace(data.trace)}
    <div class="ij-footer hidden">Back to the episode</div>
  `;
  transcriptEl.appendChild(block);
  block.scrollIntoView({ block: "nearest", behavior: "smooth" });

  playerSection.classList.add("answering-mode");

  if (data.audio_url) {
    const answerAudio = new Audio(data.audio_url);
    joinHint.textContent = "The hosts are answering your question…";
    answerAudio.play();
    answerAudio.onended = () => endInterjection(block, data.resume_ms);
  } else {
    endInterjection(block, data.resume_ms);
  }
}

// Render the agent's tool-call trace so its decisions are visible: did it
// decide to search the paper, with what queries, and what did it find?
function renderTrace(trace) {
  if (!Array.isArray(trace) || trace.length === 0) {
    return '<div class="trace trace-empty">Agent answered without searching the paper.</div>';
  }
  const rows = trace
    .map((step) => {
      const args = step.arguments || {};
      let argText = "";
      if (step.tool === "web_search") {
        argText = args.query || "";
      } else if (Array.isArray(args.queries)) {
        argText = args.queries.join(" · ");
      } else {
        argText = String(args.queries || "");
      }
      const urls = (step.source_urls || [])
        .slice(0, 2)
        .map((u) => `<span class="trace-url">${escapeHtml(u)}</span>`)
        .join("");
      return `
        <div class="trace-step">
          <span class="trace-tool">${escapeHtml(step.tool)}</span>
          <span class="trace-args">${escapeHtml(argText)}</span>
          <span class="trace-result">${escapeHtml(step.result_summary || "")}</span>
          ${urls}
        </div>`;
    })
    .join("");
  return `
    <details class="trace" open>
      <summary>Agent trace · ${trace.length} tool call${trace.length > 1 ? "s" : ""}</summary>
      ${rows}
    </details>`;
}

function endInterjection(block, resumeMs) {
  // Clear audible + visual signal that the answer is over and the show resumes.
  block.classList.remove("answering");
  const label = block.querySelector(".ij-label");
  if (label) {
    label.textContent = "Answered";
    label.classList.remove("answering-label");
  }
  const footer = block.querySelector(".ij-footer");
  if (footer) footer.classList.remove("hidden");
  playerSection.classList.remove("answering-mode");

  playChime();
  joinHint.textContent = "Resuming the episode…";

  setTimeout(() => resumeEpisode(resumeMs), 900);
}

function resumeEpisode(resumeMs) {
  joinBtn.disabled = false;
  joinHint.textContent = "Back to the episode. Raise your hand anytime.";
  if (typeof resumeMs === "number" && resumeMs >= 0) {
    podcastPlayer.currentTime = resumeMs / 1000;
  }
  podcastPlayer.play();
}

// Soft two-note chime to mark the answer->episode boundary.
function playChime() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ctx = new Ctx();
    const now = ctx.currentTime;
    [
      [660, 0.0],
      [880, 0.12],
    ].forEach(([freq, offset]) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.18, now + offset + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.22);
      osc.connect(gain).connect(ctx.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.24);
    });
    setTimeout(() => ctx.close(), 600);
  } catch (_) {}
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
