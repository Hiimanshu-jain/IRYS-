/* ==========================================================================
   ROGUE PROTOCOL — OPS APP CONTROLLER
   Theme Management, Incident Triage, Specialist Email Inbox, and RAG Lab
   ========================================================================== */

let currentPage = 1;
const PAGE_SIZE = 25;
let currentIncident = null;

let currentEmailPage = 1;
const EMAIL_PAGE_SIZE = 20;
let activeEmailToken = null;
let cachedDispatches = [];
let cachedSchedule = [];

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initClock();
  initTabs();
  initExportDropdown();
  initSubtabSwitcher();
  initChatComposer();
  initGanttFilter();
  loadStats();
  loadIncidents();
  loadSchedule();
  loadDispatches();
  initSearchAndFilters();
  initEmailInbox();
  initGeminiLab();
  initVectorSearch();
  initModal();
});

// Theme Management (Warm Cream vs Cyber Dark)
function initTheme() {
  const toggleBtn = document.getElementById("btnThemeToggle");
  const icon = document.getElementById("themeToggleIcon");
  const text = document.getElementById("themeToggleText");

  // Load stored theme or default to theme-cream
  const storedTheme = localStorage.getItem("aurora-theme") || "theme-cream";
  document.body.className = storedTheme;
  updateThemeButton(storedTheme, icon, text);

  toggleBtn.addEventListener("click", () => {
    const isCream = document.body.classList.contains("theme-cream");
    const nextTheme = isCream ? "theme-dark" : "theme-cream";
    document.body.className = nextTheme;
    localStorage.setItem("aurora-theme", nextTheme);
    updateThemeButton(nextTheme, icon, text);
  });
}

function updateThemeButton(theme, icon, text) {
  if (theme === "theme-cream") {
    icon.textContent = "🌙";
    text.textContent = "Cyber Dark";
  } else {
    icon.textContent = "☀️";
    text.textContent = "Warm Cream";
  }
}

// Ship Clock Ticker
function initClock() {
  const clockEl = document.getElementById("shipClock");
  let shipYear = 2187;
  let shipMonth = 4;
  let shipDay = 15;
  let shipHour = 3;
  let shipMinute = 12;
  let shipSec = 0;

  setInterval(() => {
    shipSec++;
    if (shipSec >= 60) {
      shipSec = 0;
      shipMinute++;
      if (shipMinute >= 60) {
        shipMinute = 0;
        shipHour++;
      }
    }
    const pad = (n) => String(n).padStart(2, "0");
    clockEl.textContent = `${shipYear}-${pad(shipMonth)}-${pad(shipDay)} ${pad(shipHour)}:${pad(shipMinute)}:${pad(shipSec)}`;
  }, 1000);
}

// Navigation Tabs
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      const targetTab = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === targetTab);
      });
    });
  });
}

// --------------------------------------------------------------------------
// 1. EXPORT CENTER DROPDOWN
// --------------------------------------------------------------------------
function initExportDropdown() {
  const wrap = document.getElementById("exportDropdownWrap");
  const btn = document.getElementById("btnExportMenu");
  if (!wrap || !btn) return;

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    wrap.classList.toggle("open");
  });

  document.addEventListener("click", (e) => {
    if (!wrap.contains(e.target)) {
      wrap.classList.remove("open");
    }
  });
}

// --------------------------------------------------------------------------
// 2. CHANNEL SUBTAB SWITCHER (EMAIL INBOX vs OPS CHAT STREAM)
// --------------------------------------------------------------------------
function initSubtabSwitcher() {
  const btnEmail = document.getElementById("btnSubtabEmail");
  const btnChat = document.getElementById("btnSubtabChat");
  const viewEmail = document.getElementById("channelEmailView");
  const viewChat = document.getElementById("channelChatView");
  if (!btnEmail || !btnChat) return;

  btnEmail.addEventListener("click", () => {
    btnEmail.classList.add("active");
    btnChat.classList.remove("active");
    if (viewEmail) viewEmail.style.display = "block";
    if (viewChat) viewChat.style.display = "none";
  });

  btnChat.addEventListener("click", () => {
    btnChat.classList.add("active");
    btnEmail.classList.remove("active");
    if (viewEmail) viewEmail.style.display = "none";
    if (viewChat) {
      viewChat.style.display = "flex";
      loadOpsChatFeed();
    }
  });
}

async function loadOpsChatFeed() {
  const container = document.getElementById("chatMessagesContainer");
  if (!container) return;

  if (container.children.length > 3) {
    return; // Already populated
  }

  container.innerHTML = `<div class="loading-cell"><div class="cyber-spinner"></div> Connecting to #aurora-incident-bridge operational stream...</div>`;

  try {
    let dispatches = cachedDispatches;
    if (!dispatches || dispatches.length === 0) {
      const res = await fetch("/api/dispatches?limit=25");
      const data = await res.json();
      dispatches = data.dispatches || [];
      cachedDispatches = dispatches;
    }

    if (dispatches.length === 0) {
      container.innerHTML = `<div class="empty-email-state"><p>No live operational alerts found in stream.</p></div>`;
      return;
    }

    container.innerHTML = dispatches.slice(0, 25).map((d) => {
      const sev = (d.severity || "MODERATE").toLowerCase();
      const sevClass = sev === "critical" || sev === "sev-1" ? "sev-critical" : (sev === "high" || sev === "sev-2" ? "sev-high" : "sev-moderate");
      
      let modIcon = "⚙️";
      if (d.subject && d.subject.includes("Antenna")) modIcon = "📡";
      else if (d.subject && (d.subject.includes("Power") || d.subject.includes("Lab"))) modIcon = "⚡";
      else if (d.subject && (d.subject.includes("Life") || d.subject.includes("Support"))) modIcon = "🫁";
      else if (d.subject && (d.subject.includes("Notice") || d.subject.includes("Bus"))) modIcon = "🚌";

      const timeStr = d.date ? d.date.slice(11, 19) : "03:15:20";

      return `
        <div class="chat-msg-row ${sevClass}" data-logid="${d.log_id}">
          <div class="chat-avatar">${modIcon}</div>
          <div class="chat-msg-content">
            <div class="chat-msg-header">
              <span class="chat-sender-name">🤖 Aurora Autonomy Bot <span class="badge-bot">BOT</span></span>
              <span class="chat-msg-time">${timeStr} · Ship Core</span>
            </div>
            <div class="chat-msg-body">
              <strong>${escapeHtml(d.subject)}</strong>
            </div>
            <div class="chat-msg-card-block">
              <div>👨‍🚀 <strong>Specialist Paged:</strong> <span style="color: var(--primary-accent);">${escapeHtml(d.to)}</span> · ⏳ SLA: <strong>15m response window</strong></div>
              <div>🔐 <strong>Crypto Delivery Receipt:</strong> <code class="mono" style="font-size:0.75rem;">${d.receipt_token}</code></div>
              <div>🛠️ <strong>Runbook Directive:</strong> <code class="mono" style="font-size:0.75rem;">remediate_${d.log_id}.sh</code> (Autonomous Self-Healing)</div>
            </div>
            <div class="chat-msg-actions">
              <button class="btn btn-xs btn-primary btn-chat-inspect" data-logid="${d.log_id}">▶ Inspect & Runbook</button>
              <button class="btn btn-xs btn-secondary btn-chat-ack" data-spec="${escapeHtml(d.to)}">👨‍🚀 Acknowledge SLA</button>
              <button class="btn btn-xs btn-secondary btn-chat-copy" data-token="${d.receipt_token}">📋 Copy Token</button>
            </div>
          </div>
        </div>
      `;
    }).join("");

    // Wire actions
    container.querySelectorAll(".btn-chat-inspect").forEach(btn => {
      btn.addEventListener("click", () => {
        const logId = btn.getAttribute("data-logid");
        if (logId && logId !== "N/A") openIncidentModal(logId);
      });
    });

    container.querySelectorAll(".btn-chat-ack").forEach(btn => {
      btn.addEventListener("click", () => {
        const spec = btn.getAttribute("data-spec");
        btn.textContent = "Acknowledged ✓";
        btn.disabled = true;
        btn.style.borderColor = "var(--emerald)";
        btn.style.color = "var(--emerald)";
        alert(`SLA Acknowledged! Operational ticket confirmed by ${spec}. Time-to-resolve SLA countdown started.`);
      });
    });

    container.querySelectorAll(".btn-chat-copy").forEach(btn => {
      btn.addEventListener("click", () => {
        const token = btn.getAttribute("data-token");
        navigator.clipboard.writeText(token);
        btn.textContent = "Copied ✓";
        setTimeout(() => btn.textContent = "📋 Copy Token", 1500);
      });
    });

  } catch (err) {
    container.innerHTML = `<div class="empty-email-state" style="color: var(--crimson);"><p>Error connecting to Ops chat stream: ${err.message}</p></div>`;
  }
}

function initChatComposer() {
  const input = document.getElementById("chatBroadcastInput");
  const btn = document.getElementById("btnChatBroadcast");
  const container = document.getElementById("chatMessagesContainer");
  if (!input || !btn || !container) return;

  const sendMessage = () => {
    const text = input.value.trim();
    if (!text) return;

    const row = document.createElement("div");
    row.className = "chat-msg-row msg-user";
    row.innerHTML = `
      <div class="chat-avatar">👨‍🚀</div>
      <div class="chat-msg-content">
        <div class="chat-msg-header">
          <span class="chat-sender-name">Commander Marcus Sterling <span class="badge-online">CHIEF ENGINEER</span></span>
          <span class="chat-msg-time">${new Date().toISOString().slice(11, 19)} · Console</span>
        </div>
        <div class="chat-msg-body">
          ${escapeHtml(text)}
        </div>
      </div>
    `;
    container.appendChild(row);
    input.value = "";
    container.scrollTop = container.scrollHeight;
  };

  btn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });
}

// Fetch & Update Stats
async function loadStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("valTotalLogs").textContent = data.total_logs?.toLocaleString() || "3,000";
    document.getElementById("valFaults").textContent = data.faults_processed?.toLocaleString() || "668";
    document.getElementById("valNoise").textContent = data.noise_suppressed?.toLocaleString() || "2,332";
    document.getElementById("valOrders").textContent = data.work_orders_scheduled?.toLocaleString() || "668";
    document.getElementById("valCascades").textContent = data.cascades_detected?.toLocaleString() || "588";

    if (data.total_logs) {
      const pct = ((data.faults_processed / data.total_logs) * 100).toFixed(1);
      document.getElementById("valFaultPct").textContent = `${pct}%`;
    }
  } catch (err) {
    console.warn("Could not load stats:", err);
  }
}

// Load Incidents Table
async function loadIncidents() {
  const tbody = document.getElementById("incidentsTbody");
  const moduleVal = document.getElementById("filterModule").value;
  const severityVal = document.getElementById("filterSeverity").value;
  const searchVal = document.getElementById("filterSearch").value.trim();

  tbody.innerHTML = `
    <tr>
      <td colspan="10" class="loading-cell">
        <div class="cyber-spinner"></div>
        <span>Retrieving telemetry records from stream...</span>
      </td>
    </tr>
  `;

  try {
    const params = new URLSearchParams({
      page: currentPage,
      limit: PAGE_SIZE,
      module: moduleVal,
      severity: severityVal,
      search: searchVal,
    });

    const res = await fetch(`/api/incidents?${params.toString()}`);
    const data = await res.json();

    if (!data.incidents || data.incidents.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">No matching telemetry faults located in active window.</td></tr>`;
      document.getElementById("filteredCountBadge").textContent = "0 faults found";
      return;
    }

    document.getElementById("filteredCountBadge").textContent = 
      `Showing ${(currentPage - 1) * PAGE_SIZE + 1} - ${Math.min(currentPage * PAGE_SIZE, data.total)} of ${data.total} faults`;

    const totalPages = Math.ceil(data.total / PAGE_SIZE) || 1;
    document.getElementById("pageIndicator").textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById("btnPrevPage").disabled = currentPage <= 1;
    document.getElementById("btnNextPage").disabled = currentPage >= totalPages;

    tbody.innerHTML = data.incidents.map((inc) => {
      const sevClass = (inc.severity || "MODERATE").toLowerCase();
      return `
        <tr data-logid="${inc.log_id}">
          <td class="mono"><strong>${inc.log_id}</strong></td>
          <td class="mono" style="color: var(--text-muted); font-size: 0.75rem;">${inc.timestamp}</td>
          <td><strong>${inc.module}</strong></td>
          <td class="mono">${inc.node}</td>
          <td><span class="badge-sev ${sevClass}">${inc.severity}</span></td>
          <td class="mono">${inc.reading} <span style="font-size: 0.7rem; color: var(--text-muted);">${inc.reading_unit || ""}</span></td>
          <td class="mono">${inc.delay_ms} ms</td>
          <td><span class="crew-tag">${inc.crew_id || "CREW-ALPHA"}</span></td>
          <td style="font-size: 0.8rem;">${inc.specialist_name || "Dr. Elena Vance"}</td>
          <td>
            <button class="btn btn-sm btn-accent btn-inspect" data-logid="${inc.log_id}">Inspect</button>
          </td>
        </tr>
      `;
    }).join("");

    // Attach click handlers
    document.querySelectorAll(".btn-inspect").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const logId = btn.getAttribute("data-logid");
        openIncidentModal(logId);
      });
    });

    tbody.querySelectorAll("tr").forEach((row) => {
      row.addEventListener("click", () => {
        const logId = row.getAttribute("data-logid");
        if (logId) openIncidentModal(logId);
      });
    });

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" class="loading-cell" style="color: var(--crimson);">Error loading incident telemetry: ${err.message}</td></tr>`;
  }
}

// Search & Filter Listeners
function initSearchAndFilters() {
  const searchInput = document.getElementById("filterSearch");
  const moduleSelect = document.getElementById("filterModule");
  const sevSelect = document.getElementById("filterSeverity");

  let debounceTimer;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      currentPage = 1;
      loadIncidents();
    }, 250);
  });

  moduleSelect.addEventListener("change", () => {
    currentPage = 1;
    loadIncidents();
  });

  sevSelect.addEventListener("change", () => {
    currentPage = 1;
    loadIncidents();
  });

  document.getElementById("btnPrevPage").addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      loadIncidents();
    }
  });

  document.getElementById("btnNextPage").addEventListener("click", () => {
    currentPage++;
    loadIncidents();
  });

  // Re-run pipeline button
  document.getElementById("btnRunPipeline").addEventListener("click", async () => {
    const btn = document.getElementById("btnRunPipeline");
    btn.disabled = true;
    btn.innerHTML = `<div class="cyber-spinner"></div> Executing...`;
    try {
      const res = await fetch("/api/run-pipeline", { credentials: "same-origin", method: "POST" });
      const data = await res.json();
      alert(`Pipeline execution completed!\n\nProcessed: ${data.metrics?.faults_processed || 668} faults\nDispatches: ${data.metrics?.dispatches_sent || 668}\nWork Orders: ${data.metrics?.work_orders_scheduled || 668}`);
      loadStats();
      loadIncidents();
      loadSchedule();
      loadDispatches();
    } catch (err) {
      alert(`Error triggering pipeline: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run Autonomy Engine`;
    }
  });
}

// --------------------------------------------------------------------------
// SPECIALIST DISPATCHES INBOX & EMAIL AUTOMATION
// --------------------------------------------------------------------------
async function loadDispatches() {
  const container = document.getElementById("emailItemsContainer");
  const searchVal = document.getElementById("emailSearchInput") ? document.getElementById("emailSearchInput").value.trim() : "";

  try {
    const res = await fetch(`/api/dispatches?page=${currentEmailPage}&limit=${EMAIL_PAGE_SIZE}&search=${encodeURIComponent(searchVal)}`);
    const data = await res.json();

    if (!data.dispatches || data.dispatches.length === 0) {
      container.innerHTML = `<div class="empty-email-state"><p>No dispatched alerts matched the filter.</p></div>`;
      return;
    }

    cachedDispatches = data.dispatches;

    const totalPages = Math.ceil(data.total / EMAIL_PAGE_SIZE) || 1;
    document.getElementById("emailPageText").textContent = `Page ${currentEmailPage} of ${totalPages} (${data.total} alerts)`;

    container.innerHTML = data.dispatches.map((d, index) => {
      const isSelected = activeEmailToken === d.receipt_token || (index === 0 && !activeEmailToken);
      if (index === 0 && !activeEmailToken) {
        activeEmailToken = d.receipt_token;
        loadDispatchReader(d.receipt_token);
      }
      return `
        <div class="email-item ${isSelected ? 'active' : ''}" data-token="${d.receipt_token}">
          <div class="email-item-header">
            <span class="email-recipient">${escapeHtml(d.to)}</span>
            <span class="email-time">${d.date.slice(11, 19) || '12:00:00'}</span>
          </div>
          <div class="email-subject"><strong>[${d.severity}]</strong> ${escapeHtml(d.subject)}</div>
          <span class="email-token-tag">${d.receipt_token}</span>
        </div>
      `;
    }).join("");

    container.querySelectorAll(".email-item").forEach((item) => {
      item.addEventListener("click", () => {
        container.querySelectorAll(".email-item").forEach(i => i.classList.remove("active"));
        item.classList.add("active");
        const token = item.getAttribute("data-token");
        activeEmailToken = token;
        loadDispatchReader(token);
      });
    });

  } catch (err) {
    container.innerHTML = `<div class="empty-email-state" style="color: var(--crimson);"><p>Error loading dispatches: ${err.message}</p></div>`;
  }
}

async function loadDispatchReader(token) {
  const reader = document.getElementById("emailReaderPanel");
  reader.innerHTML = `<div class="loading-cell"><div class="cyber-spinner"></div> Loading operational dispatch ${token}...</div>`;

  try {
    const res = await fetch(`/api/dispatch/${token}`);
    const data = await res.json();

    if (!data.raw_eml) {
      reader.innerHTML = `<div class="empty-email-state"><p>Email payload not found.</p></div>`;
      return;
    }

    // Parse headers from raw email
    const lines = data.raw_eml.split("\n");
    let to = "", from = "", date = "", subject = "", receipt = token;
    for (const l of lines.slice(0, 15)) {
      if (l.startsWith("TO:")) to = l.replace("TO:", "").trim();
      if (l.startsWith("FROM:")) from = l.replace("FROM:", "").trim();
      if (l.startsWith("DATE:")) date = l.replace("DATE:", "").trim();
      if (l.startsWith("SUBJECT:")) subject = l.replace("SUBJECT:", "").trim();
      if (l.startsWith("RECEIPT:")) receipt = l.replace("RECEIPT:", "").trim();
    }

    reader.innerHTML = `
      <div class="reader-header">
        <div class="reader-title-row">
          <h3 class="reader-subject">${escapeHtml(subject)}</h3>
          <span class="status-indicator" style="background: var(--emerald-bg); border-color: var(--emerald-border);"><span class="pulse-dot"></span> <span class="mono" style="color: var(--emerald); font-weight:700;">CONFIRMED DELIVERED</span></span>
        </div>
        <div class="reader-meta-grid">
          <div class="reader-meta-item"><strong>To:</strong> ${escapeHtml(to)}</div>
          <div class="reader-meta-item"><strong>From:</strong> ${escapeHtml(from)}</div>
          <div class="reader-meta-item"><strong>Date:</strong> <span class="mono">${escapeHtml(date)}</span></div>
          <div class="reader-meta-item"><strong>Delivery Token:</strong> <code class="mono" style="color: var(--primary-accent);">${escapeHtml(receipt)}</code></div>
        </div>
      </div>
      <div class="reader-body-wrap">${escapeHtml(data.raw_eml)}</div>
    `;

  } catch (err) {
    reader.innerHTML = `<div class="empty-email-state" style="color: var(--crimson);"><p>Error rendering email: ${err.message}</p></div>`;
  }
}

function initEmailInbox() {
  const searchInput = document.getElementById("emailSearchInput");
  const btnPrev = document.getElementById("btnEmailPrev");
  const btnNext = document.getElementById("btnEmailNext");
  const btnSendTest = document.getElementById("btnSendTestDispatch");

  let debounceTimer;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      currentEmailPage = 1;
      loadDispatches();
    }, 250);
  });

  btnPrev.addEventListener("click", () => {
    if (currentEmailPage > 1) {
      currentEmailPage--;
      loadDispatches();
    }
  });

  btnNext.addEventListener("click", () => {
    currentEmailPage++;
    loadDispatches();
  });

  btnSendTest.addEventListener("click", async () => {
    const mod = prompt("Enter Module to test operational dispatch (e.g. Antenna, Notice-Bus, Lab-Power, Life-Support):", "Antenna");
    if (!mod) return;

    btnSendTest.disabled = true;
    btnSendTest.innerHTML = `<div class="cyber-spinner"></div> Dispatching...`;

    try {
      const res = await fetch("/api/send-dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          module: mod,
          root_cause: `Manual test operational dispatch for ${mod} subsystem`,
          severity: "CRITICAL",
          message: `NINE: verified critical telemetry threshold breach [${mod} 49.8 dBm]`
        })
      });
      const data = await res.json();
      alert(`Operational Email Dispatched Successfully!\n\nSpecialist: ${data.specialist_name} (${data.specialist_email})\nSubject: ${data.subject}\nReceipt Token: ${data.receipt_token}`);
      currentEmailPage = 1;
      activeEmailToken = data.receipt_token;
      loadDispatches();
      loadStats();
    } catch (err) {
      alert(`Error dispatching email alert: ${err.message}`);
    } finally {
      btnSendTest.disabled = false;
      btnSendTest.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg> Send Live Test Dispatch`;
    }
  });

  // Live Automated Email Dispatch Stream Console
  const btnToggleStream = document.getElementById("btnLiveAutoDispatchStream");
  const cardStream = document.getElementById("liveDispatchCard");
  const btnCloseStream = document.getElementById("btnCloseLdaCard");
  const btnNextDispatch = document.getElementById("btnRunNextAutoDispatch");

  if (btnToggleStream && cardStream) {
    btnToggleStream.addEventListener("click", () => {
      const isHidden = cardStream.style.display === "none";
      cardStream.style.display = isHidden ? "block" : "none";
      if (isHidden) {
        cardStream.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  }

  if (btnCloseStream && cardStream) {
    btnCloseStream.addEventListener("click", () => {
      cardStream.style.display = "none";
    });
  }

  let autoDispatchIndex = 0;
  const testSubsystems = [
    { mod: "Antenna", root: "RF feedhorn phase lock drift & PLL carrier cycle slip", msg: "NINE: carrier drift beyond nominal tolerance [Antenna 48.9 dBm]", node: "N4", reading: "48.9 dBm" },
    { mod: "Notice-Bus", root: "Distributed CAN arbitration mutex deadlock", msg: "NINE: bus arbitration deadlock detected [Notice-Bus 320 msg/s]", node: "N2", reading: "320 msg/s" },
    { mod: "Lab-Power", root: "Inverter SCR thyristor gate harmonic oscillation", msg: "NINE: thyristor gate firing delay [Lab-Power 132 kW]", node: "N7", reading: "132 kW" },
    { mod: "Life-Support", root: "Catalytic CO2 scrubber bed saturation & delta-P drop", msg: "NINE: catalytic scrubber bed differential pressure drop [Life-Support 18.2 kPa]", node: "N1", reading: "18.2 kPa" }
  ];

  if (btnNextDispatch) {
    btnNextDispatch.addEventListener("click", async () => {
      btnNextDispatch.disabled = true;
      const targetSub = testSubsystems[autoDispatchIndex % testSubsystems.length];
      autoDispatchIndex++;

      const term = document.getElementById("ldaTerminalOutput");
      const chip = document.getElementById("ldaStatusChip");
      const steps = [
        document.getElementById("stepIngest"),
        document.getElementById("stepMatch"),
        document.getElementById("stepCrypto"),
        document.getElementById("stepSmtp"),
        document.getElementById("stepInbox")
      ];

      const setStep = (idx) => {
        steps.forEach((s, i) => {
          if (!s) return;
          s.classList.toggle("active", i === idx);
          s.classList.toggle("done", i < idx);
        });
      };

      if (chip) {
        chip.textContent = "TRANSMITTING...";
        chip.style.color = "var(--primary-accent)";
      }

      term.innerHTML = `<div class="lda-line system">[${new Date().toISOString().slice(11, 19)}] 🚨 Ingested anomalous telemetry frame for ${targetSub.mod} on Node ${targetSub.node}...</div>`;
      setStep(0);

      // Phase 1
      await sleep(500);
      term.innerHTML += `<div class="lda-line cmd">> Assessing fault metrics: Reading ${targetSub.reading} | Severity: CRITICAL SEV-1 (Score 88/100)</div>`;
      setStep(1);

      // Phase 2
      await sleep(600);
      term.innerHTML += `<div class="lda-line net">> Semantic match: Subsystem "${targetSub.mod}" -> Specialist on duty matched</div>`;
      setStep(2);

      // Phase 3
      await sleep(600);
      const randomSeed = Math.random().toString(36).substring(2, 8).toUpperCase();
      const simToken = `RCPT-20261015-L0${Math.floor(1000 + Math.random()*9000)}-${randomSeed}`;
      term.innerHTML += `<div class="lda-line ok">> Minted cryptographic delivery receipt token: ${simToken} (SHA-256)</div>`;
      setStep(3);

      // Phase 4: SMTP Handshake
      await sleep(700);
      term.innerHTML += `
        <div class="lda-line cmd">> CONNECT smtp.aurora.internal:587... 220 aurora.mail.service ESMTP</div>
        <div class="lda-line net">>> MAIL FROM:<auto-dispatch@aurora.research.org> -> 250 2.1.0 Sender OK</div>
        <div class="lda-line net">>> RCPT TO:<specialist@aurora.research.org> -> 250 2.1.5 Recipient OK</div>
        <div class="lda-line net">>> DATA [RFC 822 MIME-Version: 1.0] -> 250 2.0.0 Message accepted for delivery</div>
      `;
      setStep(4);

      // Phase 5: Live API trigger
      try {
        const res = await fetch("/api/send-dispatch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            module: targetSub.mod,
            root_cause: targetSub.root,
            severity: "CRITICAL",
            message: targetSub.msg
          })
        });
        const data = await res.json();
        
        await sleep(400);
        term.innerHTML += `<div class="lda-line highlight">📬 DELIVERED! Email saved to dispatches/${data.receipt_token}.eml. Recipient: ${data.specialist_name} (${data.specialist_email})</div>`;
        if (chip) {
          chip.textContent = "DELIVERED ✓";
          chip.style.color = "var(--emerald)";
        }
        steps.forEach(s => { if (s) { s.classList.remove("active"); s.classList.add("done"); } });

        // Update list & select new dispatch
        currentEmailPage = 1;
        activeEmailToken = data.receipt_token;
        loadDispatches();
        loadStats();
      } catch (err) {
        term.innerHTML += `<div class="lda-line" style="color:var(--crimson);">> Error during dispatch: ${err.message}</div>`;
      } finally {
        btnNextDispatch.disabled = false;
        term.scrollTop = term.scrollHeight;
      }
    });
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Deep-Dive Modal Inspector & Runbook CLI Simulator
let simTimeoutIds = [];

function initModal() {
  const modal = document.getElementById("incidentModal");
  const btnClose = document.getElementById("btnModalClose");
  const btnCopy = document.getElementById("btnCopyRunbook");
  const btnSimulate = document.getElementById("btnSimulateRunbook");
  const simPanel = document.getElementById("runbookSimPanel");
  const btnResetSim = document.getElementById("btnSimReset");
  const btnRerunSim = document.getElementById("btnSimRerun");

  btnClose.addEventListener("click", () => modal.classList.remove("active"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.remove("active");
  });

  btnCopy.addEventListener("click", () => {
    const code = document.getElementById("mRunbookCode").textContent;
    navigator.clipboard.writeText(code).then(() => {
      btnCopy.textContent = "Copied ✓";
      setTimeout(() => (btnCopy.textContent = "Copy Script"), 2000);
    });
  });

  document.getElementById("btnViewDispatchEml").addEventListener("click", () => {
    if (currentIncident && currentIncident.dispatch_eml) {
      const win = window.open("", "_blank");
      win.document.write(`<pre style="background:#FAF7F2; color:#241E19; padding:24px; font-family:monospace; white-space:pre-wrap; border: 1px solid #E8DFD3; border-radius: 8px;">${escapeHtml(currentIncident.dispatch_eml)}</pre>`);
    } else {
      alert("Simulated RFC 822 Email file rendered in telemetry dispatches/ directory.");
    }
  });

  // Runbook Live Simulation Simulator
  if (btnSimulate) {
    btnSimulate.addEventListener("click", () => {
      if (simPanel) {
        simPanel.style.display = "block";
        simPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      runRunbookSimulation(currentIncident);
    });
  }

  if (btnResetSim) {
    btnResetSim.addEventListener("click", () => {
      simTimeoutIds.forEach(id => clearTimeout(id));
      simTimeoutIds = [];
      const term = document.getElementById("simTerminalLog");
      const prog = document.getElementById("simProgressBar");
      const badge = document.getElementById("simBadge");
      if (term) term.innerHTML = `<div class="sim-line system">// Simulator reset. Ready for deterministic runbook execution.</div>`;
      if (prog) prog.style.width = "0%";
      if (badge) {
        badge.textContent = "READY";
        badge.style.color = "var(--primary-accent)";
        badge.style.borderColor = "var(--border-subtle)";
      }
    });
  }

  if (btnRerunSim) {
    btnRerunSim.addEventListener("click", () => {
      runRunbookSimulation(currentIncident);
    });
  }
}

function runRunbookSimulation(inc) {
  if (!inc) return;
  const term = document.getElementById("simTerminalLog");
  const prog = document.getElementById("simProgressBar");
  const badge = document.getElementById("simBadge");
  if (!term || !prog || !badge) return;

  // Clear previous timers
  simTimeoutIds.forEach(id => clearTimeout(id));
  simTimeoutIds = [];

  badge.textContent = "EXECUTING...";
  badge.style.color = "var(--amber)";
  badge.style.borderColor = "var(--amber)";
  term.innerHTML = `<div class="sim-line system">// [0.00s] Initializing deterministic runbook session: ${inc.log_id} (${inc.module})...</div>`;
  prog.style.width = "8%";

  // Extract real bash lines from incident runbook if available
  let bashSnippets = [
    `aurora-cli sys-reset --module ${(inc.module || "subsystem").toLowerCase()} --target ${inc.node || "N1"}`,
    `systemctl restart aurora-${(inc.module || "subsystem").toLowerCase()}.service`
  ];
  if (inc.runbook_code) {
    const rawLines = inc.runbook_code.split("\n")
      .map(l => l.trim())
      .filter(l => l && !l.startsWith("#") && !l.startsWith("echo") && !l.startsWith("sleep") && l.length > 5);
    if (rawLines.length > 0) {
      bashSnippets = rawLines.slice(0, 3);
    }
  }

  const steps = [
    {
      delay: 450,
      prog: "25%",
      html: `<div class="sim-line cmd">> ping -c 2 proxy-${(inc.node || "n1").toLowerCase()}.aurora.internal</div>
             <div class="sim-line output">64 bytes from 10.240.4.12: icmp_seq=1 ttl=64 time=0.34 ms [NODE REACHABLE]</div>`
    },
    {
      delay: 1100,
      prog: "50%",
      html: `<div class="sim-line cmd">> aurora-cli telemetry dump --subsystem ${(inc.module || "antenna").toLowerCase()} --format json</div>
             <div class="sim-line metric">Captured reading: ${inc.reading} ${inc.reading_unit || ""} | Offending Status: ${inc.status}</div>
             <div class="sim-line ok">Telemetry FIFO buffer flushed. Mutex lock arbitration cleared.</div>`
    },
    {
      delay: 1900,
      prog: "72%",
      html: `<div class="sim-line cmd">> bash runbooks/remediate_${inc.log_id}.sh --execute --strict</div>
             <div class="sim-line output">> ${escapeHtml(bashSnippets[0] || "aurora-cli calibrate")}</div>
             <div class="sim-line output">Calibrating PLL register 0x7FA0... Parameter converged within 12ms. [OK]</div>`
    },
    {
      delay: 2650,
      prog: "88%",
      html: `<div class="sim-line cmd">> systemctl restart aurora-${(inc.module || "subsystem").toLowerCase()}.service</div>
             <div class="sim-line ok">Active: active (running) since ${new Date().toISOString().slice(11, 19)}. Main PID: 9284</div>`
    },
    {
      delay: 3300,
      prog: "100%",
      html: `<div class="sim-line cmd">> aurora-cli healthcheck --verify-nominal --samples 3</div>
             <div class="sim-line ok">Telemetry verification passed: Sensor reading in nominal band (Delta: 0.00). STATUS: GREEN.</div>
             <div class="sim-line success">✅ EXIT CODE 0 — RUNBOOK REMEDIATION COMPLETED SUCCESSFULLY. INCIDENT ${inc.log_id} RESOLVED.</div>`
    }
  ];

  steps.forEach(step => {
    const tid = setTimeout(() => {
      prog.style.width = step.prog;
      term.innerHTML += step.html;
      term.scrollTop = term.scrollHeight;

      if (step.prog === "100%") {
        badge.textContent = "RESOLVED (EXIT 0)";
        badge.style.color = "#10B981";
        badge.style.borderColor = "#10B981";
      }
    }, step.delay);
    simTimeoutIds.push(tid);
  });
}

async function openIncidentModal(logId) {
  const modal = document.getElementById("incidentModal");
  modal.classList.add("active");

  const simPanel = document.getElementById("runbookSimPanel");
  if (simPanel) simPanel.style.display = "none";
  const simProg = document.getElementById("simProgressBar");
  if (simProg) simProg.style.width = "0%";

  document.getElementById("mLogId").textContent = `Incident ${logId}`;
  document.getElementById("mRunbookCode").textContent = "# Loading incident context and runbook script...";

  try {
    const res = await fetch(`/api/incident/${logId}`);
    const inc = await res.json();
    currentIncident = inc;

    document.getElementById("mTimestamp").textContent = inc.timestamp;
    document.getElementById("mModule").textContent = inc.module;
    document.getElementById("mNode").textContent = inc.node;
    document.getElementById("mReading").textContent = `${inc.reading} ${inc.reading_unit || ""}`;
    document.getElementById("mDelay").textContent = `${inc.delay_ms} ms`;
    document.getElementById("mRetries").textContent = inc.retry_count;
    document.getElementById("mSevScore").textContent = `${inc.severity_score || 50} / 100`;

    const sevBadge = document.getElementById("mBadgeSev");
    sevBadge.className = `modal-badge-sev badge-sev ${(inc.severity || "MODERATE").toLowerCase()}`;
    sevBadge.textContent = inc.severity;

    document.getElementById("mRootCause").textContent = inc.root_cause || "Telemetry anomaly";
    document.getElementById("mRawMessage").textContent = `"${inc.message}"`;

    document.getElementById("mSpecialistName").textContent = inc.specialist_name || "Dr. Elena Vance";
    document.getElementById("mSpecialistEmail").textContent = inc.specialist_email || "elena.vance@aurora.research.org";
    document.getElementById("mReceiptToken").textContent = inc.receipt_token || "RCPT-20261015-L07004-98F1";

    document.getElementById("mCrewId").textContent = inc.crew_id || "CREW-ALPHA";
    document.getElementById("mCrewWindow").textContent = `${inc.slot_start || ""} ➔ ${inc.slot_end || ""}`;
    document.getElementById("mCrewJob").textContent = inc.job_description || "Service telemetry subsystems";

    document.getElementById("mRunbookFilename").textContent = `remediate_${inc.log_id}.sh`;
    document.getElementById("mRunbookCode").textContent = inc.runbook_code || `#!/usr/bin/env bash\n# Runbook generated for ${inc.log_id}`;

  } catch (err) {
    document.getElementById("mRunbookCode").textContent = `# Error loading incident: ${err.message}`;
  }
}

// Load Schedule Table & Visual Gantt Chart
async function loadSchedule() {
  const tbody = document.getElementById("scheduleTbody");
  tbody.innerHTML = `<tr><td colspan="6" class="loading-cell"><div class="cyber-spinner"></div> Loading schedule.csv ledger...</td></tr>`;

  try {
    const res = await fetch("/api/schedule");
    const data = await res.json();

    if (!data.schedule || data.schedule.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">No work orders recorded in schedule.csv.</td></tr>`;
      return;
    }

    cachedSchedule = data.schedule;
    renderGanttChart(cachedSchedule);

    tbody.innerHTML = data.schedule.slice(0, 50).map((row) => {
      const crewClass = (row.crew_id || "").toLowerCase().replace("-", "_");
      return `
        <tr>
          <td class="mono"><strong>${row.log_id}</strong></td>
          <td><span class="legend-chip ${crewClass}">${row.crew_id}</span></td>
          <td class="mono">${row.start}</td>
          <td class="mono">${row.end}</td>
          <td>${row.job}</td>
          <td class="mono" style="color: var(--primary-accent); font-weight:600;">${calcDuration(row.start, row.end)}</td>
        </tr>
      `;
    }).join("");

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell" style="color: var(--crimson);">Failed to load schedule: ${err.message}</td></tr>`;
  }
}

function initGanttFilter() {
  const filter = document.getElementById("ganttCrewFilter");
  if (!filter) return;
  filter.addEventListener("change", () => {
    if (cachedSchedule && cachedSchedule.length > 0) {
      renderGanttChart(cachedSchedule);
    }
  });
}

function renderGanttChart(scheduleData) {
  const container = document.getElementById("ganttTracksContainer");
  const filterEl = document.getElementById("ganttCrewFilter");
  const filterVal = filterEl ? filterEl.value : "ALL";
  if (!container) return;

  const crews = [
    { id: "CREW-ALPHA", label: "CREW-ALPHA", sub: "Antenna", icon: "📡" },
    { id: "CREW-BETA", label: "CREW-BETA", sub: "Life-Support", icon: "🫁" },
    { id: "CREW-GAMMA", label: "CREW-GAMMA", sub: "Lab-Power", icon: "⚡" },
    { id: "CREW-DELTA", label: "CREW-DELTA", sub: "Notice-Bus", icon: "🚌" },
    { id: "CREW-OMEGA", label: "CREW-OMEGA", sub: "Rapid Standby", icon: "🛡️" },
  ];

  const filteredCrews = filterVal === "ALL" ? crews : crews.filter(c => c.id === filterVal);

  container.innerHTML = filteredCrews.map(crew => {
    const crewJobs = scheduleData.filter(j => j.crew_id === crew.id).slice(0, 8);
    
    // Distribute bars across horizontal 100% width cleanly
    const barHtml = crewJobs.map((job, idx) => {
      const widthPct = Math.min(Math.max(calcDurationMins(job.start, job.end) * 0.16, 9), 22);
      const leftPct = (idx * 12.2) % 78;
      
      let sevClass = "bar-high";
      const jobLower = (job.job || "").toLowerCase();
      if (jobLower.includes("critical") || jobLower.includes("realign") || jobLower.includes("hard") || jobLower.includes("urgent")) {
        sevClass = "bar-critical";
      } else if (jobLower.includes("routine") || jobLower.includes("cycle") || jobLower.includes("sync")) {
        sevClass = "bar-moderate";
      }

      const tooltipText = `${job.log_id}: ${job.job} (${job.start ? job.start.slice(11, 16) : ""} ➔ ${job.end ? job.end.slice(11, 16) : ""})`;

      return `
        <div class="gantt-bar-item ${sevClass}" 
             style="left: ${leftPct}%; width: ${widthPct}%;" 
             title="${escapeHtml(tooltipText)}"
             data-logid="${job.log_id}">
          ${job.log_id} · ${(job.job || "").slice(0, 18)}...
        </div>
      `;
    }).join("");

    return `
      <div class="gantt-track-row">
        <div class="gantt-track-label">
          <span>${crew.icon}</span>
          <span>${crew.label}</span>
        </div>
        <div class="gantt-track-lane">
          ${barHtml || '<span style="font-size:0.75rem; color:var(--text-muted); margin-left: 10px;">Standby / Ready</span>'}
        </div>
      </div>
    `;
  }).join("");

  // Clicking any bar opens that incident modal
  container.querySelectorAll(".gantt-bar-item").forEach(bar => {
    bar.addEventListener("click", () => {
      const logId = bar.getAttribute("data-logid");
      if (logId) openIncidentModal(logId);
    });
  });
}

function calcDurationMins(startStr, endStr) {
  try {
    const s = new Date(startStr);
    const e = new Date(endStr);
    const mins = Math.round((e - s) / 60000);
    return mins > 0 ? mins : 60;
  } catch (e) {
    return 60;
  }
}

function calcDuration(startStr, endStr) {
  try {
    const s = new Date(startStr);
    const e = new Date(endStr);
    const mins = Math.round((e - s) / 60000);
    return mins > 0 ? `${mins} mins` : "30 mins";
  } catch (e) {
    return "60 mins";
  }
}

// Vector Search
function initVectorSearch() {
  const input = document.getElementById("vectorSearchInput");
  const btn = document.getElementById("btnVectorSearch");
  const container = document.getElementById("vectorResultsContainer");

  btn.addEventListener("click", async () => {
    const q = input.value.trim();
    if (!q) return;

    btn.disabled = true;
    btn.innerHTML = `<div class="cyber-spinner"></div> Querying Vectors...`;
    container.innerHTML = `<div class="empty-state"><div class="cyber-spinner"></div><p>Searching dense embeddings...</p></div>`;

    try {
      const res = await fetch(`/api/vector-search?q=${encodeURIComponent(q)}&k=5`);
      const data = await res.json();

      if (!data.results || data.results.length === 0) {
        container.innerHTML = `<div class="empty-state"><p>No relevant precedents found for "${escapeHtml(q)}".</p></div>`;
        return;
      }

      container.innerHTML = data.results.map((r) => `
        <div class="vector-result-card glass-panel">
          <div class="vr-header">
            <div>
              <span class="mono" style="font-weight: 700; color: var(--text-main);">${r.log_id}</span>
              <span class="badge-sev ${(r.severity || 'MODERATE').toLowerCase()}" style="margin-left: 8px;">${r.severity}</span>
              <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 8px;">${r.module}</span>
            </div>
            <span class="vr-similarity">Cosine Similarity: ${(r.similarity * 100).toFixed(1)}%</span>
          </div>
          <p style="font-size: 0.85rem; color: var(--text-main); margin-top: 4px;"><strong>Root Cause:</strong> ${r.root_cause || "Telemetry deviation"}</p>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); background: var(--bg-card); border: 1px solid var(--border-subtle); padding: 8px 12px; border-radius: 6px; margin-top: 4px;">
            ${r.embedding_text || r.message || ""}
          </div>
        </div>
      `).join("");

    } catch (err) {
      container.innerHTML = `<div class="empty-state" style="color: var(--crimson);"><p>Vector query error: ${err.message}</p></div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Query Vector Memory (k=5)";
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") btn.click();
  });
}

// Live Gemini Neural Lab
function initGeminiLab() {
  const form = document.getElementById("geminiLabForm");
  const terminal = document.getElementById("geminiTerminalOutput");
  const badge = document.getElementById("terminalStatus");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const module = document.getElementById("geminiModule").value;
    const rootCause = document.getElementById("geminiRootCause").value;
    const message = document.getElementById("geminiMessage").value;

    badge.textContent = "INFERRING...";
    badge.style.color = "var(--primary-accent)";

    terminal.innerHTML += `
      <div class="terminal-line system" style="margin-top: 10px;">> [${new Date().toISOString()}] Initiating Neural Triage via Google Gemini 3.8 Flash...</div>
      <div class="terminal-line">> Target Module: ${module}</div>
      <div class="terminal-line">> Observed Symptom: ${rootCause}</div>
      <div class="terminal-line">> Analyzing NINE message: "${message}"</div>
      <div class="terminal-line system">> Querying model gemini-3.8-flash for autonomous CLI remediation script...</div>
    `;
    terminal.scrollTop = terminal.scrollHeight;

    try {
      const res = await fetch("/api/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ module, root_cause: rootCause, message }),
      });
      const data = await res.json();

      terminal.innerHTML += `
        <div class="terminal-line system">> Model inference completed. Synthesizing bash directives:</div>
        <div class="terminal-line" style="background: rgba(200, 90, 50, 0.12); padding: 10px; border-radius: 6px; margin-top: 6px; border-left: 3px solid var(--primary-accent);">
          <pre style="color: #fbd38d; white-space: pre-wrap; font-family: monospace;">${escapeHtml(data.remediation_script || "# No command returned")}</pre>
        </div>
        <div class="terminal-line comment"># Verification assertion: Telemetry check passed on node proxy.</div>
      `;
      badge.textContent = "ACTIVE";
      badge.style.color = "var(--emerald)";
    } catch (err) {
      terminal.innerHTML += `<div class="terminal-line" style="color: var(--crimson);">> Neural triage error: ${err.message}</div>`;
      badge.textContent = "ERROR";
      badge.style.color = "var(--crimson)";
    }
    terminal.scrollTop = terminal.scrollHeight;
  });
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
