let allPermits = [];

function formatDisplayDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric"
  });
}

function formatLastUpdated(value) {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  });
}

function populateFilters(permits) {
  const usecodeFilter = document.getElementById("usecodeFilter");
  const worktypeFilter = document.getElementById("worktypeFilter");
  const statusFilter = document.getElementById("statusFilter");

  usecodeFilter.innerHTML = `<option value="ALL">All Use Codes</option>`;
  worktypeFilter.innerHTML = `<option value="ALL">All Work Types</option>`;
  statusFilter.innerHTML = `<option value="ALL">All Statuses</option>`;

  const usecodes = [...new Set(permits.map(p => p.usecode).filter(Boolean))].sort();
  const worktypes = [...new Set(permits.map(p => p.worktype).filter(Boolean))].sort();
  const statuses = [...new Set(permits.map(p => p.status).filter(Boolean))].sort();

  for (const usecode of usecodes) {
    const option = document.createElement("option");
    option.value = usecode;
    option.textContent = usecode;
    usecodeFilter.appendChild(option);
  }

  for (const worktype of worktypes) {
    const option = document.createElement("option");
    option.value = worktype;
    option.textContent = worktype;
    worktypeFilter.appendChild(option);
  }

  for (const status of statuses) {
    const option = document.createElement("option");
    option.value = status;
    option.textContent = status;
    statusFilter.appendChild(option);
  }
}

function applyFilters() {
  const usecodeValue = document.getElementById("usecodeFilter").value;
  const worktypeValue = document.getElementById("worktypeFilter").value;
  const statusValue = document.getElementById("statusFilter").value;
  const searchValue = document.getElementById("searchInput").value.trim().toLowerCase();
  const sortValue = document.getElementById("sortSelect").value;

  const valueMin = parseFloat(document.getElementById("valueMin").value);
  const valueMax = parseFloat(document.getElementById("valueMax").value);
  const areaMin = parseFloat(document.getElementById("areaMin").value);
  const areaMax = parseFloat(document.getElementById("areaMax").value);

  let filtered = allPermits.filter(permit => {
    const matchesUsecode = usecodeValue === "ALL" || permit.usecode === usecodeValue;
    const matchesWorktype = worktypeValue === "ALL" || permit.worktype === worktypeValue;
    const matchesStatus = statusValue === "ALL" || permit.status === statusValue;

    const haystack = [
      permit.location,
      permit.status,
      permit.usecode,
      permit.worktype,
      permit.description
    ].join(" ").toLowerCase();

    const matchesSearch = !searchValue || haystack.includes(searchValue);

    const valuation = permit.declaredvaluation_num || 0;
    const area = permit.buildingarea_num || 0;

    const matchesValueMin = Number.isNaN(valueMin) ? true : valuation >= valueMin;
    const matchesValueMax = Number.isNaN(valueMax) ? true : valuation <= valueMax;
    const matchesAreaMin = Number.isNaN(areaMin) ? true : area >= areaMin;
    const matchesAreaMax = Number.isNaN(areaMax) ? true : area <= areaMax;

    return (
      matchesUsecode &&
      matchesWorktype &&
      matchesStatus &&
      matchesSearch &&
      matchesValueMin &&
      matchesValueMax &&
      matchesAreaMin &&
      matchesAreaMax
    );
  });

  filtered.sort((a, b) => {
    if (sortValue === "oldest") {
      return (a.addeddate_raw || "").localeCompare(b.addeddate_raw || "");
    }

    if (sortValue === "value_desc") {
      return (b.declaredvaluation_num || 0) - (a.declaredvaluation_num || 0);
    }

    if (sortValue === "value_asc") {
      return (a.declaredvaluation_num || 0) - (b.declaredvaluation_num || 0);
    }

    if (sortValue === "area_desc") {
      return (b.buildingarea_num || 0) - (a.buildingarea_num || 0);
    }

    if (sortValue === "area_asc") {
      return (a.buildingarea_num || 0) - (b.buildingarea_num || 0);
    }

    return (b.addeddate_raw || "").localeCompare(a.addeddate_raw || "");
  });

  renderTable(filtered);
}

function renderTable(permits) {
  const tbody = document.getElementById("permitsTableBody");
  const recordCount = document.getElementById("recordCount");
  recordCount.textContent = permits.length.toLocaleString("en-US");

  if (!permits.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="10" class="empty-row">No permits match the current filters.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = permits.map(permit => `
    <tr>
      <td class="location-cell">${escapeHtml(permit.location || "—")}</td>
      <td><span class="status-pill">${escapeHtml(permit.status || "—")}</span></td>
      <td><span class="code-pill">${escapeHtml(permit.usecode || "—")}</span></td>
      <td>${escapeHtml(permit.declaredvaluation || "—")}</td>
      <td>${escapeHtml(permit.buildingarea || "—")}</td>
      <td>${escapeHtml(permit.worktype || "—")}</td>
      <td class="description-cell">${escapeHtml(permit.description || "—")}</td>
      <td>${escapeHtml(formatDisplayDate(permit.addeddate_raw))}</td>
      <td>${escapeHtml(formatDisplayDate(permit.issueddate_raw))}</td>
      <td>${escapeHtml(formatDisplayDate(permit.finaleddate_raw))}</td>
    </tr>
  `).join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadData() {
  try {
    const response = await fetch("./data/permits.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load permits.json (${response.status})`);
    }

    const data = await response.json();
    allPermits = data.permits || [];

    document.getElementById("lastUpdated").textContent = formatLastUpdated(data.last_updated);

    populateFilters(allPermits);
    applyFilters();
  } catch (error) {
    console.error(error);
    document.getElementById("lastUpdated").textContent = "Load error";
    document.getElementById("recordCount").textContent = "--";
    document.getElementById("permitsTableBody").innerHTML = `
      <tr>
        <td colspan="10" class="empty-row">
          Could not load permit data. Run the Python update script first.
        </td>
      </tr>
    `;
  }
}

document.getElementById("usecodeFilter").addEventListener("change", applyFilters);
document.getElementById("worktypeFilter").addEventListener("change", applyFilters);
document.getElementById("statusFilter").addEventListener("change", applyFilters);
document.getElementById("searchInput").addEventListener("input", applyFilters);
document.getElementById("valueMin").addEventListener("input", applyFilters);
document.getElementById("valueMax").addEventListener("input", applyFilters);
document.getElementById("areaMin").addEventListener("input", applyFilters);
document.getElementById("areaMax").addEventListener("input", applyFilters);
document.getElementById("sortSelect").addEventListener("change", applyFilters);

document.getElementById("resetBtn").addEventListener("click", () => {
  document.getElementById("usecodeFilter").value = "ALL";
  document.getElementById("worktypeFilter").value = "ALL";
  document.getElementById("statusFilter").value = "ALL";
  document.getElementById("searchInput").value = "";
  document.getElementById("valueMin").value = "";
  document.getElementById("valueMax").value = "";
  document.getElementById("areaMin").value = "";
  document.getElementById("areaMax").value = "";
  document.getElementById("sortSelect").value = "newest";
  applyFilters();
});

loadData();