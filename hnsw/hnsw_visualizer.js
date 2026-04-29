"use strict";

const els = {
  canvas: document.getElementById("graphCanvas"),
  resetBtn: document.getElementById("resetBtn"),
  randomBtn: document.getElementById("randomBtn"),
  buildBtn: document.getElementById("buildBtn"),
  scenarioBtn: document.getElementById("scenarioBtn"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  playBtn: document.getElementById("playBtn"),
  slider: document.getElementById("stepSlider"),
  stepNumber: document.getElementById("stepNumber"),
  stepTotal: document.getElementById("stepTotal"),
  mInput: document.getElementById("mInput"),
  mmaxInput: document.getElementById("mmaxInput"),
  efConstructionInput: document.getElementById("efConstructionInput"),
  efSearchInput: document.getElementById("efSearchInput"),
  kInput: document.getElementById("kInput"),
  layerSelect: document.getElementById("layerSelect"),
  stepTitle: document.getElementById("stepTitle"),
  stepDetail: document.getElementById("stepDetail"),
  storyAlgorithm: document.getElementById("storyAlgorithm"),
  storyTitle: document.getElementById("storyTitle"),
  storyDetail: document.getElementById("storyDetail"),
  storyMove: document.getElementById("storyMove"),
  storyFocus: document.getElementById("storyFocus"),
  storyHelp: document.getElementById("storyHelp"),
  storyParams: document.getElementById("storyParams"),
  algorithmLabel: document.getElementById("algorithmLabel"),
  layerLabel: document.getElementById("layerLabel"),
  entryLabel: document.getElementById("entryLabel"),
  currentLabel: document.getElementById("currentLabel"),
  candidateQueue: document.getElementById("candidateQueue"),
  foundQueue: document.getElementById("foundQueue"),
  removedQueue: document.getElementById("removedQueue"),
  selectedQueue: document.getElementById("selectedQueue"),
  prunedQueue: document.getElementById("prunedQueue"),
  tabs: Array.from(document.querySelectorAll(".tab")),
};

const ctx = els.canvas.getContext("2d");
const colors = {
  edge: "#b9c3cf",
  edgeActive: "#45515f",
  node: "#697786",
  nodeStroke: "#ffffff",
  query: "#111827",
  entry: "#1f6feb",
  candidate: "#c26b18",
  found: "#7c3aed",
  selected: "#18864b",
  pruned: "#c94646",
  removed: "#d94d2b",
  visited: "#8c98a5",
  text: "#17202a",
  muted: "#5e6b78",
  layer: "#e6ecf2",
  exact: "#0f7b6c",
};

const basePoints = [
  [11, 18], [17, 34], [24, 16], [30, 44], [36, 26], [42, 62],
  [47, 37], [54, 18], [59, 54], [66, 32], [71, 72], [78, 46],
  [84, 20], [88, 64], [18, 72], [34, 82], [52, 82], [72, 88],
];

let app = {
  mode: "insert",
  points: [],
  graph: createEmptyGraph(),
  entryId: null,
  frames: [],
  step: 0,
  playing: false,
  timer: null,
  animationHandle: null,
  lastRenderedFrame: null,
  seed: 7,
  activeLayer: 0,
};

function createEmptyGraph() {
  return { nodes: new Map(), links: new Map(), maxLayer: 0 };
}

function readParams() {
  return {
    M: clampInt(els.mInput.value, 1, 8),
    Mmax: clampInt(els.mmaxInput.value, 2, 12),
    efConstruction: clampInt(els.efConstructionInput.value, 1, 12),
    efSearch: clampInt(els.efSearchInput.value, 1, 12),
    K: clampInt(els.kInput.value, 1, 8),
  };
}

function clampInt(value, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (Number.isNaN(parsed)) return min;
  return Math.max(min, Math.min(max, parsed));
}

function makeRng(seed) {
  let state = seed >>> 0;
  return function random() {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function nodeLabel(id) {
  if (id === null || id === undefined) return "-";
  return `v${id}`;
}

function queryLabel(q) {
  if (!q) return "q";
  return q.id === "q" ? "q/query" : `${nodeLabel(q.id)} / vector baru`;
}

function asLabels(ids) {
  if (!ids || ids.length === 0) return "-";
  return ids.map(nodeLabel).join(", ");
}

function createPoints(randomize = false) {
  const rng = makeRng(randomize ? Date.now() % 100000 : 22);
  return basePoints.map(([x, y], i) => {
    const jitter = randomize ? 6 : 0;
    const jx = (rng() - 0.5) * jitter;
    const jy = (rng() - 0.5) * jitter;
    const plannedLevel = i === 0 ? 2 : i === 3 || i === 10 ? 1 : rng() < 0.18 ? 1 : 0;
    return {
      id: i,
      x: clampNumber(x + jx, 7, 93),
      y: clampNumber(y + jy, 9, 91),
      level: plannedLevel,
      inserted: false,
    };
  });
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resetDataset(randomize = false) {
  stopPlayback();
  cancelAnimation();
  app.points = createPoints(randomize);
  app.graph = createEmptyGraph();
  app.entryId = null;
  app.frames = [];
  app.step = 0;
  app.activeLayer = 0;
  app.lastRenderedFrame = null;
  updateLayerSelect(0);
  pushFrame({
    title: "Dataset siap",
    detail: "Setiap titik adalah vector 2D. Layer atas berisi shortcut jarak jauh; layer 0 adalah graph neighbor yang paling padat.",
    algorithm: "-",
    layer: 0,
  });
  renderCurrentFrame();
}

function cloneGraph(graph) {
  const nodes = Array.from(graph.nodes.values()).map((n) => ({ ...n }));
  const links = {};
  graph.links.forEach((byLayer, id) => {
    links[id] = {};
    byLayer.forEach((set, layer) => {
      links[id][layer] = Array.from(set);
    });
  });
  return {
    nodes,
    links,
    maxLayer: graph.maxLayer,
  };
}

function getNode(graph, id) {
  return graph.nodes.get(id);
}

function ensureLinkMap(graph, id) {
  if (!graph.links.has(id)) graph.links.set(id, new Map());
  return graph.links.get(id);
}

function ensureLayerSet(graph, id, layer) {
  const byLayer = ensureLinkMap(graph, id);
  if (!byLayer.has(layer)) byLayer.set(layer, new Set());
  return byLayer.get(layer);
}

function neighborsOf(graph, id, layer) {
  const byLayer = graph.links.get(id);
  if (!byLayer) return [];
  const set = byLayer.get(layer);
  if (!set) return [];
  return Array.from(set).filter((nid) => graph.nodes.has(nid));
}

function addBidirectional(graph, a, b, layer) {
  if (a === b) return;
  ensureLayerSet(graph, a, layer).add(b);
  ensureLayerSet(graph, b, layer).add(a);
}

function setNeighbors(graph, id, layer, neighborIds) {
  ensureLayerSet(graph, id, layer).clear();
  neighborIds.forEach((nid) => ensureLayerSet(graph, id, layer).add(nid));

  graph.nodes.forEach((_, otherId) => {
    if (otherId === id) return;
    const set = ensureLayerSet(graph, otherId, layer);
    if (set.has(id) && !neighborIds.includes(otherId)) set.delete(id);
  });

  neighborIds.forEach((nid) => ensureLayerSet(graph, nid, layer).add(id));
}

function sortedByDistance(graph, ids, q) {
  return unique(ids)
    .filter((id) => graph.nodes.has(id))
    .sort((a, b) => distance(getNode(graph, a), q) - distance(getNode(graph, b), q));
}

function unique(ids) {
  return Array.from(new Set(ids));
}

function pushFrame(extra) {
  const frame = {
    title: "Step",
    detail: "",
    algorithm: "-",
    layer: app.activeLayer,
    entryId: app.entryId,
    currentId: null,
    query: null,
    candidates: [],
    found: [],
    removed: [],
    selected: [],
    pruned: [],
    visited: [],
    exact: [],
    activeEdges: [],
    graph: cloneGraph(app.graph),
    ...extra,
  };
  app.frames.push(frame);
}

function buildBaseGraph() {
  stopPlayback();
  cancelAnimation();
  app.graph = createEmptyGraph();
  app.entryId = null;
  app.frames = [];
  app.step = 0;
  app.lastRenderedFrame = null;
  const params = readParams();
  const buildCount = app.mode === "insert" ? app.points.length - 1 : app.points.length;

  pushFrame({
    title: "Algorithm 1: mulai indexing",
    detail: "Base graph dibuat dengan memasukkan vector satu per satu. Ini gambaran proses saat vector di-upload ke collection yang memakai HNSW.",
    algorithm: "Algorithm 1",
    layer: 0,
  });

  for (let i = 0; i < buildCount; i += 1) {
    insertPoint(app.points[i], params, { record: false });
  }

  updateLayerSelect(app.graph.maxLayer);
  app.activeLayer = Math.min(app.activeLayer, app.graph.maxLayer);
  pushFrame({
    title: "Base graph selesai",
    detail: app.mode === "insert"
      ? "Satu vector sengaja belum dimasukkan. Prepare Scenario akan memasukkannya step by step."
      : "Semua vector sudah ter-index. Prepare Scenario akan menjalankan Algorithm 5 dengan satu query vector.",
    algorithm: "Algorithm 1",
    layer: app.activeLayer,
  });
  app.step = app.frames.length - 1;
  renderCurrentFrame();
}

function prepareScenario() {
  stopPlayback();
  cancelAnimation();
  if (app.graph.nodes.size === 0) buildBaseGraph();
  app.frames = [];
  app.step = 0;
  app.lastRenderedFrame = null;
  const params = readParams();
  if (app.mode === "insert") {
    let nextPoint = app.points.find((p) => !app.graph.nodes.has(p.id));
    if (!nextPoint) {
      buildBaseGraph();
      app.frames = [];
      app.step = 0;
      nextPoint = app.points.find((p) => !app.graph.nodes.has(p.id));
    }
    insertPoint(nextPoint, params, { record: true });
  } else {
    runKnnSearch({ id: "q", x: 63, y: 59 }, params);
  }
  updateLayerSelect(app.graph.maxLayer);
  renderCurrentFrame();
}

function insertPoint(point, params, options = { record: true }) {
  const record = Boolean(options.record);
  const q = { ...point };

  if (app.entryId === null) {
    app.graph.nodes.set(q.id, { ...q, inserted: true });
    app.graph.maxLayer = Math.max(app.graph.maxLayer, q.level);
    app.entryId = q.id;
    if (record) {
      pushFrame({
        title: "Algorithm 1: vector pertama menjadi entry point",
        detail: `${nodeLabel(q.id)} belum punya graph untuk dicari, jadi node ini langsung menjadi entry point.`,
        algorithm: "Algorithm 1",
        layer: q.level,
        query: q,
        entryId: q.id,
      });
    }
    return;
  }

  let ep = [app.entryId];
  const topLayer = getNode(app.graph, app.entryId).level;
  const newLevel = q.level;

  if (record) {
    pushFrame({
      title: "Algorithm 1: pilih level vector baru",
      detail: `${nodeLabel(q.id)} mendapat level ${newLevel}. Di Qdrant, level juga dipilih secara random: kebanyakan vector hanya di layer 0, sebagian kecil naik menjadi shortcut di layer atas.`,
      algorithm: "Algorithm 1",
      layer: topLayer,
      query: q,
      entryId: app.entryId,
    });
  }

  for (let lc = topLayer; lc > newLevel; lc -= 1) {
    const result = searchLayer(app.graph, q, ep, 1, lc, record ? {
      prefix: "Algorithm 1 phase 1",
      query: q,
      note: "Layer atas memakai ef=1, jadi ini greedy routing murni untuk mencari entry point yang lebih dekat.",
    } : null);
    ep = result.found.slice(0, 1);
    if (record) {
      pushFrame({
        title: "Algorithm 1: entry point turun layer",
        detail: `Di layer ${lc}, greedy search menemukan ${asLabels(ep)} sebagai titik masuk yang lebih dekat untuk layer berikutnya.`,
        algorithm: "Algorithm 1 + 2",
        layer: lc,
        query: q,
        entryId: ep[0],
        found: ep,
      });
    }
  }

  app.graph.nodes.set(q.id, { ...q, inserted: true });
  app.graph.maxLayer = Math.max(app.graph.maxLayer, newLevel);
  for (let layer = 0; layer <= newLevel; layer += 1) ensureLayerSet(app.graph, q.id, layer);

  for (let lc = Math.min(topLayer, newLevel); lc >= 0; lc -= 1) {
    const result = searchLayer(app.graph, q, ep, params.efConstruction, lc, record ? {
      prefix: "Algorithm 1 phase 2",
      query: q,
      note: "Pada layer insertion, efConstruction memperlebar candidate set sebelum link graph dipilih.",
    } : null);
    const candidates = result.found;
    const select = selectNeighborsHeuristic(app.graph, q, candidates, params.M, lc, {
      extendCandidates: false,
      keepPrunedConnections: true,
      record,
      baseLabel: nodeLabel(q.id),
    });
    select.selected.forEach((nid) => addBidirectional(app.graph, q.id, nid, lc));

    if (record) {
      pushFrame({
        title: "Algorithm 1: tambah link dua arah",
        detail: `${nodeLabel(q.id)} terhubung ke ${asLabels(select.selected)} di layer ${lc}. Karena link HNSW bersifat dua arah, semua node itu juga mendapat link balik ke ${nodeLabel(q.id)}. Setelah ini, HNSW mengecek satu per satu apakah link mereka melebihi Mmax=${params.Mmax}.`,
        algorithm: "Algorithm 1 + 4",
        layer: lc,
        query: q,
        currentId: q.id,
        selected: select.selected,
        pruned: select.pruned,
        activeEdges: select.selected.map((nid) => [q.id, nid]),
      });
    }

    for (const neighborId of select.selected) {
      const current = neighborsOf(app.graph, neighborId, lc);
      if (current.length > params.Mmax) {
        const currentText = asLabels(current);
        if (record) {
          pushFrame({
            title: "Algorithm 1: cek neighbor lama yang ikut berubah",
            detail: `${nodeLabel(q.id)} sudah selesai memilih link. Sekarang giliran mengecek ${nodeLabel(neighborId)}, salah satu neighbor lama yang mendapat link balik. Link ${nodeLabel(neighborId)} sekarang: ${currentText} (${current.length} link). Karena ${current.length} > Mmax=${params.Mmax}, ${nodeLabel(neighborId)} harus masuk Algorithm 4 untuk pruning. Kalau neighbor lain juga > Mmax, mereka akan diproses seperti ini juga.`,
            algorithm: "Algorithm 1 + 4",
            layer: lc,
            query: q,
            currentId: neighborId,
            selected: [neighborId],
            activeEdges: [[q.id, neighborId]],
          });
        }
        const shrink = selectNeighborsHeuristic(app.graph, getNode(app.graph, neighborId), current, params.Mmax, lc, {
          extendCandidates: false,
          keepPrunedConnections: true,
          record,
          baseLabel: nodeLabel(neighborId),
          mode: "shrink-existing",
          insertedLabel: nodeLabel(q.id),
        });
        setNeighbors(app.graph, neighborId, lc, shrink.selected);
        if (record) {
          pushFrame({
            title: "Algorithm 1: pangkas neighbor yang penuh",
            detail: `${nodeLabel(neighborId)} punya ${current.length} link di layer ${lc}, melewati Mmax=${params.Mmax}. Setelah ${nodeLabel(q.id)} ditambahkan, Algorithm 4 memilih ulang maksimal ${params.Mmax} link terbaik untuk ${nodeLabel(neighborId)} dan membuang link yang redundant.`,
            algorithm: "Algorithm 1 + 4",
            layer: lc,
            query: getNode(app.graph, neighborId),
            currentId: neighborId,
            selected: shrink.selected,
            pruned: shrink.pruned,
          });
        }
      }
    }
    ep = candidates.length ? candidates : ep;
  }

  if (newLevel > topLayer) {
    app.entryId = q.id;
    if (record) {
      pushFrame({
        title: "Algorithm 1: update entry point",
        detail: `${nodeLabel(q.id)} memiliki level tertinggi, sehingga node ini menjadi entry point graph.`,
        algorithm: "Algorithm 1",
        layer: newLevel,
        query: q,
        entryId: q.id,
      });
    }
  }
}

function searchLayer(graph, q, entryPoints, ef, layer, recordOptions = null) {
  const visited = new Set(entryPoints);
  const cameFrom = new Map(entryPoints.map((id) => [id, null]));
  let candidates = sortedByDistance(graph, entryPoints, q);
  let found = sortedByDistance(graph, entryPoints, q).slice(0, ef);
  const prefix = recordOptions?.prefix || "Algorithm 2";

  if (recordOptions) {
    pushFrame({
      title: `${prefix}: initialize SEARCH-LAYER`,
      detail: `Mulai dari ${asLabels(entryPoints)} dengan ef=${ef}. Semua perbandingan "dekat" di sini selalu diukur ke ${queryLabel(q)}, bukan ke node yang sedang dibuka.`,
      algorithm: "Algorithm 2",
      layer,
      query: recordOptions.query || q,
      entryId: entryPoints[0],
      candidates,
      found,
      visited: Array.from(visited),
    });
  }

  while (candidates.length > 0) {
    candidates = sortedByDistance(graph, candidates, q);
    found = sortedByDistance(graph, found, q).slice(0, ef);
    const current = candidates.shift();
    const parent = cameFrom.get(current);
    const furthest = sortedByDistance(graph, found, q).at(-1);
    const currentDistance = distance(getNode(graph, current), q);
    const furthestDistance = furthest === undefined ? Infinity : distance(getNode(graph, furthest), q);

    if (recordOptions) {
      pushFrame({
        title: `${prefix}: extract nearest candidate`,
        detail: `${nodeLabel(current)} adalah item terdekat di C terhadap ${queryLabel(q)}. Bandingkan distance(${nodeLabel(current)}, ${queryLabel(q)})=${currentDistance.toFixed(1)} dengan batas terjauh di W=${furthestDistance.toFixed(1)}.`,
        algorithm: "Algorithm 2",
        layer,
        query: recordOptions.query || q,
        entryId: entryPoints[0],
        currentId: current,
        candidates,
        found,
        visited: Array.from(visited),
        activeEdges: parent === null || parent === undefined ? [] : [[parent, current]],
      });
    }

    if (currentDistance > furthestDistance) {
      if (recordOptions) {
        pushFrame({
          title: `${prefix}: stop condition`,
          detail: `${nodeLabel(current)} lebih jauh dari ${queryLabel(q)} dibanding item terjauh di W, jadi Algorithm 2 berhenti scan layer ini. ${recordOptions.note || ""}`.trim(),
          algorithm: "Algorithm 2",
          layer,
          query: recordOptions.query || q,
          currentId: current,
          candidates,
          found,
          visited: Array.from(visited),
        });
      }
      break;
    }

    const neighbors = neighborsOf(graph, current, layer);
    for (const nid of neighbors) {
      if (visited.has(nid)) continue;
      visited.add(nid);
      const worst = sortedByDistance(graph, found, q).at(-1);
      const neighborDistance = distance(getNode(graph, nid), q);
      const boundaryDistance = worst === undefined ? Infinity : distance(getNode(graph, worst), q);
      const shouldAdd = found.length < ef || neighborDistance < boundaryDistance;
      const beforeFound = found.slice();
      let removedFromW = [];
      if (shouldAdd) {
        cameFrom.set(nid, current);
        candidates.push(nid);
        found.push(nid);
        found = sortedByDistance(graph, unique(found), q).slice(0, ef);
        removedFromW = beforeFound.filter((id) => !found.includes(id));
      }
      if (recordOptions) {
        const beforeText = asLabels(beforeFound);
        const afterText = asLabels(found);
        const removedText = asLabels(removedFromW);
        pushFrame({
          title: shouldAdd ? `${prefix}: neighbor masuk C dan W` : `${prefix}: neighbor hanya ditandai visited`,
          detail: shouldAdd
            ? `${nodeLabel(nid)} diterima karena distance(${nodeLabel(nid)}, ${queryLabel(q)})=${neighborDistance.toFixed(1)} lebih kecil dari batas W=${boundaryDistance.toFixed(1)}, atau W belum penuh. W sebelum: ${beforeText}. W sesudah: ${afterText}. ${removedFromW.length ? `Dibuang dari W: ${removedText}.` : "Belum ada yang dibuang dari W."}`
            : `${nodeLabel(nid)} dikunjungi dari ${nodeLabel(current)}, tetapi distance(${nodeLabel(nid)}, ${queryLabel(q)})=${neighborDistance.toFixed(1)} tidak lebih kecil dari batas W=${boundaryDistance.toFixed(1)}.`,
          algorithm: "Algorithm 2",
          layer,
          query: recordOptions.query || q,
          currentId: current,
          candidates: sortedByDistance(graph, unique(candidates), q),
          found,
          removed: removedFromW,
          visited: Array.from(visited),
          activeEdges: [[current, nid]],
        });
      }
    }
    candidates = sortedByDistance(graph, unique(candidates), q);
  }

  return { found: sortedByDistance(graph, found, q).slice(0, ef), visited: Array.from(visited) };
}

function selectNeighborsHeuristic(graph, q, candidateIds, M, layer, options) {
  const extendCandidates = Boolean(options.extendCandidates);
  const keepPrunedConnections = Boolean(options.keepPrunedConnections);
  let working = unique(candidateIds).filter((id) => graph.nodes.has(id) && id !== q.id);
  const selected = [];
  const pruned = [];
  const baseName = options.baseLabel || "q";
  const isShrink = options.mode === "shrink-existing";
  const purpose = isShrink
    ? `${baseName} adalah node lama yang baru menerima link dari ${options.insertedLabel || "node baru"}. Candidate list ini adalah daftar link milik ${baseName} yang harus dipilih ulang agar tidak melebihi Mmax.`
    : `${baseName} adalah vector baru/query yang sedang memilih neighbor untuk dibuat link.`;

  if (options.record) {
    pushFrame({
      title: "Algorithm 4: mulai RNG pruning",
      detail: `${purpose} Candidate list untuk ${baseName}: ${asLabels(working)}. Heuristic ini mencari link yang arahnya beragam, bukan sekadar titik paling dekat.`,
      algorithm: "Algorithm 4",
      layer,
      query: q,
      currentId: q.id,
      candidates: sortedByDistance(graph, working, q),
    });
  }

  if (extendCandidates) {
    for (const id of candidateIds) {
      for (const adj of neighborsOf(graph, id, layer)) {
        if (!working.includes(adj) && adj !== q.id) working.push(adj);
      }
    }
  }

  while (working.length > 0 && selected.length < M) {
    working = sortedByDistance(graph, working, q);
    const candidate = working.shift();
    const candidateNode = getNode(graph, candidate);
    const blockedBy = selected.find((sid) => distance(candidateNode, getNode(graph, sid)) < distance(candidateNode, q));
    if (blockedBy === undefined) {
      selected.push(candidate);
      if (options.record) {
        pushFrame({
          title: "Algorithm 4: simpan neighbor yang beragam",
          detail: `${nodeLabel(candidate)} disimpan sebagai link untuk ${baseName}. Alasannya: ${nodeLabel(candidate)} lebih dekat ke ${baseName} daripada ke semua neighbor yang sudah dipilih, jadi arahnya masih berguna dan tidak redundant.`,
          algorithm: "Algorithm 4",
          layer,
          query: q,
          currentId: candidate,
          candidates: working,
          selected: selected.slice(),
          pruned: pruned.slice(),
        });
      }
    } else {
      pruned.push(candidate);
      if (options.record) {
        pushFrame({
          title: "Algorithm 4: pangkas neighbor redundant",
          detail: `${nodeLabel(candidate)} dipangkas dari daftar link ${baseName} karena lebih dekat ke ${nodeLabel(blockedBy)} daripada ke ${baseName}. Artinya ${nodeLabel(candidate)} bisa dicapai lewat ${nodeLabel(blockedBy)}, sehingga link langsungnya dianggap redundant.`,
          algorithm: "Algorithm 4",
          layer,
          query: q,
          currentId: candidate,
          candidates: working,
          selected: selected.slice(),
          pruned: pruned.slice(),
          activeEdges: [[candidate, blockedBy]],
        });
      }
    }
  }

  if (keepPrunedConnections) {
    while (pruned.length > 0 && selected.length < M) {
      const backfill = sortedByDistance(graph, pruned, q)[0];
      pruned.splice(pruned.indexOf(backfill), 1);
      selected.push(backfill);
      if (options.record) {
        pushFrame({
          title: "Algorithm 4: isi slot dari Wd",
          detail: `${nodeLabel(backfill)} ditambahkan kembali ke link ${baseName} dari Wd karena keepPrunedConnections aktif dan R masih punya slot kosong.`,
          algorithm: "Algorithm 4",
          layer,
          query: q,
          currentId: backfill,
          selected: selected.slice(),
          pruned: pruned.slice(),
        });
      }
    }
  }

  return { selected, pruned };
}

function runKnnSearch(query, params) {
  const exact = Array.from(app.graph.nodes.keys())
    .sort((a, b) => distance(getNode(app.graph, a), query) - distance(getNode(app.graph, b), query))
    .slice(0, params.K);
  let ep = [app.entryId];
  const topLayer = getNode(app.graph, app.entryId).level;

  pushFrame({
    title: "Algorithm 5: mulai K-NN search",
    detail: `Mulai dari entry point ${nodeLabel(app.entryId)} di top layer ${topLayer}. Layer atas memakai ef=1 agar routing cepat mendekati query.`,
    algorithm: "Algorithm 5",
    layer: topLayer,
    query,
    entryId: app.entryId,
    exact,
  });

  for (let lc = topLayer; lc >= 1; lc -= 1) {
    const result = searchLayer(app.graph, query, ep, 1, lc, {
      prefix: "Algorithm 5 upper-layer routing",
      query,
      note: "Hasil terdekat menjadi entry point untuk layer berikutnya.",
    });
    ep = result.found.slice(0, 1);
    pushFrame({
      title: "Algorithm 5: turun satu layer",
      detail: `${asLabels(ep)} menjadi entry point setelah search di layer ${lc}.`,
      algorithm: "Algorithm 5 + 2",
      layer: lc,
      query,
      entryId: ep[0],
      found: ep,
      exact,
    });
  }

  const base = searchLayer(app.graph, query, ep, params.efSearch, 0, {
    prefix: "Algorithm 5 base-layer search",
    query,
    note: "Layer 0 memakai efSearch untuk mengumpulkan candidate list yang lebih lebar sebelum mengambil K hasil.",
  });
  const result = sortedByDistance(app.graph, base.found, query).slice(0, params.K);
  pushFrame({
    title: "Algorithm 5: return K nearest",
    detail: `Ambil K=${params.K} terbaik dari W: ${asLabels(result)}. Exact nearest pada dataset kecil ini adalah ${asLabels(exact)}; ef yang lebih besar biasanya menaikkan recall.`,
    algorithm: "Algorithm 5",
    layer: 0,
    query,
    found: base.found,
    selected: result,
    exact,
  });
}

function updateLayerSelect(maxLayer) {
  els.layerSelect.innerHTML = "";
  for (let layer = maxLayer; layer >= 0; layer -= 1) {
    const option = document.createElement("option");
    option.value = String(layer);
    option.textContent = `Layer ${layer}`;
    els.layerSelect.append(option);
  }
  els.layerSelect.value = String(Math.min(app.activeLayer, maxLayer));
}

function setMode(mode) {
  app.mode = mode;
  els.tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.mode === mode));
  buildBaseGraph();
}

function renderCurrentFrame(options = {}) {
  if (app.frames.length === 0) return;
  app.step = Math.max(0, Math.min(app.step, app.frames.length - 1));
  const frame = app.frames[app.step];
  if (frame.layer !== undefined) {
    app.activeLayer = Math.max(0, Math.min(frame.layer, frame.graph.maxLayer || 0));
    if (els.layerSelect.value !== String(app.activeLayer)) els.layerSelect.value = String(app.activeLayer);
  }
  updateText(frame);
  const fromFrame = options.animate ? app.lastRenderedFrame : null;
  app.lastRenderedFrame = frame;
  animateDraw(fromFrame, frame, Boolean(options.animate && fromFrame));
}

function updateText(frame) {
  els.slider.max = String(Math.max(0, app.frames.length - 1));
  els.slider.value = String(app.step);
  els.stepNumber.textContent = String(app.step + 1);
  els.stepTotal.textContent = String(app.frames.length);
  els.stepTitle.textContent = frame.title;
  els.stepDetail.textContent = frame.detail;
  els.algorithmLabel.textContent = frame.algorithm || "-";
  els.layerLabel.textContent = frame.layer ?? "-";
  els.entryLabel.textContent = nodeLabel(frame.entryId);
  els.currentLabel.textContent = nodeLabel(frame.currentId);
  els.candidateQueue.textContent = asLabels(frame.candidates);
  els.foundQueue.textContent = asLabels(frame.found);
  els.removedQueue.textContent = asLabels(frame.removed);
  els.selectedQueue.textContent = asLabels(frame.selected);
  els.prunedQueue.textContent = asLabels(frame.pruned);
  els.storyAlgorithm.textContent = frame.algorithm || "Step";
  els.storyTitle.textContent = frame.title;
  els.storyDetail.textContent = frame.detail;
  els.storyMove.textContent = movementText(frame);
  els.storyFocus.textContent = focusText(frame);
  els.storyHelp.innerHTML = helperHtml(frame);
  els.storyParams.innerHTML = paramsHtml(frame);
}

function paramsHtml(frame) {
  const params = readParams();
  const algorithm = frame.algorithm || "";
  const activeEf = activeEfLabel(frame, params);

  if (algorithm.includes("5")) {
    return `
      <strong>Parameter yang dipakai</strong>
      <div class="param-grid">
        <span><b>K=${params.K}</b><br>Jumlah hasil akhir yang ingin diambil.</span>
        <span><b>efSearch=${params.efSearch}</b><br>Lebar W saat query search. Lebih besar biasanya lebih akurat, tapi lebih lambat.</span>
        <span><b>Active layer=${app.activeLayer}</b><br>Layer yang sedang divisualisasikan sekarang.</span>
        <span><b>ef aktif=${activeEf}</b><br>Di layer atas biasanya 1; di layer 0 memakai efSearch.</span>
      </div>
    `;
  }

  if (algorithm.includes("4")) {
    return `
      <strong>Parameter yang dipakai</strong>
      <div class="param-grid">
        <span><b>M=${params.M}</b><br>Jumlah neighbor/link yang ingin dipilih untuk node baru.</span>
        <span><b>Mmax=${params.Mmax}</b><br>Batas maksimal link per node. Jika lewat, link dipangkas.</span>
        <span><b>Active layer=${app.activeLayer}</b><br>Layer tempat link sedang dipilih/dipangkas.</span>
        <span><b>efConstruction=${params.efConstruction}</b><br>Lebar pencarian saat indexing untuk mencari calon neighbor.</span>
      </div>
    `;
  }

  if (algorithm.includes("2")) {
    return `
      <strong>Parameter yang dipakai</strong>
      <div class="param-grid">
        <span><b>ef aktif=${activeEf}</b><br>Ukuran maksimal W pada step ini.</span>
        <span><b>Active layer=${app.activeLayer}</b><br>Layer graph yang sedang dicari.</span>
        <span><b>efConstruction=${params.efConstruction}</b><br>Dipakai saat indexing agar kandidat neighbor lebih luas.</span>
        <span><b>efSearch=${params.efSearch}</b><br>Dipakai saat query search di layer 0.</span>
      </div>
    `;
  }

  return `
    <strong>Parameter yang dipakai</strong>
    <div class="param-grid">
      <span><b>M=${params.M}</b><br>Jumlah neighbor/link yang dipilih.</span>
      <span><b>Mmax=${params.Mmax}</b><br>Batas maksimal link per node.</span>
      <span><b>K=${params.K}</b><br>Jumlah hasil nearest neighbor.</span>
      <span><b>Layer=${app.activeLayer}</b><br>Layer yang sedang dilihat.</span>
    </div>
  `;
}

function activeEfLabel(frame, params) {
  const title = frame.title || "";
  if (title.includes("phase 1") || title.includes("upper-layer")) return "1";
  if (title.includes("phase 2")) return String(params.efConstruction);
  if (title.includes("base-layer") || frame.algorithm?.includes("5")) return String(params.efSearch);
  return frame.found?.length ? `maks ${Math.max(frame.found.length, params.efConstruction)}` : "-";
}

function helperHtml(frame) {
  const algorithm = frame.algorithm || "";
  if (algorithm.includes("2")) {
    return `
      <strong>Penjelasan istilah step ini</strong>
      <ul>
        <li><b>C / candidate queue</b>: daftar node yang sudah ditemukan dan masih mungkin di-expand. Algorithm mengambil node terdekat dari C untuk dicek berikutnya.</li>
        <li><b>W / nearest list</b>: daftar node terbaik sementara yang paling dekat ke <b>${escapeHtml(queryLabel(frame.query))}</b>. Kalau W lebih besar dari <b>ef</b>, node paling jauh dari query dibuang.</li>
        <li><b>Visited</b>: node yang sudah pernah dilihat, supaya tidak dihitung berulang.</li>
      </ul>
    `;
  }
  if (algorithm.includes("4")) {
    return `
      <strong>Penjelasan istilah step ini</strong>
      <ul>
        <li><b>R</b>: neighbor yang dipilih untuk menjadi link graph.</li>
        <li><b>Wd</b>: kandidat yang dipangkas karena arahnya dianggap redundant.</li>
        <li><b>RNG pruning</b>: kandidat disimpan kalau memberi arah baru; dipangkas kalau lebih dekat ke neighbor terpilih daripada ke node baru/query.</li>
        <li><b>Saat pindah ke node lama</b>: HNSW sedang merapikan ulang link node lama yang baru menerima link dua arah. Ini terjadi kalau jumlah link node itu <b>&gt; Mmax</b>.</li>
        <li><b>Berlaku untuk semua neighbor</b>: setiap node yang baru terhubung ke node baru akan dicek. Yang tidak melebihi Mmax dilewati; yang melebihi Mmax dipruning seperti contoh ini.</li>
      </ul>
    `;
  }
  if (algorithm.includes("5")) {
    return `
      <strong>Penjelasan istilah step ini</strong>
      <ul>
        <li><b>Top layer</b>: dipakai untuk lompat cepat mendekati area query.</li>
        <li><b>Layer 0</b>: layer paling padat, tempat final search mengambil K nearest.</li>
        <li><b>efSearch</b>: ukuran W saat search; makin besar biasanya recall makin baik, tetapi lebih lambat.</li>
      </ul>
    `;
  }
  return `
    <strong>Penjelasan istilah</strong>
    <ul>
      <li><b>Entry point</b>: node awal tempat HNSW mulai berjalan.</li>
      <li><b>Layer atas</b>: shortcut jarak jauh. <b>Layer 0</b>: graph paling padat untuk hasil akhir.</li>
    </ul>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function movementText(frame) {
  if (frame.activeEdges?.length) {
    return `Gerak: ${frame.activeEdges.map(([a, b]) => `${nodeLabel(a)} -> ${nodeLabel(b)}`).join(", ")}`;
  }
  if (frame.currentId !== null && frame.currentId !== undefined) return `Gerak: scan ${nodeLabel(frame.currentId)}`;
  if (frame.entryId !== null && frame.entryId !== undefined) return `Gerak: mulai dari ${nodeLabel(frame.entryId)}`;
  return "Gerak: -";
}

function focusText(frame) {
  const parts = [];
  if (frame.query) parts.push(frame.query.id === "q" ? "q" : nodeLabel(frame.query.id));
  if (frame.currentId !== null && frame.currentId !== undefined) parts.push(nodeLabel(frame.currentId));
  if (frame.selected?.length) parts.push(`R=${asLabels(frame.selected)}`);
  if (frame.found?.length) parts.push(`W=${asLabels(frame.found)}`);
  return `Fokus: ${parts.length ? parts.join(" | ") : "-"}`;
}

function animateDraw(fromFrame, toFrame, animated) {
  cancelAnimation();
  if (!animated) {
    draw(toFrame, 1, null);
    return;
  }

  const startedAt = performance.now();
  const duration = 520;
  const tick = (now) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    draw(toFrame, easeInOut(progress), fromFrame);
    if (progress < 1) {
      app.animationHandle = requestAnimationFrame(tick);
    } else {
      app.animationHandle = null;
    }
  };
  app.animationHandle = requestAnimationFrame(tick);
}

function cancelAnimation() {
  if (app.animationHandle) cancelAnimationFrame(app.animationHandle);
  app.animationHandle = null;
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function canvasPoint(node, camera) {
  return {
    x: camera.centerX + (node.x - camera.cx) * camera.scale,
    y: camera.centerY + (node.y - camera.cy) * camera.scale,
  };
}

function graphNode(frame, id) {
  return frame.graph.nodes.find((node) => node.id === id);
}

function computeCamera(frame) {
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const baseScale = Math.min((width - 150) / 100, (height - 150) / 100);
  const focus = focusPoints(frame);
  if (focus.length === 0) {
    return { cx: 50, cy: 50, scale: baseScale, centerX: width / 2, centerY: height / 2 };
  }

  const minX = Math.min(...focus.map((p) => p.x));
  const maxX = Math.max(...focus.map((p) => p.x));
  const minY = Math.min(...focus.map((p) => p.y));
  const maxY = Math.max(...focus.map((p) => p.y));
  const spanX = Math.max(34, maxX - minX + 18);
  const spanY = Math.max(30, maxY - minY + 18);
  const closeScale = Math.min((width - 190) / spanX, (height - 220) / spanY);
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    scale: clampNumber(closeScale, baseScale, baseScale * 2.55),
    centerX: width * 0.55,
    centerY: height * 0.57,
  };
}

function focusPoints(frame) {
  if (!frame || frame.title === "Base graph selesai" || frame.title === "Dataset siap") return [];
  const points = [];
  const addId = (id) => {
    const node = graphNode(frame, id);
    if (node && node.level >= app.activeLayer) points.push(node);
  };

  if (frame.query) points.push(frame.query);
  addId(frame.entryId);
  addId(frame.currentId);
  (frame.activeEdges || []).forEach(([a, b]) => {
    addId(a);
    addId(b);
  });
  [...(frame.selected || []), ...(frame.pruned || []), ...(frame.found || []).slice(0, 4), ...(frame.candidates || []).slice(0, 3)].forEach(addId);

  if (points.length === 1 && frame.currentId !== null && frame.currentId !== undefined) {
    neighborsOfSnapshot(frame, frame.currentId, app.activeLayer).slice(0, 3).forEach(addId);
  }
  return points;
}

function neighborsOfSnapshot(frame, id, layer) {
  const links = frame.graph.links[String(id)];
  return links?.[layer] || [];
}

function mixCamera(a, b, t) {
  if (!a) return b;
  return {
    cx: lerp(a.cx, b.cx, t),
    cy: lerp(a.cy, b.cy, t),
    scale: lerp(a.scale, b.scale, t),
    centerX: lerp(a.centerX, b.centerX, t),
    centerY: lerp(a.centerY, b.centerY, t),
  };
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function draw(frame, progress = 1, fromFrame = null) {
  fitCanvasToDisplay();
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  const camera = mixCamera(fromFrame ? computeCamera(fromFrame) : null, computeCamera(frame), progress);
  drawBackground(frame, camera);
  drawEdges(frame, camera, progress);
  drawQueryRings(frame, camera);
  drawNodes(frame, camera, progress);
  drawSideAnnotations(frame);
  drawQueueOverlay(frame);
  drawLayerTransition(fromFrame, frame, progress);
}

function fitCanvasToDisplay() {
  const rect = els.canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * scale);
  const height = Math.round(rect.height * scale);
  if (els.canvas.width !== width || els.canvas.height !== height) {
    els.canvas.width = width;
    els.canvas.height = height;
  }
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

function drawBackground(frame, camera) {
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = colors.layer;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const world = i * 10;
    const x1 = canvasPoint({ x: world, y: 0 }, camera);
    const x2 = canvasPoint({ x: world, y: 100 }, camera);
    const y1 = canvasPoint({ x: 0, y: world }, camera);
    const y2 = canvasPoint({ x: 100, y: world }, camera);
    ctx.beginPath();
    ctx.moveTo(x1.x, x1.y);
    ctx.lineTo(x2.x, x2.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(y1.x, y1.y);
    ctx.lineTo(y2.x, y2.y);
    ctx.stroke();
  }
  ctx.fillStyle = colors.text;
  ctx.font = "600 16px Inter, sans-serif";
  ctx.fillText(`Layer aktif ${app.activeLayer}`, width - 265, 146);
  ctx.fillStyle = colors.muted;
  ctx.font = "13px Inter, sans-serif";
  ctx.fillText("Kamera zoom mengikuti step.", width - 265, 164);
  if (frame.exact?.length) {
    ctx.fillStyle = colors.exact;
    ctx.fillText(`Exact top-K: ${asLabels(frame.exact)}`, 24, height - 62);
  }
  ctx.restore();
}

function drawLayerTransition(fromFrame, frame, progress) {
  if (!fromFrame || fromFrame.layer === frame.layer) return;
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const opacity = Math.sin(progress * Math.PI);
  if (opacity <= 0.01) return;

  const direction = frame.layer < fromFrame.layer ? "Turun layer" : "Naik layer";
  const x = width / 2 - 170;
  const y = Math.max(110, height / 2 - 62);

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.fillStyle = "rgba(15, 123, 108, 0.10)";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
  ctx.strokeStyle = "#0f7b6c";
  ctx.lineWidth = 2;
  roundRect(ctx, x, y, 340, 124, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = colors.accent || "#0f7b6c";
  ctx.font = "800 12px Inter, sans-serif";
  ctx.fillText(direction.toUpperCase(), x + 22, y + 28);

  ctx.fillStyle = colors.text;
  ctx.font = "800 30px Inter, sans-serif";
  ctx.fillText(`Layer ${fromFrame.layer} -> Layer ${frame.layer}`, x + 22, y + 66);

  ctx.fillStyle = colors.muted;
  ctx.font = "13px Inter, sans-serif";
  const message = frame.layer === 0
    ? "Sekarang masuk ground layer: pencarian lebih detail."
    : "Graph yang ditampilkan berganti ke layer baru.";
  ctx.fillText(message, x + 22, y + 92);

  const arrowX = x + 292;
  const arrowStart = y + 38;
  const arrowEnd = y + 92;
  ctx.strokeStyle = "#0f7b6c";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(arrowX, arrowStart);
  ctx.lineTo(arrowX, arrowEnd);
  ctx.stroke();
  const arrowAngle = frame.layer < fromFrame.layer ? Math.PI / 2 : -Math.PI / 2;
  drawTriangle(arrowX, arrowEnd, arrowAngle, 12, "#0f7b6c");
  ctx.restore();
}

function drawTriangle(x, y, angle, size, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
  ctx.lineTo(x + Math.cos(angle + 2.35) * size, y + Math.sin(angle + 2.35) * size);
  ctx.lineTo(x + Math.cos(angle - 2.35) * size, y + Math.sin(angle - 2.35) * size);
  ctx.closePath();
  ctx.fill();
}

function drawEdges(frame, camera, progress) {
  const graph = frame.graph;
  const layer = app.activeLayer;
  const activePairs = new Set((frame.activeEdges || []).map(([a, b]) => pairKey(a, b)));
  ctx.save();
  for (const [idText, byLayer] of Object.entries(graph.links)) {
    const id = Number(idText);
    const neighbors = byLayer[layer] || [];
    for (const nid of neighbors) {
      if (id > nid) continue;
      const a = graph.nodes.find((n) => n.id === id);
      const b = graph.nodes.find((n) => n.id === nid);
      if (!a || !b) continue;
      const pa = canvasPoint(a, camera);
      const pb = canvasPoint(b, camera);
      const active = activePairs.has(pairKey(id, nid));
      ctx.strokeStyle = active ? colors.edgeActive : colors.edge;
      ctx.lineWidth = active ? 3 : 1.4;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
  }
  drawTravelMarkers(frame, camera, progress);
  ctx.restore();
}

function drawTravelMarkers(frame, camera, progress) {
  for (const [aId, bId] of frame.activeEdges || []) {
    const a = graphNode(frame, aId);
    const b = graphNode(frame, bId);
    if (!a || !b) continue;
    const pa = canvasPoint(a, camera);
    const pb = canvasPoint(b, camera);
    const t = clampNumber(progress, 0, 1);
    const x = lerp(pa.x, pb.x, t);
    const y = lerp(pa.y, pb.y, t);
    ctx.fillStyle = "#111827";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    drawArrowHead(pa, pb);
  }
}

function drawArrowHead(from, to) {
  const angle = Math.atan2(to.y - from.y, to.x - from.x);
  const size = 10;
  ctx.fillStyle = colors.edgeActive;
  ctx.beginPath();
  ctx.moveTo(to.x, to.y);
  ctx.lineTo(to.x - Math.cos(angle - 0.45) * size, to.y - Math.sin(angle - 0.45) * size);
  ctx.lineTo(to.x - Math.cos(angle + 0.45) * size, to.y - Math.sin(angle + 0.45) * size);
  ctx.closePath();
  ctx.fill();
}

function pairKey(a, b) {
  return [a, b].sort().join(":");
}

function drawQueryRings(frame, camera) {
  if (!frame.query) return;
  const q = frame.query;
  const pq = canvasPoint(q, camera);
  ctx.save();
  ctx.strokeStyle = "rgba(17, 24, 39, 0.18)";
  ctx.lineWidth = 1;
  [10, 18, 26].forEach((r) => {
    ctx.beginPath();
    ctx.arc(pq.x, pq.y, r * camera.scale, 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.restore();
}

function drawNodes(frame, camera, progress) {
  const graph = frame.graph;
  const selected = new Set(frame.selected || []);
  const pruned = new Set(frame.pruned || []);
  const removed = new Set(frame.removed || []);
  const found = new Set(frame.found || []);
  const candidates = new Set(frame.candidates || []);
  const visited = new Set(frame.visited || []);
  const exact = new Set(frame.exact || []);

  for (const node of graph.nodes) {
    if (node.level < app.activeLayer) continue;
    const p = canvasPoint(node, camera);
    let fill = colors.node;
    if (visited.has(node.id)) fill = colors.visited;
    if (candidates.has(node.id)) fill = colors.candidate;
    if (found.has(node.id)) fill = colors.found;
    if (removed.has(node.id)) fill = colors.removed;
    if (pruned.has(node.id)) fill = colors.pruned;
    if (selected.has(node.id)) fill = colors.selected;
    if (node.id === frame.entryId) fill = colors.entry;
    const isCurrent = node.id === frame.currentId;
    if (isCurrent) fill = "#000000";

    ctx.save();
    if (exact.has(node.id)) {
      ctx.strokeStyle = colors.exact;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 15, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.fillStyle = fill;
    ctx.strokeStyle = colors.nodeStroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    const pulse = isCurrent ? 1 + Math.sin(progress * Math.PI) * 0.35 : 1;
    ctx.arc(p.x, p.y, (node.level > app.activeLayer ? 11 : 9) * pulse, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = colors.text;
    ctx.font = "600 12px Inter, sans-serif";
    ctx.fillText(nodeLabel(node.id), p.x + 12, p.y - 10);
    if (node.level > 0) {
      ctx.fillStyle = colors.muted;
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(`L${node.level}`, p.x + 12, p.y + 5);
    }
    ctx.restore();
  }

  if (frame.query) {
    const p = canvasPoint(frame.query, camera);
    ctx.save();
    ctx.fillStyle = colors.query;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.x, p.y - 12);
    ctx.lineTo(p.x + 12, p.y);
    ctx.lineTo(p.x, p.y + 12);
    ctx.lineTo(p.x - 12, p.y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = colors.text;
    ctx.font = "700 12px Inter, sans-serif";
    ctx.fillText(frame.query.id === "q" ? "q" : nodeLabel(frame.query.id), p.x + 15, p.y - 12);
    ctx.restore();
  }
}

function drawSideAnnotations(frame) {
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  ctx.save();
  ctx.fillStyle = "rgba(246, 247, 249, 0.92)";
  ctx.strokeStyle = "#d6dde5";
  ctx.lineWidth = 1;
  roundRect(ctx, width - 265, 18, 245, 104, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = colors.text;
  ctx.font = "700 13px Inter, sans-serif";
  ctx.fillText("Fokus pseudocode", width - 248, 42);
  ctx.fillStyle = colors.muted;
  ctx.font = "12px Inter, sans-serif";
  const lines = pseudocodeLines(frame.algorithm);
  lines.forEach((line, index) => ctx.fillText(line, width - 248, 64 + index * 17));
  ctx.restore();
}

function drawQueueOverlay(frame) {
  const rect = els.canvas.getBoundingClientRect();
  const width = rect.width;
  const x = Math.max(16, width - 265);
  const y = 184;
  const cardWidth = Math.min(245, width - x - 16);
  const rows = [
    { label: "C candidates", value: asLabels(frame.candidates), color: colors.candidate },
    { label: "W nearest list", value: asLabels(frame.found), color: colors.found },
    { label: "Dibuang dari W", value: asLabels(frame.removed), color: colors.removed },
    { label: "R selected", value: asLabels(frame.selected), color: colors.selected },
    { label: "Wd pruned", value: asLabels(frame.pruned), color: colors.pruned },
  ];
  const rowHeight = 34;
  const cardHeight = 42 + rows.length * rowHeight;

  ctx.save();
  ctx.fillStyle = "rgba(255, 255, 255, 0.94)";
  ctx.strokeStyle = "#d6dde5";
  ctx.lineWidth = 1;
  roundRect(ctx, x, y, cardWidth, cardHeight, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = colors.text;
  ctx.font = "700 13px Inter, sans-serif";
  ctx.fillText("Queue / List", x + 16, y + 24);

  rows.forEach((row, index) => {
    const rowY = y + 45 + index * rowHeight;
    ctx.fillStyle = row.color;
    ctx.beginPath();
    ctx.arc(x + 17, rowY - 5, 4.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = colors.muted;
    ctx.font = "600 11px Inter, sans-serif";
    ctx.fillText(row.label, x + 30, rowY - 9);
    ctx.fillStyle = colors.text;
    ctx.font = "12px SFMono-Regular, Consolas, monospace";
    ctx.fillText(truncateText(row.value, cardWidth - 44), x + 30, rowY + 8);
  });
  ctx.restore();
}

function truncateText(text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let value = text;
  while (value.length > 1 && ctx.measureText(`${value}...`).width > maxWidth) {
    value = value.slice(0, -1);
  }
  return `${value}...`;
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function pseudocodeLines(algorithm) {
  if ((algorithm || "").includes("4")) return ["R <- link terpilih", "W <- candidate queue", "jika arah baru: keep", "jika redundant: Wd"];
  if ((algorithm || "").includes("5")) return ["ep <- entry point", "layer atas: ef=1", "layer 0: efSearch", "return best K"];
  if ((algorithm || "").includes("2")) return ["C <- candidates", "W <- nearest list", "expand nearest C", "stop jika terlalu jauh"];
  if ((algorithm || "").includes("1")) return ["sample level l", "search top-down", "select neighbors", "add bidirectional links"];
  return ["Buat base graph", "Prepare Scenario", "lalu tekan Next"];
}

function nextStep() {
  if (app.step < app.frames.length - 1) {
    app.step += 1;
    renderCurrentFrame({ animate: true });
  } else {
    stopPlayback();
  }
}

function prevStep() {
  app.step = Math.max(0, app.step - 1);
  renderCurrentFrame({ animate: true });
}

function stopPlayback() {
  app.playing = false;
  els.playBtn.textContent = "Play";
  if (app.timer) window.clearInterval(app.timer);
  app.timer = null;
}

function togglePlayback() {
  if (app.playing) {
    stopPlayback();
    return;
  }
  app.playing = true;
  els.playBtn.textContent = "Pause";
  app.timer = window.setInterval(nextStep, 900);
}

els.resetBtn.addEventListener("click", () => resetDataset(false));
els.randomBtn.addEventListener("click", () => resetDataset(true));
els.buildBtn.addEventListener("click", buildBaseGraph);
els.scenarioBtn.addEventListener("click", prepareScenario);
els.prevBtn.addEventListener("click", prevStep);
els.nextBtn.addEventListener("click", nextStep);
els.playBtn.addEventListener("click", togglePlayback);
els.slider.addEventListener("input", (event) => {
  app.step = Number.parseInt(event.target.value, 10);
  renderCurrentFrame({ animate: true });
});
els.layerSelect.addEventListener("change", (event) => {
  app.activeLayer = Number.parseInt(event.target.value, 10);
  const frame = app.frames[app.step];
  if (frame) draw(frame, 1, null);
});
els.tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
[els.mInput, els.mmaxInput, els.efConstructionInput, els.efSearchInput, els.kInput].forEach((input) => {
  input.addEventListener("change", () => {
    if (app.graph.nodes.size > 0) buildBaseGraph();
  });
});
window.addEventListener("resize", () => renderCurrentFrame());

resetDataset(false);
buildBaseGraph();
