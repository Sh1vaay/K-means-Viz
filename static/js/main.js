/* ===== State ===== */
let chart;
let points = [];       // Full 3D: [[x,y,z], ...]
let points2d = [];     // 2D slice: [[x,y], ...] — used for algorithms
let k = 4;
let centroids = null;
let historyTrails = [];
let isRunning = false;
let iteration = 0;
let currentPhase = 'init';
let animationSpeed = 500;
let initMode = 'random';
let currentAlgorithm = 'kmeans';
let eps = 0.5;
let minSamples = 5;
let currentTheme = 'cyberpunk';
let currentTool = 'pointer';
let brushRadius = 15;
let lastLabels = [];
let lastOutliers = [];
let viewMode = '2d';   // '2d' | '3d'
let stateHistory = [];  // Undo stack: [{centroids, labels, phase, iteration, outliers}]

// DBSCAN
let dbscanFrames = [], dbscanFrameIdx = 0, currentDbscanFrame = null;
// GMM
let gmmFrames = [], gmmFrameIdx = 0, currentGmmFrame = null, gmmComponents = 4;
// Hierarchical
let hierK = 4, hierDendrogramData = null;
// Drag / Spray
let isDragging = false, dragPointIdx = -1, isSpraying = false;
// Analysis
let analysisChart = null;
let silhouetteChartObj = null;
let featureChartObj = null;
let autokChartObj = null;
let pcaChartObj = null;
let distanceMetric = 'euclidean';
let currentPalette = 'neon';

const PALETTES = {
    neon: [
        'rgba(255,99,132,0.8)', 'rgba(54,162,235,0.8)', 'rgba(255,206,86,0.8)',
        'rgba(75,192,192,0.8)', 'rgba(153,102,255,0.8)', 'rgba(255,159,64,0.8)',
        'rgba(231,233,237,0.8)', 'rgba(255,0,255,0.8)', 'rgba(0,255,127,0.8)', 'rgba(0,0,128,0.8)'
    ],
    viridis: [
        'rgba(68,1,84,0.8)', 'rgba(59,82,139,0.8)', 'rgba(33,145,140,0.8)',
        'rgba(94,201,98,0.8)', 'rgba(253,231,37,0.8)', 'rgba(122,209,81,0.8)',
        'rgba(34,168,132,0.8)', 'rgba(42,120,142,0.8)', 'rgba(65,68,135,0.8)', 'rgba(72,36,117,0.8)'
    ],
    plasma: [
        'rgba(13,8,135,0.8)', 'rgba(84,2,163,0.8)', 'rgba(139,10,165,0.8)',
        'rgba(185,50,137,0.8)', 'rgba(219,92,104,0.8)', 'rgba(244,136,73,0.8)',
        'rgba(254,188,43,0.8)', 'rgba(240,249,33,0.8)', 'rgba(204,71,120,0.8)', 'rgba(126,3,168,0.8)'
    ],
    earthy: [
        'rgba(139,90,43,0.8)', 'rgba(85,107,47,0.8)', 'rgba(165,42,42,0.8)',
        'rgba(184,134,11,0.8)', 'rgba(107,142,35,0.8)', 'rgba(160,82,45,0.8)',
        'rgba(128,128,0,0.8)', 'rgba(188,143,143,0.8)', 'rgba(112,128,144,0.8)', 'rgba(85,60,43,0.8)'
    ]
};

let CC = PALETTES.neon;

let CCL = CC.map(c => c.replace('0.8', '0.08'));
// Plotly hex colors — derived from active palette
function rgbaToHex(rgba) {
    const m = rgba.match(/\d+/g);
    return '#' + [m[0], m[1], m[2]].map(n => parseInt(n).toString(16).padStart(2, '0')).join('');
}
let CC_HEX = CC.map(rgbaToHex);

/* ===== Helper: sync points2d from points ===== */
function syncPoints2d() {
    points2d = points.map(p => [p[0], p[1]]);
}

/* ===== Voronoi Plugin ===== */
const voronoiPlugin = {
    id: 'voronoiPlugin',
    beforeDatasetsDraw(chart) {
        if (currentAlgorithm !== 'kmeans' || !centroids || centroids.length < 2) return;
        const ctx = chart.ctx, xS = chart.scales.x, yS = chart.scales.y, a = chart.chartArea;
        const px = centroids.map(c => [xS.getPixelForValue(c[0]), yS.getPixelForValue(c[1])]);
        try {
            const del = d3.Delaunay.from(px);
            const vor = del.voronoi([a.left, a.top, a.right, a.bottom]);
            ctx.save(); ctx.beginPath(); ctx.rect(a.left, a.top, a.right - a.left, a.bottom - a.top); ctx.clip();
            for (let i = 0; i < px.length; i++) {
                const c = vor.cellPolygon(i); if (!c) continue;
                ctx.beginPath(); ctx.moveTo(c[0][0], c[0][1]);
                for (let j = 1; j < c.length; j++) ctx.lineTo(c[j][0], c[j][1]);
                ctx.closePath(); ctx.fillStyle = CCL[i % CCL.length]; ctx.fill();
                ctx.strokeStyle = CC[i % CC.length].replace('0.8', '0.25'); ctx.lineWidth = 1; ctx.stroke();
            }
            ctx.restore();
        } catch (e) { }
    }
};

/* ===== DBSCAN Radius Plugin ===== */
const dbscanRadiusPlugin = {
    id: 'dbscanRadiusPlugin',
    afterDatasetsDraw(chart) {
        if (currentAlgorithm !== 'dbscan' || !currentDbscanFrame || currentDbscanFrame.current_idx < 0) return;
        const ctx = chart.ctx, xS = chart.scales.x, yS = chart.scales.y;
        const idx = currentDbscanFrame.current_idx;
        if (idx >= points2d.length) return;
        const px = xS.getPixelForValue(points2d[idx][0]), py = yS.getPixelForValue(points2d[idx][1]);
        const pixR = Math.abs(xS.getPixelForValue(eps) - xS.getPixelForValue(0));
        ctx.save();
        ctx.beginPath(); ctx.arc(px, py, pixR, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0,206,201,0.6)'; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.stroke();
        ctx.fillStyle = 'rgba(0,206,201,0.06)'; ctx.fill(); ctx.setLineDash([]);
        (currentDbscanFrame.neighbors || []).forEach(ni => {
            if (ni === idx || ni >= points2d.length) return;
            ctx.beginPath(); ctx.moveTo(px, py);
            ctx.lineTo(xS.getPixelForValue(points2d[ni][0]), yS.getPixelForValue(points2d[ni][1])); ctx.strokeStyle = 'rgba(0,206,201,0.25)'; ctx.lineWidth = 1; ctx.stroke();
        });
        ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,206,201,0.9)'; ctx.fill(); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
    }
};

/* ===== Centroid Trail Plugin ===== */
const centroidTrailPlugin = {
    id: 'centroidTrailPlugin',
    beforeDatasetsDraw(chart) {
        if (currentAlgorithm !== 'kmeans' || historyTrails.length < 2) return;
        const ctx = chart.ctx, xS = chart.scales.x, yS = chart.scales.y;
        ctx.save();
        // Determine how many centroids (from the latest snapshot)
        const numCentroids = historyTrails[0].length;
        for (let c = 0; c < numCentroids; c++) {
            ctx.beginPath();
            let started = false;
            for (let t = 0; t < historyTrails.length; t++) {
                if (!historyTrails[t][c]) continue;
                const px = xS.getPixelForValue(historyTrails[t][c][0]);
                const py = yS.getPixelForValue(historyTrails[t][c][1]);
                if (!started) { ctx.moveTo(px, py); started = true; }
                else ctx.lineTo(px, py);
            }
            // Also draw to current centroid position
            if (centroids && centroids[c]) {
                ctx.lineTo(xS.getPixelForValue(centroids[c][0]), yS.getPixelForValue(centroids[c][1]));
            }
            ctx.strokeStyle = CC[c % CC.length].replace('0.8', '0.5');
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
            // Draw dots at each historical position
            for (let t = 0; t < historyTrails.length; t++) {
                if (!historyTrails[t][c]) continue;
                const px = xS.getPixelForValue(historyTrails[t][c][0]);
                const py = yS.getPixelForValue(historyTrails[t][c][1]);
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fillStyle = CC[c % CC.length].replace('0.8', '0.35');
                ctx.fill();
            }
        }
        ctx.restore();
    }
};

/* ===== GMM Ellipse Plugin ===== */
const gmmEllipsePlugin = {
    id: 'gmmEllipsePlugin',
    afterDatasetsDraw(chart) {
        if (currentAlgorithm !== 'gmm' || !currentGmmFrame || !currentGmmFrame.ellipses) return;
        const ctx = chart.ctx, xS = chart.scales.x, yS = chart.scales.y, a = chart.chartArea;
        ctx.save(); ctx.beginPath(); ctx.rect(a.left, a.top, a.right - a.left, a.bottom - a.top); ctx.clip();
        currentGmmFrame.ellipses.forEach((e, i) => {
            const cx = xS.getPixelForValue(e.cx), cy = yS.getPixelForValue(e.cy);
            const rxP = Math.abs(xS.getPixelForValue(e.rx) - xS.getPixelForValue(0));
            const ryP = Math.abs(yS.getPixelForValue(0) - yS.getPixelForValue(e.ry));
            const ang = -e.angle * Math.PI / 180;
            const col = CC[i % CC.length];
            for (let s = 2; s >= 1; s--) {
                ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang);
                ctx.beginPath(); ctx.ellipse(0, 0, rxP * s / 2, ryP * s / 2, 0, 0, Math.PI * 2);
                ctx.strokeStyle = col.replace('0.8', String(0.5 / s)); ctx.lineWidth = s === 2 ? 1 : 2;
                ctx.setLineDash(s === 2 ? [4, 3] : []); ctx.stroke();
                if (s === 2) { ctx.fillStyle = col.replace('0.8', '0.04'); ctx.fill(); }
                ctx.restore();
            }
            ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2; ctx.stroke();
            ctx.fillStyle = col; ctx.font = 'bold 11px Outfit'; ctx.textAlign = 'center';
            ctx.fillText(`w=${e.weight.toFixed(2)}`, cx, cy - 12);
        });
        ctx.restore();
    }
};

/* ===== Spray Cursor ===== */
let sprayPreviewPos = null;
const sprayCursorPlugin = {
    id: 'sprayCursorPlugin',
    afterDatasetsDraw(chart) {
        if (currentTool !== 'spray' || !sprayPreviewPos) return;
        const ctx = chart.ctx; ctx.save(); ctx.beginPath();
        ctx.arc(sprayPreviewPos.x, sprayPreviewPos.y, brushRadius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0,206,201,0.5)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.stroke(); ctx.setLineDash([]); ctx.restore();
    }
};

/* ===== Init ===== */
document.addEventListener('DOMContentLoaded', () => {
    Chart.register(voronoiPlugin, dbscanRadiusPlugin, gmmEllipsePlugin, sprayCursorPlugin, centroidTrailPlugin);
    initChart();
    generateData('blobs');
    setupCanvasInteractions();

    document.getElementById('k-value').addEventListener('input', e => { k = parseInt(e.target.value); document.getElementById('k-display').textContent = k; if (initMode === 'random') reset(); });
    document.getElementById('speed-val').addEventListener('input', e => { animationSpeed = parseInt(e.target.value); });
    document.getElementById('eps-value').addEventListener('input', e => { eps = parseFloat(e.target.value); document.getElementById('eps-display').textContent = eps.toFixed(2); });
    document.getElementById('min-samples-value').addEventListener('input', e => { minSamples = parseInt(e.target.value); document.getElementById('min-samples-display').textContent = minSamples; });
    document.getElementById('brush-size').addEventListener('input', e => { brushRadius = parseInt(e.target.value); document.getElementById('brush-display').textContent = brushRadius; });
    document.getElementById('gmm-k-value').addEventListener('input', e => { gmmComponents = parseInt(e.target.value); document.getElementById('gmm-k-display').textContent = gmmComponents; });
    document.getElementById('hier-k-value').addEventListener('input', e => { hierK = parseInt(e.target.value); document.getElementById('hier-k-display').textContent = hierK; });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAnalysisModal(); });
});

/* ===== Theme ===== */
function setTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    event.currentTarget.classList.add('active');
    if (viewMode === '3d') render3D();
}

/* ===== View Mode Toggle ===== */
function setViewMode(mode) {
    viewMode = mode;
    document.getElementById('view-2d').className = mode === '2d' ? 'toggle-btn active' : 'toggle-btn';
    document.getElementById('view-3d').className = mode === '3d' ? 'toggle-btn active' : 'toggle-btn';

    const chart2d = document.getElementById('clusterChart');
    const dend = document.getElementById('dendrogramCanvas');
    const plotly3d = document.getElementById('plotly3d');
    const toolSection = document.querySelectorAll('.tool-bar, #brush-size-row');

    if (mode === '3d') {
        chart2d.style.display = 'none';
        if (dend) dend.style.display = 'none';
        plotly3d.style.display = 'block';
        toolSection.forEach(el => el.style.opacity = '0.4');
        document.getElementById('canvas-overlay').querySelector('p').textContent = 'Rotate & zoom the 3D scatter plot!';
        render3D();
    } else {
        chart2d.style.display = 'block';
        plotly3d.style.display = 'none';
        toolSection.forEach(el => el.style.opacity = '1');
        document.getElementById('canvas-overlay').querySelector('p').textContent = 'Select a tool and interact with the canvas!';
        chart.update();
    }
}

/* ===== 3D Rendering with Plotly ===== */
function render3D() {
    const container = document.getElementById('plotly3d');
    const isDark = currentTheme === 'cyberpunk';

    // Build traces grouped by cluster label
    const traceMap = {};
    const defaultColor = isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.3)';

    points.forEach((p, i) => {
        const label = (lastLabels && lastLabels.length > 0 && lastLabels[i] !== undefined) ? lastLabels[i] : -2;
        if (!traceMap[label]) traceMap[label] = { x: [], y: [], z: [], color: null };
        traceMap[label].x.push(p[0]);
        traceMap[label].y.push(p[1]);
        traceMap[label].z.push(p.length > 2 ? p[2] : 0);
    });

    const traces = [];
    for (const [label, group] of Object.entries(traceMap)) {
        const l = parseInt(label);
        let color;
        if (l === -2) color = defaultColor;
        else if (l === -1) color = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';
        else color = CC_HEX[l % CC_HEX.length];

        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: group.x, y: group.y, z: group.z,
            name: l === -2 ? 'Unassigned' : l === -1 ? 'Noise' : `Cluster ${l}`,
            marker: {
                size: l === -1 ? 3 : 5,
                color,
                opacity: l === -1 ? 0.3 : 0.9,
                line: { color: isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)', width: l === -1 ? 0 : 1 }
            }
        });
    }

    // Add centroid markers if available
    if (centroids && centroids.length > 0 && (currentAlgorithm === 'kmeans' || currentAlgorithm === 'gmm')) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: centroids.map(c => c[0]),
            y: centroids.map(c => c[1]),
            z: centroids.map(() => 0), // centroids are 2D
            name: 'Centroids',
            marker: {
                size: 8, symbol: 'diamond',
                color: centroids.map((_, i) => CC_HEX[i % CC_HEX.length]),
                line: { color: 'white', width: 2 }
            }
        });
    }

    const bgColor = isDark ? 'rgba(10,10,30,0)' : 'rgba(255,255,255,0)';
    const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
    const textColor = isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)';

    const layout = {
        paper_bgcolor: bgColor,
        plot_bgcolor: bgColor,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        showlegend: true,
        legend: { font: { color: textColor }, x: 0, y: 1 },
        scene: {
            bgcolor: bgColor,
            xaxis: { title: 'X', gridcolor: gridColor, zerolinecolor: gridColor, color: textColor, showbackground: false },
            yaxis: { title: 'Y', gridcolor: gridColor, zerolinecolor: gridColor, color: textColor, showbackground: false },
            zaxis: { title: 'Z', gridcolor: gridColor, zerolinecolor: gridColor, color: textColor, showbackground: false },
            camera: { eye: { x: 1.2, y: 1.2, z: 0.8 } }
        }
    };

    Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
}

/* ===== Tool Switching ===== */
function setTool(tool) {
    currentTool = tool;
    const wrap = document.getElementById('canvas-wrap');
    wrap.classList.remove('spray-mode', 'drag-mode');
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`tool-${tool}`).classList.add('active');
    document.getElementById('brush-size-row').style.display = tool === 'spray' ? 'block' : 'none';
    const msg = tool === 'spray' ? 'Hold and drag to spray points!' : tool === 'drag' ? 'Click and drag a point!' : 'Click to add points or centroids!';
    if (tool === 'spray') wrap.classList.add('spray-mode'); else if (tool === 'drag') wrap.classList.add('drag-mode');
    document.getElementById('canvas-overlay').querySelector('p').textContent = msg;
}

/* ===== Algorithm ===== */
function setAlgorithm(algo) {
    if (isRunning) return;
    currentAlgorithm = algo;
    ['kmeans', 'dbscan', 'gmm', 'hierarchical'].forEach(a => {
        document.getElementById(`algo-${a}`).className = a === algo ? 'toggle-btn active' : 'toggle-btn';
    });
    document.getElementById('kmeans-params').style.display = algo === 'kmeans' ? 'block' : 'none';
    document.getElementById('dbscan-params').style.display = algo === 'dbscan' ? 'block' : 'none';
    document.getElementById('gmm-params').style.display = algo === 'gmm' ? 'block' : 'none';
    document.getElementById('hier-params').style.display = algo === 'hierarchical' ? 'block' : 'none';
    if (algo === 'dbscan') { document.getElementById('metric-label').textContent = 'Noise:'; document.getElementById('inertia-val').textContent = '0'; document.getElementById('iteration-meta').style.display = 'none'; }
    else if (algo === 'gmm') { document.getElementById('metric-label').textContent = 'Log-Likelihood:'; document.getElementById('inertia-val').textContent = '0.00'; document.getElementById('iteration-meta').style.display = 'block'; }
    else if (algo === 'hierarchical') { document.getElementById('metric-label').textContent = 'Clusters:'; document.getElementById('inertia-val').textContent = '0'; document.getElementById('iteration-meta').style.display = 'none'; }
    else { document.getElementById('metric-label').textContent = 'Inertia:'; document.getElementById('inertia-val').textContent = '0.00'; document.getElementById('iteration-meta').style.display = 'block'; }

    const btn = document.getElementById('analysisBtn');
    if (algo === 'hierarchical') btn.innerHTML = '📊 View Dendrogram';
    else if (algo === 'dbscan') btn.innerHTML = '📊 K-Distance Graph';
    else btn.innerHTML = '📊 Elbow Method Graph';

    reset();
}

function setInitMode(mode) {
    initMode = mode;
    document.getElementById('init-random').className = mode === 'random' ? 'toggle-btn active' : 'toggle-btn';
    document.getElementById('init-manual').className = mode === 'manual' ? 'toggle-btn active' : 'toggle-btn';
    reset();
    if (mode === 'manual') { centroids = []; updateStatusPanel('ready', `Click to place ${k} centroids.`, 0, 0); }
}

/* ===== Chart ===== */
function initChart() {
    chart = new Chart(document.getElementById('clusterChart').getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Points', data: [], backgroundColor: 'rgba(255,255,255,0.6)', pointRadius: 6, borderWidth: 0 },
                { label: 'Centroids', data: [], backgroundColor: 'white', borderColor: 'white', borderWidth: 2, pointStyle: 'crossRot', pointRadius: 15, pointHoverRadius: 15 }
            ]
        },
        options: {
            animation: { duration: 300, easing: 'easeOutQuart' },
            responsive: true, maintainAspectRatio: false,
            scales: { x: { display: false }, y: { display: false } },
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            onClick: (e) => {
                if (isRunning || currentTool !== 'pointer' || viewMode === '3d') return;
                const pos = Chart.helpers.getRelativePosition(e, chart);
                const dx = chart.scales.x.getValueForPixel(pos.x), dy = chart.scales.y.getValueForPixel(pos.y);
                if (currentAlgorithm === 'kmeans' && initMode === 'manual' && currentPhase === 'init') {
                    if (!centroids) centroids = [];
                    if (centroids.length < k) {
                        centroids.push([dx, dy]);
                        updateChartData(points2d, centroids, []);
                        updateStatusPanel('ready', centroids.length === k ? "All placed. Click Start." : `Placed ${centroids.length}/${k}...`, 0, 0);
                    }
                } else if (currentPhase === 'init') {
                    points.push([dx, dy, 0]); syncPoints2d();
                    updateChartData(points2d, centroids || [], []);
                }
            }
        }
    });
}

/* ===== Canvas interactions ===== */
function setupCanvasInteractions() {
    const canvas = document.getElementById('clusterChart');
    canvas.addEventListener('mousedown', e => {
        if (isRunning || viewMode === '3d') return;
        const rect = canvas.getBoundingClientRect(), px = e.clientX - rect.left, py = e.clientY - rect.top;
        if (currentTool === 'spray') { isSpraying = true; sprayPoints(px, py); }
        else if (currentTool === 'drag') { const c = findClosest(px, py); if (c.dist < 20) { isDragging = true; dragPointIdx = c.idx; } }
    });
    canvas.addEventListener('mousemove', e => {
        if (viewMode === '3d') return;
        const rect = canvas.getBoundingClientRect(), px = e.clientX - rect.left, py = e.clientY - rect.top;
        if (currentTool === 'spray') { sprayPreviewPos = { x: px, y: py }; chart.update('none'); }
        if (isSpraying && currentTool === 'spray') sprayPoints(px, py);
        if (isDragging && currentTool === 'drag' && dragPointIdx >= 0) {
            const dx = chart.scales.x.getValueForPixel(px), dy = chart.scales.y.getValueForPixel(py);
            points[dragPointIdx] = [dx, dy, points[dragPointIdx][2] || 0]; syncPoints2d();
            updateChartData(points2d, centroids || [], lastLabels);
        }
    });
    canvas.addEventListener('mouseup', () => {
        if (isSpraying) { isSpraying = false; updateChartData(points2d, centroids || [], []); }
        if (isDragging) { isDragging = false; dragPointIdx = -1; if (currentPhase !== 'init') instantRecluster(); }
    });
    canvas.addEventListener('mouseleave', () => { isSpraying = false; isDragging = false; dragPointIdx = -1; sprayPreviewPos = null; chart.update('none'); });
}

function sprayPoints(px, py) {
    for (let i = 0; i < 3; i++) {
        const a = Math.random() * Math.PI * 2, r = Math.random() * brushRadius;
        const dx = chart.scales.x.getValueForPixel(px + Math.cos(a) * r);
        const dy = chart.scales.y.getValueForPixel(py + Math.sin(a) * r);
        points.push([dx, dy, Math.random() * 2 - 1]); // random z
    }
    syncPoints2d();
    updateChartData(points2d, centroids || [], []);
}

function findClosest(px, py) {
    let md = Infinity, mi = -1;
    for (let i = 0; i < points2d.length; i++) {
        const dx = chart.scales.x.getPixelForValue(points2d[i][0]) - px, dy = chart.scales.y.getPixelForValue(points2d[i][1]) - py;
        const d = Math.sqrt(dx * dx + dy * dy); if (d < md) { md = d; mi = i; }
    }
    return { idx: mi, dist: md };
}

async function instantRecluster() {
    try {
        if (currentAlgorithm === 'kmeans' && centroids) {
            const r = await fetch('/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, k, centroids, phase: 'assign', distance_metric: distanceMetric }) });
            const d = await r.json(); centroids = d.centroids; lastLabels = d.labels;
            updateChartData(points2d, centroids, d.labels); updateStatusPanel('assign', 'Recalculated.', d.inertia, iteration);
        } else if (currentAlgorithm === 'dbscan') {
            const r = await fetch('/dbscan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, eps, min_samples: minSamples }) });
            const d = await r.json(); lastLabels = d.labels; updateChartData(points2d, [], d.labels);
            document.getElementById('step-desc').textContent = `${d.n_clusters} cluster(s), ${d.n_noise} noise.`;
            document.getElementById('inertia-val').textContent = d.n_noise;
        }
        fetchMetrics(); if (viewMode === '3d') render3D();
    } catch (e) { }
}

/* ===== Update Chart (2D) ===== */
function tc(p) { return getComputedStyle(document.documentElement).getPropertyValue(p).trim(); }
function updateChartData(pts, cents, labels) {
    lastLabels = labels;
    const def = tc('--point-default') || 'rgba(255,255,255,0.3)', noise = tc('--noise-color') || 'rgba(255,255,255,0.15)';
    const outlierSet = new Set(lastOutliers);
    const colors = pts.map((_, i) => {
        if (outlierSet.has(i)) return 'rgba(255,118,117,0.9)';  // red for outliers
        if (!labels || !labels.length || labels[i] === undefined) return def;
        if (labels[i] === -1) return noise;
        return CC[labels[i] % CC.length];
    });
    const radii = pts.map((_, i) => {
        if (outlierSet.has(i)) return 8;  // larger for outliers
        if (currentAlgorithm === 'dbscan' && labels && labels.length && labels[i] === -1) return 4;
        return 6;
    });
    const borders = pts.map((_, i) => outlierSet.has(i) ? 2 : 0);
    const borderColors = pts.map((_, i) => outlierSet.has(i) ? 'rgba(255,0,0,0.8)' : 'transparent');
    chart.data.datasets[0].data = pts.map(p => ({ x: p[0], y: p[1] }));
    chart.data.datasets[0].backgroundColor = colors;
    chart.data.datasets[0].pointRadius = radii;
    chart.data.datasets[0].borderWidth = borders;
    chart.data.datasets[0].borderColor = borderColors;
    const sc = cents || [];
    chart.data.datasets[1].data = sc.map(c => ({ x: c[0], y: c[1] }));
    chart.data.datasets[1].borderColor = sc.map((_, i) => CC[i % CC.length]);
    chart.data.datasets[1].backgroundColor = 'white';
    chart.update();
    if (viewMode === '3d') render3D();
}

/* ===== Data Generation ===== */
async function generateData(type) {
    if (isRunning) return;
    try {
        const r = await fetch('/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type, k }) });
        const d = await r.json();
        points = d.points; syncPoints2d(); reset();
    } catch (e) { }
}

/* ===== Simulation Controls ===== */
async function toggleSimulation() {
    const btn = document.getElementById('runBtn');
    if (isRunning) { isRunning = false; btn.textContent = "Resume"; return; }
    if (currentAlgorithm === 'kmeans') {
        if (initMode === 'manual' && (!centroids || centroids.length < k)) { alert(`Place ${k} centroids.`); return; }
        isRunning = true; btn.textContent = "Pause"; runKmeansLoop();
    } else if (currentAlgorithm === 'dbscan') {
        if (!dbscanFrames.length) {
            updateStatusPanel('running', 'Computing...', 0, 0);
            try {
                const r = await fetch('/dbscan_frames', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, eps, min_samples: minSamples }) });
                dbscanFrames = (await r.json()).frames; dbscanFrameIdx = 0;
            } catch (e) { return; }
        }
        isRunning = true; btn.textContent = "Pause"; runDbscanLoop();
    } else if (currentAlgorithm === 'gmm') {
        if (!gmmFrames.length) {
            updateStatusPanel('running', 'Computing EM...', 0, 0);
            try {
                const r = await fetch('/gmm_frames', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, n_components: gmmComponents }) });
                gmmFrames = (await r.json()).frames; gmmFrameIdx = 0;
            } catch (e) { return; }
        }
        isRunning = true; btn.textContent = "Pause"; runGmmLoop();
    } else if (currentAlgorithm === 'hierarchical') {
        await runHierarchical();
    }
}
async function manualStep() {
    if (isRunning) { isRunning = false; document.getElementById('runBtn').textContent = "Resume"; }
    if (currentAlgorithm === 'kmeans') await doKmeansStep();
    else if (currentAlgorithm === 'dbscan') await doDbscanStep();
    else if (currentAlgorithm === 'gmm') await doGmmStep();
    else await runHierarchical();
}

/* ===== K-Means ===== */
async function runKmeansLoop() { if (!isRunning) return; const f = await doKmeansStep(); if (!f && isRunning) setTimeout(runKmeansLoop, animationSpeed); }
async function doKmeansStep() {
    try {
        // Save state for undo before each step
        stateHistory.push({
            centroids: centroids ? JSON.parse(JSON.stringify(centroids)) : null,
            labels: [...lastLabels],
            phase: currentPhase,
            iteration: iteration,
            outliers: [...lastOutliers]
        });
        if (stateHistory.length > 50) stateHistory.shift(); // cap at 50

        const r = await fetch('/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, k, centroids, phase: currentPhase, distance_metric: distanceMetric }) });
        const d = await r.json();
        if (centroids && d.centroids) historyTrails.push(JSON.parse(JSON.stringify(centroids)));
        centroids = d.centroids; currentPhase = d.phase; lastLabels = d.labels;
        lastOutliers = d.outliers || [];
        updateChartData(points2d, centroids, d.labels); updateStatusPanel(d.phase, d.description, d.inertia, iteration);
        if (d.phase === 'assign') iteration++;
        if (d.converged) { isRunning = false; document.getElementById('runBtn').textContent = "Restart"; updateStatusPanel('converged', d.description, d.inertia, iteration); fetchMetrics(); return true; }
        return false;
    } catch (e) { isRunning = false; return true; }
}

/* ===== DBSCAN ===== */
async function runDbscanLoop() { if (!isRunning) return; const f = await doDbscanStep(); if (!f && isRunning) setTimeout(runDbscanLoop, animationSpeed); }
async function doDbscanStep() {
    if (!dbscanFrames.length) {
        try {
            const r = await fetch('/dbscan_frames', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, eps, min_samples: minSamples }) });
            dbscanFrames = (await r.json()).frames; dbscanFrameIdx = 0;
        } catch (e) { return true; }
    }
    if (dbscanFrameIdx >= dbscanFrames.length) { isRunning = false; document.getElementById('runBtn').textContent = "Restart"; return true; }
    const f = dbscanFrames[dbscanFrameIdx]; currentDbscanFrame = f; dbscanFrameIdx++; lastLabels = f.labels;
    updateChartData(points2d, [], f.labels);
    document.getElementById('phase-badge').textContent = f.current_idx < 0 ? 'DONE' : 'RUNNING';
    document.getElementById('phase-badge').className = `badge ${f.current_idx < 0 ? 'converged' : 'running'}`;
    document.getElementById('step-desc').textContent = f.desc; document.getElementById('inertia-val').textContent = f.n_noise;
    if (f.current_idx < 0) { isRunning = false; currentDbscanFrame = null; document.getElementById('runBtn').textContent = "Restart"; chart.update(); fetchMetrics(); return true; }
    return false;
}

/* ===== GMM ===== */
async function runGmmLoop() { if (!isRunning) return; const f = await doGmmStep(); if (!f && isRunning) setTimeout(runGmmLoop, animationSpeed); }
async function doGmmStep() {
    if (!gmmFrames.length) {
        try {
            const r = await fetch('/gmm_frames', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, n_components: gmmComponents }) });
            gmmFrames = (await r.json()).frames; gmmFrameIdx = 0;
        } catch (e) { return true; }
    }
    if (gmmFrameIdx >= gmmFrames.length) { isRunning = false; document.getElementById('runBtn').textContent = "Restart"; return true; }
    const f = gmmFrames[gmmFrameIdx]; currentGmmFrame = f; gmmFrameIdx++; lastLabels = f.labels; iteration = f.iteration;
    updateChartData(points2d, f.means, f.labels);
    document.getElementById('phase-badge').textContent = f.converged ? 'CONVERGED' : 'EM';
    document.getElementById('phase-badge').className = `badge ${f.converged ? 'converged' : 'running'}`;
    document.getElementById('step-desc').textContent = f.desc;
    document.getElementById('inertia-val').textContent = f.log_likelihood.toFixed(2);
    document.getElementById('iteration').textContent = f.iteration;
    if (f.converged) { isRunning = false; document.getElementById('runBtn').textContent = "Restart"; fetchMetrics(); return true; }
    return false;
}

/* ===== Hierarchical ===== */
async function runHierarchical() {
    if (points2d.length < 3) return;
    updateStatusPanel('running', 'Computing...', 0, 0);
    try {
        const r = await fetch('/hierarchical', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, n_clusters: hierK, linkage_method: 'ward' }) });
        const d = await r.json(); hierDendrogramData = d; lastLabels = d.labels;
        updateChartData(points2d, [], d.labels);
        document.getElementById('phase-badge').textContent = 'DONE'; document.getElementById('phase-badge').className = 'badge converged';
        document.getElementById('step-desc').textContent = `Ward: ${d.n_clusters} clusters.`;
        document.getElementById('inertia-val').textContent = d.n_clusters;
        document.getElementById('runBtn').textContent = "Restart"; fetchMetrics();
    } catch (e) { }
}

function renderDendrogramModal() {
    if (!hierDendrogramData) return;
    const data = hierDendrogramData;
    if (analysisChart) { analysisChart.destroy(); analysisChart = null; }
    const canvas = document.getElementById('analysisChart');
    const ctx = canvas.getContext('2d');

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 40;
    canvas.height = 300;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const nodes = data.nodes, n = points2d.length, maxD = data.max_distance || 1;
    const pt = 20, pb = canvas.height - 20, pl = 30, pr = canvas.width - 30, w = pr - pl, h = pb - pt;

    const leafs = nodes.filter(n => n.is_leaf).sort((a, b) => a.x - b.x);
    const sp = w / (leafs.length - 1 || 1), lx = {};
    leafs.forEach((l, i) => { lx[l.id] = pl + i * sp; });

    function gp(id) { const nd = nodes[id]; if (nd.is_leaf) return { x: lx[nd.id], y: pb }; const l = gp(nd.left), r = gp(nd.right); return { x: (l.x + r.x) / 2, y: pb - (nd.y / maxD) * h }; }

    ctx.strokeStyle = tc('--secondary') || 'rgba(0,206,201,0.6)'; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.8;
    nodes.forEach(nd => {
        if (nd.is_leaf) return; const me = gp(nd.id), l = gp(nd.left), r = gp(nd.right);
        ctx.beginPath(); ctx.moveTo(l.x, l.y); ctx.lineTo(l.x, me.y); ctx.lineTo(r.x, me.y); ctx.lineTo(r.x, r.y); ctx.stroke();
    });

    if (data.merge_steps.length >= 1) {
        const ci = Math.max(0, data.merge_steps.length - hierK);
        const cd = data.merge_steps[ci] ? data.merge_steps[ci].distance : maxD / 2;
        const cy = pb - (cd / maxD) * h;
        ctx.globalAlpha = 1.0; ctx.strokeStyle = '#ff7675'; ctx.lineWidth = 2; ctx.setLineDash([8, 4]);
        ctx.beginPath(); ctx.moveTo(pl - 10, cy); ctx.lineTo(pr + 10, cy); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#ff7675'; ctx.font = 'bold 12px Outfit'; ctx.textAlign = 'left'; ctx.fillText(`Cut → ${hierK} clusters`, pl, cy - 6);
    }
}

/* ===== Metrics ===== */
async function fetchMetrics() {
    if (!lastLabels || !lastLabels.length || !points.length) { document.getElementById('silhouette-val').textContent = '—'; document.getElementById('db-val').textContent = '—'; return; }
    try {
        const r = await fetch('/metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        const s = document.getElementById('silhouette-val'), b = document.getElementById('db-val');
        if (d.silhouette !== null) { s.textContent = d.silhouette.toFixed(4); s.className = 'mono ' + (d.silhouette > 0.5 ? 'metric-good' : d.silhouette > 0.25 ? 'metric-warn' : 'metric-bad'); }
        else { s.textContent = '—'; s.className = 'mono'; }
        if (d.davies_bouldin !== null) { b.textContent = d.davies_bouldin.toFixed(4); b.className = 'mono ' + (d.davies_bouldin < 1 ? 'metric-good' : d.davies_bouldin < 2 ? 'metric-warn' : 'metric-bad'); }
        else { b.textContent = '—'; b.className = 'mono'; }
    } catch (e) { }
}

/* ===== Analysis Modal ===== */
let activeModalTab = 'elbow';

function switchModalTab(tab) {
    activeModalTab = tab;
    document.querySelectorAll('.modal-tab').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-tab') === tab);
    });
    document.querySelectorAll('.modal-tab-content').forEach(c => {
        c.style.display = 'none';
        c.classList.remove('active');
    });
    const target = document.getElementById('tab-' + tab);
    if (target) { target.style.display = 'block'; target.classList.add('active'); }

    if (tab === 'elbow') {
        if (currentAlgorithm === 'hierarchical') renderDendrogramModal();
        else if (currentAlgorithm === 'dbscan') renderKDist();
        else renderElbow();
    } else if (tab === 'silhouette') renderSilhouettePlot();
    else if (tab === 'stats') renderClusterStats();
    else if (tab === 'features') renderFeatureImportance();
    else if (tab === 'autok') renderAutoK();
    else if (tab === 'pca') renderPCAChart();
    else if (tab === 'parcoords') renderParallelCoords();
    else if (tab === 'boxplot') renderBoxPlot();
}

async function openAnalysisModal() {
    if (!points.length) return;
    document.getElementById('analysis-modal').style.display = 'flex';
    document.getElementById('modal-title').textContent = 'Analysis Dashboard';
    switchModalTab(activeModalTab);
}

function closeAnalysisModal() {
    document.getElementById('analysis-modal').style.display = 'none';
    if (analysisChart) { analysisChart.destroy(); analysisChart = null; }
    if (silhouetteChartObj) { silhouetteChartObj.destroy(); silhouetteChartObj = null; }
    if (featureChartObj) { featureChartObj.destroy(); featureChartObj = null; }
    if (autokChartObj) { autokChartObj.destroy(); autokChartObj = null; }
    if (pcaChartObj) { pcaChartObj.destroy(); pcaChartObj = null; }
}

async function renderElbow() {
    try {
        const r = await fetch('/elbow', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, max_k: 10 }) });
        const res = await r.json(); const ls = res.data.map(d => d.k), vs = res.data.map(d => d.inertia);
        const curK = currentAlgorithm === 'gmm' ? gmmComponents : k;
        if (analysisChart) analysisChart.destroy();
        document.getElementById('analysis-hint').textContent = 'Look for the "elbow" — the point where inertia stops dropping significantly.';
        analysisChart = new Chart(document.getElementById('analysisChart').getContext('2d'), { type: 'line', data: { labels: ls, datasets: [{ label: 'Inertia', data: vs, borderColor: '#6c5ce7', backgroundColor: 'rgba(108,92,231,0.1)', fill: true, tension: 0.3, pointRadius: 6, pointBackgroundColor: ls.map(l => l === curK ? '#00cec9' : '#6c5ce7'), pointBorderColor: ls.map(l => l === curK ? '#fff' : 'transparent'), pointBorderWidth: ls.map(l => l === curK ? 3 : 0) }] }, options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'K', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } }, y: { title: { display: true, text: 'Inertia', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } } } } });
    } catch (e) { }
}
async function renderKDist() {
    try {
        const r = await fetch('/k_distance', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, k_neighbors: minSamples }) });
        const ds = (await r.json()).distances;
        if (analysisChart) analysisChart.destroy();
        document.getElementById('analysis-hint').textContent = 'Sharp bend indicates good eps value for DBSCAN.';
        analysisChart = new Chart(document.getElementById('analysisChart').getContext('2d'), { type: 'line', data: { labels: ds.map((_, i) => i + 1), datasets: [{ label: `${minSamples}-NN`, data: ds, borderColor: '#00cec9', backgroundColor: 'rgba(0,206,201,0.08)', fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2 }] }, options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Points', color: tc('--text-muted') }, ticks: { display: false }, grid: { color: tc('--glass-border') } }, y: { title: { display: true, text: 'Distance', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } } } } });
    } catch (e) { }
}

/* ===== Silhouette Plot ===== */
async function renderSilhouettePlot() {
    if (!lastLabels || !lastLabels.length) {
        document.getElementById('silhouette-hint').textContent = 'Run a clustering algorithm first.';
        return;
    }
    try {
        const r = await fetch('/silhouette_plot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        if (!d.samples.length) { document.getElementById('silhouette-hint').textContent = 'Need ≥2 clusters.'; return; }

        if (silhouetteChartObj) silhouetteChartObj.destroy();

        const labels = d.samples.map((_, i) => i);
        const values = d.samples.map(s => s.value);
        const colors = d.samples.map(s => s.cluster >= 0 ? CC[s.cluster % CC.length] : 'rgba(150,150,150,0.5)');

        silhouetteChartObj = new Chart(document.getElementById('silhouetteChart').getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [{ data: values, backgroundColor: colors, borderWidth: 0, barPercentage: 1.0, categoryPercentage: 1.0 }] },
            options: {
                responsive: true, indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Silhouette Value', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') }, min: -0.2, max: 1 },
                    y: { display: false }
                }
            }
        });
        document.getElementById('silhouette-hint').textContent = `Average silhouette: ${d.avg.toFixed(4)} | ${d.n_clusters} clusters`;
    } catch (e) { }
}

/* ===== Cluster Stats Table ===== */
async function renderClusterStats() {
    const container = document.getElementById('stats-container');
    if (!lastLabels || !lastLabels.length) {
        container.innerHTML = '<p class="small-meta">Run a clustering algorithm first.</p>';
        return;
    }
    try {
        const r = await fetch('/cluster_profile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        let html = '';
        d.clusters.forEach(c => {
            html += `<table class="stats-table">
                <caption>Cluster ${c.id} (${c.count} points)</caption>
                <thead><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std Dev</th></tr></thead>
                <tbody>`;
            c.features.forEach(f => {
                html += `<tr><td>${f.name}</td><td>${f.mean.toFixed(4)}</td><td>${f.median.toFixed(4)}</td><td>${f.std.toFixed(4)}</td></tr>`;
            });
            html += '</tbody></table>';
        });
        container.innerHTML = html;
    } catch (e) { container.innerHTML = '<p class="small-meta">Error loading stats.</p>'; }
}

/* ===== Feature Importance ===== */
async function renderFeatureImportance() {
    if (!lastLabels || !lastLabels.length) {
        document.getElementById('feature-hint').textContent = 'Run a clustering algorithm first.';
        return;
    }
    try {
        const r = await fetch('/feature_importance', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        if (!d.length) { document.getElementById('feature-hint').textContent = 'Need ≥2 clusters.'; return; }

        if (featureChartObj) featureChartObj.destroy();

        const names = d.map(f => f.name);
        const importances = d.map(f => f.importance);
        const barColors = importances.map(v => v > 0.7 ? '#00b894' : v > 0.3 ? '#fdcb6e' : '#ff7675');

        featureChartObj = new Chart(document.getElementById('featureChart').getContext('2d'), {
            type: 'bar',
            data: { labels: names, datasets: [{ label: 'Importance', data: importances, backgroundColor: barColors, borderRadius: 6, borderWidth: 0 }] },
            options: {
                responsive: true, indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Importance (Between/Total Var)', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') }, min: 0, max: 1 },
                    y: { ticks: { color: tc('--text-faint') }, grid: { display: false } }
                }
            }
        });
        document.getElementById('feature-hint').textContent = `Top feature: ${d[0].name} (${(d[0].importance * 100).toFixed(1)}% variance explained by clusters)`;
    } catch (e) { }
}

/* ===== Auto-K Suggestion ===== */
async function renderAutoK() {
    try {
        document.getElementById('autok-hint').textContent = 'Computing silhouette for K=2..10...';
        const r = await fetch('/auto_k', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, max_k: 10 }) });
        const d = await r.json();

        if (autokChartObj) autokChartObj.destroy();

        const ks = d.scores.map(s => s.k);
        const sils = d.scores.map(s => s.silhouette);

        autokChartObj = new Chart(document.getElementById('autokChart').getContext('2d'), {
            type: 'bar',
            data: { labels: ks, datasets: [{ label: 'Silhouette', data: sils, backgroundColor: ks.map(kv => kv === d.best_k ? '#00cec9' : 'rgba(108,92,231,0.6)'), borderRadius: 6, borderWidth: 0 }] },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'K', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } },
                    y: { title: { display: true, text: 'Silhouette Score', color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') }, min: 0, max: 1 }
                }
            }
        });
        document.getElementById('autok-hint').textContent = `Best K = ${d.best_k} (Silhouette: ${d.best_score.toFixed(4)})`;
    } catch (e) { }
}

async function suggestK() {
    if (!points.length) return;
    document.getElementById('auto-k-result').textContent = 'Analyzing...';
    try {
        const r = await fetch('/auto_k', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, max_k: 10 }) });
        const d = await r.json();
        document.getElementById('auto-k-result').textContent = `🎯 Best K = ${d.best_k} (Silhouette: ${d.best_score.toFixed(4)})`;
        document.getElementById('auto-k-result').style.color = tc('--secondary');
    } catch (e) { document.getElementById('auto-k-result').textContent = 'Error'; }
}

/* ===== Distance Metric ===== */
function setDistanceMetric(metric) {
    distanceMetric = metric;
    ['euclidean', 'manhattan', 'cosine'].forEach(m => {
        document.getElementById('dist-' + m).className = m === metric ? 'toggle-btn active' : 'toggle-btn';
    });
    reset();
}

/* ===== Color Palette ===== */
function setColorPalette(name) {
    currentPalette = name;
    CC = PALETTES[name] || PALETTES.neon;
    CCL = CC.map(c => c.replace('0.8', '0.08'));
    CC_HEX = CC.map(rgbaToHex);
    document.querySelectorAll('[data-palette]').forEach(b => {
        b.classList.toggle('active', b.getAttribute('data-palette') === name);
    });
    // Re-render if we have labels
    if (lastLabels && lastLabels.length && points2d.length) {
        updateChartData(points2d, centroids, lastLabels);
    }
}

/* ===== PCA Projection ===== */
async function renderPCAChart() {
    if (!points.length) { document.getElementById('pca-hint').textContent = 'Load data first.'; return; }
    try {
        const r = await fetch('/pca_projection', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels || [] }) });
        const d = await r.json();
        if (pcaChartObj) pcaChartObj.destroy();

        const datasets = [];
        const clusterIds = [...new Set(d.labels)].sort((a, b) => a - b);
        clusterIds.forEach(cid => {
            const pts = [];
            d.points.forEach((p, i) => { if (d.labels[i] === cid) pts.push({ x: p[0], y: p[1] }); });
            datasets.push({
                label: cid === -1 ? 'Noise' : `Cluster ${cid}`,
                data: pts,
                backgroundColor: cid === -1 ? 'rgba(150,150,150,0.4)' : CC[cid % CC.length],
                pointRadius: 4, pointHoverRadius: 6, borderWidth: 0
            });
        });

        pcaChartObj = new Chart(document.getElementById('pcaChart').getContext('2d'), {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: tc('--text-faint'), font: { size: 10 } } } },
                scales: {
                    x: { title: { display: true, text: `PC1 (${(d.explained_variance[0] * 100).toFixed(1)}%)`, color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } },
                    y: { title: { display: true, text: `PC2 (${(d.explained_variance[1] ? (d.explained_variance[1] * 100).toFixed(1) : 0)}%)`, color: tc('--text-muted') }, ticks: { color: tc('--text-faint') }, grid: { color: tc('--glass-border') } }
                }
            }
        });
        const totalVar = d.explained_variance.reduce((a, b) => a + b, 0) * 100;
        document.getElementById('pca-hint').textContent = `Total variance explained: ${totalVar.toFixed(1)}% by 2 principal components`;
    } catch (e) { }
}

/* ===== Parallel Coordinates ===== */
async function renderParallelCoords() {
    if (!lastLabels || !lastLabels.length) {
        document.getElementById('parcoords-hint').textContent = 'Run a clustering algorithm first.';
        return;
    }
    try {
        const r = await fetch('/parallel_coords', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        const canvas = document.getElementById('parcoordsCanvas');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = 300 * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = '300px';
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, rect.width, 300);

        const nF = d.features.length;
        const pad = { l: 50, r: 30, t: 25, b: 30 };
        const w = rect.width - pad.l - pad.r;
        const h = 300 - pad.t - pad.b;
        const gap = w / (nF - 1);

        // Draw axes
        ctx.strokeStyle = tc('--glass-border') || '#444';
        ctx.lineWidth = 1;
        ctx.fillStyle = tc('--text-faint') || '#aaa';
        ctx.font = '11px Outfit, sans-serif';
        ctx.textAlign = 'center';
        for (let i = 0; i < nF; i++) {
            const x = pad.l + i * gap;
            ctx.beginPath();
            ctx.moveTo(x, pad.t);
            ctx.lineTo(x, pad.t + h);
            ctx.stroke();
            ctx.fillText(d.features[i], x, pad.t + h + 18);
            // Scale labels
            ctx.fillText('1', x - 14, pad.t + 6);
            ctx.fillText('0', x - 14, pad.t + h + 6);
        }

        // Draw lines
        ctx.lineWidth = 0.6;
        ctx.globalAlpha = 0.35;
        d.lines.forEach(line => {
            const col = line.cluster >= 0 ? CC[line.cluster % CC.length] : 'rgba(150,150,150,0.3)';
            ctx.strokeStyle = col;
            ctx.beginPath();
            for (let i = 0; i < nF; i++) {
                const x = pad.l + i * gap;
                const y = pad.t + h - (line.values[i] * h);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
        ctx.globalAlpha = 1;
        document.getElementById('parcoords-hint').textContent = `${d.lines.length} points across ${nF} features (normalised 0-1)`;
    } catch (e) { }
}

/* ===== Box Plot ===== */
async function renderBoxPlot() {
    const container = document.getElementById('boxplot-container');
    if (!lastLabels || !lastLabels.length) {
        container.innerHTML = '<p class="small-meta">Run a clustering algorithm first.</p>';
        return;
    }
    try {
        const r = await fetch('/box_plot_data', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastLabels }) });
        const d = await r.json();
        let html = '';
        d.clusters.forEach(c => {
            const col = CC[c.id % CC.length];
            html += `<div style="margin-bottom:1.2rem;"><h4 style="color:${col}; margin:0 0 0.4rem;">Cluster ${c.id} (${c.count} pts)</h4>`;
            d.features.forEach((fname, fi) => {
                const s = c.feature_stats[fi];
                const range = s.max - s.min;
                const w = range > 0 ? range : 1;
                const pctQ1 = ((s.q1 - s.min) / w * 100);
                const pctMed = ((s.median - s.min) / w * 100);
                const pctQ3 = ((s.q3 - s.min) / w * 100);
                const boxLeft = pctQ1, boxWidth = pctQ3 - pctQ1;
                html += `<div style="display:flex; align-items:center; gap:0.5rem; margin:0.2rem 0; font-size:0.75rem;">
                    <span style="width:28px; color:${tc('--text-faint')}; text-align:right;">${fname}</span>
                    <div style="flex:1; height:18px; position:relative; background:${tc('--glass-border') || '#333'}; border-radius:3px; overflow:hidden;">
                        <div style="position:absolute; left:${boxLeft}%; width:${boxWidth}%; height:100%; background:${col}; opacity:0.5; border-radius:2px;"></div>
                        <div style="position:absolute; left:${pctMed}%; width:2px; height:100%; background:${col};"></div>
                    </div>
                    <span style="width:85px; font-family:monospace; color:${tc('--text-faint')}; font-size:0.65rem;">${s.min.toFixed(1)} → ${s.max.toFixed(1)}</span>
                </div>`;
            });
            html += '</div>';
        });
        container.innerHTML = html;
        document.getElementById('boxplot-hint').textContent = `Box plots: Q1–Q3 range with median line per feature per cluster`;
    } catch (e) { container.innerHTML = '<p class="small-meta">Error loading box plots.</p>'; }
}


function updateStatusPanel(phase, desc, inertia, iter) {
    document.getElementById('phase-badge').textContent = phase.toUpperCase();
    document.getElementById('phase-badge').className = `badge ${phase}`;
    document.getElementById('step-desc').textContent = desc;
    if (currentAlgorithm === 'kmeans') { document.getElementById('inertia-val').textContent = inertia.toFixed(2); document.getElementById('iteration').textContent = iter; }
}

/* ===== Reset ===== */
function reset() {
    isRunning = false; historyTrails = []; iteration = 0; currentPhase = 'init';
    stateHistory = [];
    dbscanFrames = []; dbscanFrameIdx = 0; currentDbscanFrame = null;
    gmmFrames = []; gmmFrameIdx = 0; currentGmmFrame = null;
    hierDendrogramData = null; lastLabels = []; lastOutliers = [];
    document.getElementById('silhouette-val').textContent = '—'; document.getElementById('silhouette-val').className = 'mono';
    document.getElementById('db-val').textContent = '—'; document.getElementById('db-val').className = 'mono';
    if (currentAlgorithm === 'kmeans') {
        if (initMode === 'manual') { centroids = []; updateStatusPanel('ready', `Click to place ${k} centroids.`, 0, 0); }
        else { centroids = null; updateStatusPanel('ready', 'Select data and click Start.', 0, 0); }
    } else if (currentAlgorithm === 'gmm') { centroids = null; updateStatusPanel('ready', 'Adjust and click Start.', 0, 0); document.getElementById('inertia-val').textContent = '0.00'; }
    else if (currentAlgorithm === 'hierarchical') { centroids = null; updateStatusPanel('ready', 'Click Start for hierarchy.', 0, 0); document.getElementById('inertia-val').textContent = '0'; }
    else { centroids = null; updateStatusPanel('ready', 'Adjust and click Start.', 0, 0); document.getElementById('inertia-val').textContent = '0'; }
    document.getElementById('runBtn').textContent = "Start Animation";
    updateChartData(points2d, centroids || [], []);
}

/* ===== Download Clustered CSV ===== */
function downloadCSV() {
    if (!points.length) return;
    let csv = 'X,Y,Z,Cluster_Label\n';
    for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const label = (lastLabels && lastLabels.length > i) ? lastLabels[i] : -1;
        csv += `${p[0]},${p[1]},${p.length > 2 ? p[2] : 0},${label}\n`;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'clustered_data.csv';
    a.click();
    URL.revokeObjectURL(url);
}

/* ===== Undo Step ===== */
function undoStep() {
    if (currentAlgorithm !== 'kmeans' || stateHistory.length === 0) return;
    if (isRunning) { isRunning = false; document.getElementById('runBtn').textContent = 'Resume'; }
    const prev = stateHistory.pop();
    centroids = prev.centroids;
    lastLabels = prev.labels;
    currentPhase = prev.phase;
    iteration = prev.iteration;
    lastOutliers = prev.outliers;
    if (historyTrails.length > 0) historyTrails.pop();
    if (centroids && lastLabels.length) {
        updateChartData(points2d, centroids, lastLabels);
        updateStatusPanel(currentPhase, `Undone to step ${iteration}.`, 0, iteration);
    } else {
        updateChartData(points2d, [], []);
        updateStatusPanel('ready', 'Undone to initial state.', 0, 0);
    }
}

/* ===== Comparison Mode ===== */
async function toggleCompareMode() {
    const container = document.getElementById('compare-results');
    if (container.style.display !== 'none') { container.style.display = 'none'; return; }
    if (!points.length) return;
    container.style.display = 'block';
    container.innerHTML = '<p class="small-meta">Running all algorithms...</p>';

    const results = [];

    // K-Means
    try {
        let c = null, l = [], ph = 'init';
        for (let i = 0; i < 30; i++) {
            const r = await fetch('/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, k, centroids: c, phase: ph }) });
            const d = await r.json();
            c = d.centroids; l = d.labels; ph = d.phase;
            if (d.converged) break;
        }
        const m = await (await fetch('/metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: l }) })).json();
        results.push({ name: 'K-Means', sil: m.silhouette, db: m.davies_bouldin });
    } catch (e) { results.push({ name: 'K-Means', sil: null, db: null }); }

    // DBSCAN
    try {
        const r = await fetch('/dbscan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, eps, min_samples: minSamples }) });
        const d = await r.json();
        const m = await (await fetch('/metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: d.labels }) })).json();
        results.push({ name: 'DBSCAN', sil: m.silhouette, db: m.davies_bouldin });
    } catch (e) { results.push({ name: 'DBSCAN', sil: null, db: null }); }

    // GMM
    try {
        const r = await fetch('/gmm_frames', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, n_components: k }) });
        const d = await r.json();
        const lastFrame = d.frames[d.frames.length - 1];
        const m = await (await fetch('/metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: lastFrame.labels }) })).json();
        results.push({ name: 'GMM', sil: m.silhouette, db: m.davies_bouldin });
    } catch (e) { results.push({ name: 'GMM', sil: null, db: null }); }

    // Hierarchical
    try {
        const r = await fetch('/hierarchical', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, n_clusters: k }) });
        const d = await r.json();
        const m = await (await fetch('/metrics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ points: points2d, labels: d.labels }) })).json();
        results.push({ name: 'Hierarchical', sil: m.silhouette, db: m.davies_bouldin });
    } catch (e) { results.push({ name: 'Hierarchical', sil: null, db: null }); }

    // Find best
    const best = results.reduce((a, b) => (a.sil || 0) > (b.sil || 0) ? a : b);

    let html = '<table class="stats-table"><thead><tr><th>Algorithm</th><th>Silhouette</th><th>DB Index</th></tr></thead><tbody>';
    results.forEach(r => {
        const isBest = r.name === best.name;
        html += `<tr style="${isBest ? 'background:rgba(0,206,201,0.1);' : ''}"><td>${r.name} ${isBest ? '🏆' : ''}</td><td class="mono">${r.sil !== null ? r.sil.toFixed(4) : '—'}</td><td class="mono">${r.db !== null ? r.db.toFixed(4) : '—'}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

/* ===== Interactive Tutorial ===== */
const TUTORIAL_STEPS = [
    { target: '.algo-toggle', text: 'Choose a clustering algorithm here: K-Means, DBSCAN, GMM, or Hierarchical. Each uses a different approach to find patterns in your data.' },
    { target: '.btn-group', text: 'Pick a dataset shape. Try "Moons" or "Circles" to see how different algorithms handle non-convex clusters!' },
    { target: '#kmeans-params', text: 'Adjust parameters like the number of clusters (K), initialization mode, and distance metric. Try "🔮 Suggest Best K" to let the algorithm recommend an optimal value.' },
    { target: '#runBtn', text: 'Click "Start Animation" to watch the algorithm converge step by step, or use "Next Step" for manual control. You can also undo steps!' },
    { target: '#clusterChart', text: 'This is your main visualization canvas. Points are colored by cluster assignment. Watch centroids move, Voronoi regions shift, and outliers get highlighted!' },
    { target: '#metrics-panel', text: 'Evaluation metrics update automatically. Silhouette Score measures cluster cohesion (higher = better). Davies-Bouldin Index measures separation (lower = better).' },
    { target: '#analysisBtn', text: 'Open the Analysis Dashboard to explore 8 different views: Elbow, Silhouette Plot, Cluster Stats, Feature Importance, Auto-K, PCA, Parallel Coordinates, and Box Plots.' }
];
let tutorialStep = 0;

function startTutorial() {
    tutorialStep = 0;
    document.getElementById('tutorial-overlay').style.display = 'block';
    showTutorialStep();
}

function endTutorial() {
    document.getElementById('tutorial-overlay').style.display = 'none';
}

function nextTutorialStep() {
    tutorialStep++;
    if (tutorialStep >= TUTORIAL_STEPS.length) { endTutorial(); return; }
    showTutorialStep();
}

function showTutorialStep() {
    const step = TUTORIAL_STEPS[tutorialStep];
    const el = document.querySelector(step.target);
    if (!el) { nextTutorialStep(); return; }

    const rect = el.getBoundingClientRect();
    const pad = 8;

    // Position spotlight
    const spotlight = document.getElementById('tutorial-spotlight');
    spotlight.style.left = (rect.left - pad) + 'px';
    spotlight.style.top = (rect.top - pad) + 'px';
    spotlight.style.width = (rect.width + pad * 2) + 'px';
    spotlight.style.height = (rect.height + pad * 2) + 'px';

    // Position tooltip
    const tooltip = document.getElementById('tutorial-tooltip');
    const tooltipLeft = Math.min(rect.left, window.innerWidth - 340);
    const tooltipTop = rect.bottom + 16;
    tooltip.style.left = Math.max(10, tooltipLeft) + 'px';
    tooltip.style.top = (tooltipTop > window.innerHeight - 120 ? rect.top - 120 : tooltipTop) + 'px';

    document.getElementById('tutorial-text').textContent = step.text;
    document.getElementById('tutorial-step-num').textContent = `${tutorialStep + 1} / ${TUTORIAL_STEPS.length}`;
    document.getElementById('tutorial-next-btn').textContent = tutorialStep === TUTORIAL_STEPS.length - 1 ? 'Finish' : 'Next';
}
