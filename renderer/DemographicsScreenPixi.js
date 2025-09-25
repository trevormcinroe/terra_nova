
import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";

function loadGoogleFonts() {
  if (!document.getElementById('gf-cinzel-crimson')) {
    const link = Object.assign(document.createElement('link'), {
      id  : 'gf-cinzel-crimson',
      rel : 'stylesheet',
      href: 'https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&' +
            'family=Crimson+Text:wght@400;600&display=swap',
    });
    document.head.appendChild(link);
  }
  return document.fonts.ready;          // resolves when fonts are usable
}

await loadGoogleFonts();
const headerStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 40,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 28,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyleSmall = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 18,
  fontWeight : 600,
  fill       : 0xffffff,
});
/* =========================================================
   DemographicsScreenRendererPixi — Time series graphs
   (2025‑08‑07 refactor)
   ========================================================= */

/* ───────────────────────── constants ───────────────────────── */
const METRICS = [
  { id: 'population', label: 'Population', color: 0x71C837 },
  { id: 'land',       label: 'Land',   color: 0x8B4513 },
  { id: 'literacy',   label: 'Literacy',    color: 0x9370DB },
  // yields-based metrics
  { id: 'food',       label: 'Food',        color: 0x32CD32 },
  { id: 'production', label: 'Production',  color: 0xFF8C00 },
  { id: 'gold',       label: 'Gold',        color: 0xFFD700 },
  { id: 'faith',      label: 'Faith',       color: 0xF0E68C },
  { id: 'culture',    label: 'Culture',     color: 0xDA70D6 },
  { id: 'science',    label: 'Science',     color: 0x00CED1 },
  { id: 'happiness',  label: 'Happiness',   color: 0xFF69B4 },
  { id: 'tourism',    label: 'Tourism',     color: 0x40E0D0 },
  { id: "treasury", label: "Treasury", color: 0x32CD32 },
  { id: "netHappiness", label: "Net Happiness", color: 0x32CD32 },
  { id: "goldenAgeTurns", label: "Golden Age Turns", color: 0x32CD32 },
];


const YIELD_METRIC_INDEX = {
  food: 0,
  production: 1,
  gold: 2,
  faith: 3,
  culture: 4,
  science: 5,
  happiness: 6,
  //tourism: 7
};


//const PLAYER_COLORS = [0x141B51, 0x66023C, 0x4A92D9, 0x71C837, 0xAAAAAA, 0x706f1f];
//const PLAYER_COLORS = ["#FF2A2F", "#FFD700", "#0A63BA", "#286B32", "#FFFFFF", "#91A1D6"]
const PLAYER_COLORS = constants.playerColorsScreens;
const GRAPH_LINE_WIDTH = 2;
const POINT_RADIUS = 3;
const MAX_X_LABELS = 8; // keep axis readable

/* ========================================================= */
export class DemographicsScreenRendererPixi {
  constructor(app) {
    /* ------------- scene graph ---------------- */
    this.app = app;
    this.stage = new PIXI.Container();
    this.app.stage.addChild(this.stage);

    /* ------------- state ---------------------- */
    this.currentMetric = 'population'; // Default metric
    this.turn = 0;
    this.maxTurn = 0;
    this.demographicsData = {}; // Raw data sources
    this.timeSeriesData = [];   // Processed data for active metric

    /* ------------- geometry ------------------- */
    const { width: W, height: H } = app.renderer;
    this.boxWidth = Math.min(W * 0.85, 1400);
    this.boxHeight = Math.min(H * 0.85, 800);
    this.boxX = (W - this.boxWidth) / 2;
    this.boxY = (H - this.boxHeight) / 2;

    // Graph area dimensions
    this.graphX = this.boxX + 100;
    this.graphY = this.boxY + 150;
    this.graphWidth = this.boxWidth - 200;
    this.graphHeight = this.boxHeight - 220;

    /* ------------- layers --------------------- */
    this.lBackground = new PIXI.Container();
    this.lGrid       = new PIXI.Graphics();
    this.lAxes       = new PIXI.Graphics();
    this.lLines      = new PIXI.Graphics();
    this.lPoints     = new PIXI.Container();
    this.lLabels     = new PIXI.Container();
    this.lUI         = new PIXI.Container();

    this.stage.addChild(
      this.lBackground,
      this.lGrid,
      this.lAxes,
      this.lLines,
      this.lPoints,
      this.lLabels,
      this.lUI
    );

    this.buildInterface();
  }

  /* =========================================================
       PUBLIC API
     ========================================================= */
  setDemographicsData(data) {
    this.demographicsData = data;
    this.processTimeSeriesData();
    this.redraw();
  }

  setTurn(turn) {
    this.turn = turn;
    this.processTimeSeriesData();
    this.redraw();
  }

  setMaxTurn(maxTurn) {
    this.maxTurn = maxTurn;
  }

  start() {
    this.stage.visible = true;
    this.redraw();
  }

  stop() {
    this.stage.visible = false;
  }

  /* =========================================================
       BUILD INTERFACE
     ========================================================= */
  buildInterface() {
    const bg = new PIXI.Graphics();
    bg.beginFill(0x0a0a0a, 0.95)
      .drawRect(this.boxX, this.boxY, this.boxWidth, this.boxHeight)
      .endFill();
    bg.lineStyle(3, 0x4a8a8a)
      .drawRect(this.boxX, this.boxY, this.boxWidth, this.boxHeight);
    this.lBackground.addChild(bg);

    this.titleText = new PIXI.Text('Demographics', headerStyle);
    this.titleText.anchor.set(0.5, 0);
    this.titleText.position.set(this.boxX + this.boxWidth / 2, this.boxY + 20);
    this.lBackground.addChild(this.titleText);

    this.createMetricSelector();

    //this.turnLabel = new PIXI.Text('Turn: 0', { fontSize: 18, fill: 0xffffff });
    //this.turnLabel.position.set(this.boxX + this.boxWidth - 100, this.boxY + 25);
    //this.lBackground.addChild(this.turnLabel);

    this.createLegend();

    this.drawStaticGrid();
    this.drawAxes();
  }

  /* =========================================================
       METRIC SELECTOR
     ========================================================= */
  createMetricSelector() {
    const selectorX = this.boxX + 30;
    const selectorY = this.boxY + 40;

    // Dropdown background
    const dropdownBg = new PIXI.Graphics();
    dropdownBg.beginFill(0x2a2a2a)
      .drawRect(0, 0, 200, 30)
      .endFill();
    dropdownBg.lineStyle(1, 0x4a8a8a)
      .drawRect(0, 0, 200, 30);
    dropdownBg.position.set(selectorX, selectorY);

    // Current selection text
    this.currentMetricText = new PIXI.Text(
      METRICS.find(m => m.id === this.currentMetric).label,
      detailStyle
    );
    this.currentMetricText.anchor.set(0, 0.5);
    this.currentMetricText.position.set(selectorX + 10, selectorY + 15);

    // Dropdown arrow
    const arrow = new PIXI.Text('▼', { fontSize: 14, fill: 0xffffff });
    arrow.anchor.set(1, 0.5);
    arrow.position.set(selectorX + 190, selectorY + 15);

    // Make dropdown interactive
    dropdownBg.eventMode = 'static';
    dropdownBg.cursor    = 'pointer';

    // Dropdown menu (initially hidden)
    this.dropdownMenu = new PIXI.Container();
    this.dropdownMenu.visible = false;
    this.dropdownMenu.position.set(selectorX, selectorY + 30);

    // Create menu items
    METRICS.forEach((metric, idx) => {
      const itemBg = new PIXI.Graphics();
      itemBg.beginFill(0x2a2a2a)
        .drawRect(0, idx * 25, 200, 25)
        .endFill();
      itemBg.lineStyle(1, 0x4a8a8a)
        .drawRect(0, idx * 25, 200, 25);

      const itemText = new PIXI.Text(metric.label, { fontSize: 14, fill: 0xffffff });
      itemText.position.set(10, idx * 25 + 5);

      itemBg.eventMode = 'static';
      itemBg.cursor    = 'pointer';

      // Select metric (no hover effects to minimise listeners)
      itemBg.on('pointerdown', () => {
        this.currentMetric      = metric.id;
        this.currentMetricText.text = metric.label;
        this.dropdownMenu.visible   = false;
        this.processTimeSeriesData();
        this.redraw();
      });

      this.dropdownMenu.addChild(itemBg, itemText);
    });

    // Toggle dropdown
    dropdownBg.on('pointerdown', () => {
      this.dropdownMenu.visible = !this.dropdownMenu.visible;
    });

    this.lUI.addChild(dropdownBg, this.currentMetricText, arrow, this.dropdownMenu);
  }

  /* =========================================================
       LEGEND
     ========================================================= */
  createLegend() {
     const legendX = this.boxX + (this.boxWidth / 5) + 15;
     const legendY = this.boxY + 100;

     this.legendContainer = new PIXI.Container();
     this.legendContainer.position.set(legendX, legendY);
     this.lUI.addChild(this.legendContainer);
   }

   updateLegend() {
     this.legendContainer.removeChildren();
     if (!this.timeSeriesData?.length) return;

     const players = new Set();
     this.timeSeriesData.forEach(turnData => {
       turnData?.forEach((v, pIdx) => {
         if (v !== null && v !== undefined) players.add(pIdx);
       });
     });

     let xOffset = 0;
     players.forEach(pIdx => {
       const box = new PIXI.Graphics();
       box.beginFill(PLAYER_COLORS[pIdx])
          .drawRect(xOffset, 2, 25, 25)
          .endFill();

       const label = new PIXI.Text(`Player ${pIdx + 1}`, detailStyle);
       label.position.set(xOffset + 30, 0);

       this.legendContainer.addChild(box, label);
       xOffset += 138;  // Adjust spacing as needed (was 20 for vertical)
     });
   }
  /* =========================================================
       GRID & AXES
     ========================================================= */
  drawStaticGrid() {
    // Single initial draw; dynamic redraw will clear & repaint
    const g = this.lGrid;
    g.clear();
    g.beginFill(0x1a1a1a, 0.5)
      .drawRect(this.graphX, this.graphY, this.graphWidth, this.graphHeight)
      .endFill();
  }

  drawDynamicGrid(xStep) {
    const g = this.lGrid;
    g.clear();

    // Background
    g.beginFill(0x1a1a1a, 0.5)
      .drawRect(this.graphX, this.graphY, this.graphWidth, this.graphHeight)
      .endFill();

    // Vertical grid lines aligned with X‑axis labels
    g.lineStyle(1, 0x3a3a3a, 0.5);
    for (let i = 0; i < this.timeSeriesData.length; i += xStep) {
      const x = this.graphX + (i / Math.max(1, this.timeSeriesData.length - 1)) * this.graphWidth;
      g.moveTo(x, this.graphY);
      g.lineTo(x, this.graphY + this.graphHeight);
      g.stroke();
    }
    // Ensure last line at the end
    if ((this.timeSeriesData.length - 1) % xStep !== 0) {
      const x = this.graphX + this.graphWidth;
      g.moveTo(x, this.graphY);
      g.lineTo(x, this.graphY + this.graphHeight);
      g.stroke();
    }

    // Horizontal grid lines (fixed 10 divisions)
    g.lineStyle(1, 0x3a3a3a, 0.5);
    for (let i = 1; i < 10; i++) {
      const y = this.graphY + (i / 10) * this.graphHeight;
      g.moveTo(this.graphX, y);
      g.lineTo(this.graphX + this.graphWidth, y);
      g.stroke();
    }
  }

  drawAxes() {
    const g = this.lAxes;
    g.clear();

    // Axes
    g.lineStyle(3, 0x6a6a6a);
    g.moveTo(this.graphX, this.graphY);
    g.lineTo(this.graphX, this.graphY + this.graphHeight).stroke();
    g.moveTo(this.graphX, this.graphY + this.graphHeight);
    g.lineTo(this.graphX + this.graphWidth, this.graphY + this.graphHeight).stroke();

    // Border
    g.lineStyle(2, 0x4a4a4a);
    g.drawRect(this.graphX, this.graphY, this.graphWidth, this.graphHeight);

    // X‑axis label
    this.xAxisStaticLabel = new PIXI.Text('Turn', detailStyle);
    this.xAxisStaticLabel.anchor.set(0.5, 0);
    this.xAxisStaticLabel.position.set(
      this.graphX + this.graphWidth / 2,
      this.graphY + this.graphHeight + 20
    );
    this.lLabels.addChild(this.xAxisStaticLabel);

    // Y‑axis label (dynamic text only)
    this.yAxisLabel = new PIXI.Text('Value', detailStyle);
    this.yAxisLabel.anchor.set(0.5, 1);
    this.yAxisLabel.rotation = -Math.PI / 2;
    this.yAxisLabel.position.set(this.graphX - 40, this.graphY + this.graphHeight / 2);
    this.lLabels.addChild(this.yAxisLabel);
  }

  /* =========================================================
       DATA PROCESSING (unchanged)
     ========================================================= */
  processTimeSeriesData() {
    this.timeSeriesData = [];

    // Prioritise yield‑based aggregation when available
    if (YIELD_METRIC_INDEX.hasOwnProperty(this.currentMetric) && this.demographicsData.yields) {
      this.aggregateYieldData(this.demographicsData.yields, YIELD_METRIC_INDEX[this.currentMetric]);
      return;
    }

    const src = this.demographicsData[this.currentMetric];
    if (!src) return;

    switch (this.currentMetric) {
      case 'land':       this.aggregateLandData(src);       break;
      case 'population': this.aggregatePopulationData(src); break;
      case 'literacy': this.aggregateLiteracyData(src); break;
      case 'tourism':   this.aggregateTourismData(src);   break;
      // fallback for other per‑player arrays
      default:           this.aggregateSimpleData(src);
    }
  }

  /* ── Yield aggregation ────────────────────── */
  aggregateYieldData(yieldArray, yieldIdx) {
    // yieldArray shape: [turn][player][city][8]
    for (let t = 0; t <= this.turn && t < yieldArray.length; t++) {
      const turnSlice = yieldArray[t];
      if (!turnSlice) continue;

      const playerTotals = turnSlice.map(playerSlice => {
        if (!playerSlice) return 0;
        return playerSlice.reduce((sum, citySlice) => {
          const val = citySlice?.[yieldIdx] || 0;
          return sum + val;
        }, 0);
      });

      this.timeSeriesData.push(playerTotals);
    }
  }

  /* ── Tourism aggregation ─────────────────── */
  aggregateTourismData(arr) {
    // shape: [turn][player][targetPlayer]
    for (let t = 0; t <= this.turn && t < arr.length; t++) {
      const turnSlice = arr[t] || [];
      const totals = turnSlice.map((exerted, pIdx) => {
        if (!Array.isArray(exerted)) return 0;
        return exerted.reduce((sum, val, kIdx) => sum + (kIdx === pIdx ? 0 : (val || 0)), 0);
      });
      this.timeSeriesData.push(totals);
    }
  }
  
  /* ── Literacy aggregation ─────────────────── */
  aggregateLiteracyData(litArray) {
    // litArray shape: [turn][player][tech] (0/1)
    for (let t = 0; t <= this.turn && t < litArray.length; t++) {
      const turnSlice = litArray[t] || [];
      const playerTotals = turnSlice.map(playerTechs => Array.isArray(playerTechs) ? playerTechs.reduce((s, v) => s + (v || 0), 0) : 0);
      this.timeSeriesData.push(playerTotals);
    }
  }

  aggregateLandData(landData) {
    for (let t = 0; t <= this.turn && t < landData.length; t++) {
      const turnData = landData[t];
      if (!turnData) continue;
      const counts = new Array(6).fill(0);
      for (let y = 0; y < turnData.length; y++) {
        for (let x = 0; x < (turnData[y]?.length || 0); x++) {
          const owner = turnData[y][x];
          if (owner > 0 && owner <= 6) counts[owner - 1]++;
        }
      }
      this.timeSeriesData.push(counts);
    }
  }

  aggregatePopulationData(popData) {
    for (let t = 0; t <= this.turn && t < popData.length; t++) {
      const turnData = popData[t];
      if (!turnData) continue;
      const playerTotals = turnData.map(entry => Array.isArray(entry)
        ? entry.reduce((s, c) => s + (c || 0), 0)
        : (entry || 0));
      this.timeSeriesData.push(playerTotals);
    }
  }

  aggregateSimpleData(data) {
    for (let t = 0; t <= this.turn && t < data.length; t++) {
      this.timeSeriesData.push(data[t] || []);
    }
  }

  /* =========================================================
       DRAWING
     ========================================================= */
  redraw() {
    if (!this.timeSeriesData.length) return;

    /* ----- clean layers ----- */
    this.lLines.clear();
    this.lPoints.removeChildren();
    // Labels: keep static x/y labels only
    this.lLabels.removeChildren();
    this.lLabels.addChild(this.xAxisStaticLabel, this.yAxisLabel);

    /* ----- update labels ----- */
    //this.turnLabel.text    = `Turn: ${this.turn}`;
    const metricObj        = METRICS.find(m => m.id === this.currentMetric);
    this.yAxisLabel.text   = metricObj?.label || 'Value';

    /* ----- calc value range (min always 0) ----- */
    //let maxVal = -Infinity;
    //let numPlayers = 0;
    //this.timeSeriesData.forEach(td => {
    //  numPlayers = Math.max(numPlayers, td.length);
    //  td.forEach(v => { if (v !== null && v !== undefined) maxVal = Math.max(maxVal, v); });
    //});
    //if (maxVal === -Infinity) maxVal = 100;
    //const padding = maxVal * 0.1 || 1;
    //const minVal  = 0;
    //maxVal += padding;
let minVal = +Infinity;
let maxVal = -Infinity;
let numPlayers = 0;

this.timeSeriesData.forEach(td => {
  numPlayers = Math.max(numPlayers, td.length);
  td.forEach(v => {
    if (v !== null && v !== undefined) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  });
});

// Fallback if no finite data
if (!isFinite(minVal) || !isFinite(maxVal)) {
  minVal = 0;
  maxVal = 100;
}

// Add padding; handle flat series
let span = maxVal - minVal;
if (span === 0) {
  const pad = (Math.abs(maxVal) || 1) * 0.1; // give a little breathing room
  minVal -= pad;
  maxVal += pad;
  span = maxVal - minVal;
} else {
  const pad = span * 0.1;
  minVal -= pad;
  maxVal += pad;
  span = maxVal - minVal;
}

    /* ----- X‑axis grid/labels step ----- */
    const numLabels = Math.min(MAX_X_LABELS, this.timeSeriesData.length);
    const xStep     = Math.max(1, Math.floor(this.timeSeriesData.length / numLabels));

    /* ----- redraw grid aligned to step ----- */
    this.drawDynamicGrid(xStep);

    /* ----- plot lines & (non‑interactive) points ----- */
    for (let pIdx = 0; pIdx < numPlayers; pIdx++) {
      const pts = [];
      this.timeSeriesData.forEach((td, tIdx) => {
        const val = td[pIdx];
        if (val !== null && val !== undefined) {
          const x = this.graphX + (tIdx / Math.max(1, this.timeSeriesData.length - 1)) * this.graphWidth;
          const y = this.graphY + this.graphHeight - ((val - minVal) / (maxVal - minVal)) * this.graphHeight;
          pts.push({ x, y });
        }
      });
      if (!pts.length) continue;

      // Line
      if (pts.length > 1) {
        this.lLines.lineStyle(GRAPH_LINE_WIDTH, PLAYER_COLORS[pIdx], 0.8);
        this.lLines.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) this.lLines.lineTo(pts[i].x, pts[i].y);
        this.lLines.stroke();
      }

      // Points (render‑only, no events)
      pts.forEach(pt => {
        const c = new PIXI.Graphics();
        c.beginFill(PLAYER_COLORS[pIdx]).drawCircle(0, 0, POINT_RADIUS).endFill();
        c.position.set(pt.x, pt.y);
        this.lPoints.addChild(c);
      });
    }

    /* ----- axis labels ----- */
    this.drawAxisLabels(xStep, minVal, maxVal);

    /* ----- legend ----- */
    this.updateLegend();
  }

  drawAxisLabels(xStep, minVal, maxVal) {
    // X‑axis (turns)
    for (let i = 0; i < this.timeSeriesData.length; i += xStep) {
      const x = this.graphX + (i / Math.max(1, this.timeSeriesData.length - 1)) * this.graphWidth;
      const lbl = new PIXI.Text(i.toString(), detailStyleSmall);
      lbl.anchor.set(0.5, 0);
      lbl.position.set(x, this.graphY + this.graphHeight + 5);
      this.lLabels.addChild(lbl);
    }
    // Ensure last label is shown
    if ((this.timeSeriesData.length - 1) % xStep !== 0) {
      const x = this.graphX + this.graphWidth;
      const lbl = new PIXI.Text((this.timeSeriesData.length - 1).toString(), detailStyleSmall);
      lbl.anchor.set(0.5, 0);
      lbl.position.set(x, this.graphY + this.graphHeight + 5);
      this.lLabels.addChild(lbl);
    }

    // Y‑axis (values) — 5 labels incl. 0 baseline
    const yLabels = 5;
    for (let i = 0; i <= yLabels; i++) {
      const value = minVal + (i / yLabels) * (maxVal - minVal);
      const y = this.graphY + this.graphHeight - (i / yLabels) * this.graphHeight;
      const lbl = new PIXI.Text(Math.round(value).toString(), detailStyleSmall);
      lbl.anchor.set(1, 0.5);
      lbl.position.set(this.graphX - 10, y);
      this.lLabels.addChild(lbl);
    }
  }
}
