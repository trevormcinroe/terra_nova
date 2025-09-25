import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";

let resTextures = await constants.loadResourceTextures();
/* =========================================================
   TradeScreenRendererPixi — Trade overview screen
   ========================================================= */

/* ───────────────────────── Load Fonts ───────────────────────── */
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
  return document.fonts.ready;
}

await loadGoogleFonts();

/* ───────────────────────── Text Styles ───────────────────────── */
const headerStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 26,
  fontWeight : 600,
  fill       : 0xffffff,
  resolution: 4,
});

const tabStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 18,
  fontWeight : 400,
  fill       : 0xd4af37,
  resolution: 4,
});

const tabStyleActive = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 18,
  fontWeight : 600,
  fill       : 0xffd700,
  resolution: 4,
});

const contentHeaderStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 22,
  fontWeight : 600,
  fill       : 0xffd700,
  resolution: 4,
});

const descStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 18,
  fontWeight : 400,
  fill       : 0xffffff,
  wordWrap   : true,
  lineHeight : 26,
});

const requirementStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 600,
  fill       : 0xc9b037,
  lineHeight : 24,
  resolution: 4,
});

const statusStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 400,
  fill       : 0xa89968,
  lineHeight : 22,
  resolution: 4,
});

/* ───────────────────────── Constants ───────────────────────── */
const TAB_TYPES = ['Trade Routes', 'Trade Deals', 'Resources'];
const TAB_HEIGHT = 50;

// Civilization-style color palette
const COLORS = {
  bg: 0x1a1612,
  stoneBg: 0x2c2416,
  stoneLight: 0x3d3426,
  parchment: 0x3d3426,
  goldBorder: 0xc9b037,
  goldBright: 0xffd700,
  goldDark: 0x8b7355,
  headerBg: 0x81ADC8,
  tabActive: 0x2c2416,
  tabInactive: 0x1a1612,
  divider: 0x4a3f28,
  shadow: 0x000000,
  success: 0x16a34a,
  pending: 0xca8a04,
  failed: 0xBB4430,
  trade: 0x3498db,
};

/* ========================================================= */
export class TradeScreenRendererPixi {
  constructor(app) {
    /* ------------- scene graph ---------------- */
    this.app = app;
    this.stage = new PIXI.Container();
    this.app.stage.addChild(this.stage);
    
    // Ensure transparent background
    this.app.renderer.background.alpha = 0;

    /* ------------- state ---------------------- */
    this.activeTab = 0;  // Default to Trade Routes
    this.turn = 0;
    this.tradeData = null;
    this.dealData = null;
    this.tradeRouteRemaining = [];

    this.resourcesScrollY = 0;

    /* ------------- geometry ------------------- */
    const { width: W, height: H } = app.renderer;
    this.boxWidth = Math.min(W * 0.85, 1200);
    this.boxHeight = Math.min(H * 0.85, 700);
    this.boxX = (W - this.boxWidth) / 2;
    this.boxY = (H - this.boxHeight) / 2 - 50;

    /* ------------- layers --------------------- */
    this.lBackground = new PIXI.Container();
    this.lMain = new PIXI.Container();
    this.lTabs = new PIXI.Container();
    this.lContent = new PIXI.Container();
    this.stage.addChild(this.lBackground, this.lMain, this.lTabs, this.lContent);

    /* ------------- interaction ---------------- */
    this.tabButtons = [];
    this.routesScrollY = 0;
    this.dealsScrollY = 0;

    this.buildInterface();
  }
  // Precompute (T, 6, R) by summing cities for each turn, once.
  precomputeResourceTotals() {
    const data = this.resourcesOwned;
    const dataAdj = this.tradeResourceAdj;
    if (!data || !data.length) {
      this._resTotals = null;
      this._resTotalsR = 0;
      return;
    }

    const T = data.length;
    const P = Math.min(6, data[0]?.length || 0);

    // Infer R (num_resources) by finding the first non-empty city row
    let R = 0;
    outer:
    for (let t = 0; t < T; t++) {
      const sliceT = data[t];
      if (!sliceT) continue;
      for (let p = 0; p < P; p++) {
        const cities = sliceT[p];
        if (!cities) continue;
        for (let c = 0; c < cities.length; c++) {
          const row = cities[c];
          if (row && row.length) { R = row.length; break outer; }
        }
      }
    }
    if (!R) { this._resTotals = null; this._resTotalsR = 0; return; }

    // Allocate result
    const totals = new Array(T);

    // Sum cities → (6, R) per turn
    for (let t = 0; t < T; t++) {
      const sliceT = data[t] || [];
      const turnTotals = new Array(P);
      for (let p = 0; p < P; p++) {
        const cities = sliceT[p] || [];
        const sums = new Array(R).fill(0);
        for (let c = 0; c < cities.length; c++) {
          const row = cities[c] || [];
          // Add row into sums
          for (let r = 0; r < R; r++) {
            if (dataAdj[t][p][r] != 0) {
              sums[r] += row[r] + dataAdj[t][p][r] / cities.length;
            } else {
              sums[r] += row[r] || 0;
            }
          }
        }
        turnTotals[p] = sums;
      }
      totals[t] = turnTotals;
    }

    this._resTotals = totals; // (T, 6, R)
    this._resTotalsR = R;
  }

  // Lightweight accessor
  getResourceTotalsForTurn(turn) {
    return this._resTotals?.[turn] || null; // (6, R) or null
  }

  /* =========================================================
       PUBLIC API
     ========================================================= */
  setTurn(turn) {
    this.turn = turn; 
    this.redrawContent();
  }
  
  setData(tradeLedger, tradeLengthLedger, tradeGPTAdj, tradeResourceAdj, unitsMilitary, unitsTradePlayerTo, unitsTradeCityTo, unitsTradeCityFrom,  unitsTradeYields, unitsEngaged, resourcesOwned) {
    /* ------------- data ----------------------- */
    this.tradeLedger = tradeLedger;
    this.tradeLengthLedger = tradeLengthLedger;
    this.tradeGPTAdj = tradeGPTAdj;
    this.tradeResourceAdj = tradeResourceAdj;
    this.unitsMilitary = unitsMilitary;
    this.unitsTradePlayerTo = unitsTradePlayerTo;
    this.unitsTradeCityTo = unitsTradeCityTo;
    this.unitsTradeCityFrom = unitsTradeCityFrom;
    this.unitsTradeYields = unitsTradeYields;
    this.unitsEngaged = unitsEngaged;
    this.resourcesOwned = resourcesOwned;
    this.precomputeResourceTotals();
  }
  //setData(tradeRoutes, tradeDeals, resourceData) {
  //  this.tradeRoutes = tradeRoutes;
  //  this.tradeDeals = tradeDeals;
  //  this.resourceData = resourceData;
  //}
  
  start() { 
    this.stage.visible = true;
    this.redrawContent(); 
  }
  
  stop() { 
    this.stage.visible = false;
  }


  /* =========================================================
       BUILD INTERFACE
     ========================================================= */
  buildInterface() {
    /* ── Background with stone texture effect ─────── */
    const bgGraphics = new PIXI.Graphics();
    
    // Dark background
    bgGraphics.beginFill(COLORS.bg, 0.95);
    bgGraphics.drawRect(this.boxX - 10, this.boxY - 10, this.boxWidth + 20, this.boxHeight + 20);
    bgGraphics.endFill();
    
    // Add stone texture effect
    for (let i = 0; i < 30; i++) {
      const x = this.boxX + Math.random() * this.boxWidth;
      const y = this.boxY + Math.random() * this.boxHeight;
      const w = 50 + Math.random() * 100;
      const h = 50 + Math.random() * 100;
      bgGraphics.beginFill(COLORS.stoneBg, Math.random() * 0.1);
      bgGraphics.drawRect(x, y, w, h);
      bgGraphics.endFill();
    }
    
    this.lBackground.addChild(bgGraphics);

    /* ── Main container ─────────────────────────── */
    const mainBox = new PIXI.Graphics();
    
    // Drop shadow
    mainBox.beginFill(COLORS.shadow, 0.5);
    mainBox.drawRect(this.boxX + 3, this.boxY + 3, this.boxWidth, this.boxHeight);
    mainBox.endFill();
    
    // Main panel - parchment style
    mainBox.beginFill(COLORS.parchment);
    mainBox.drawRect(this.boxX, this.boxY, this.boxWidth, this.boxHeight);
    mainBox.endFill();
    
    // Add texture variation
    for (let t = 0; t < 5; t++) {
      mainBox.beginFill(COLORS.stoneLight, 0.1);
      mainBox.drawRect(
        this.boxX + Math.random() * this.boxWidth * 0.8,
        this.boxY + Math.random() * this.boxHeight * 0.8,
        30 + Math.random() * 60,
        30 + Math.random() * 60
      );
      mainBox.endFill();
    }
    
    // Ornate border
    mainBox.lineStyle(3, COLORS.goldDark);
    mainBox.drawRect(this.boxX, this.boxY, this.boxWidth, this.boxHeight);
    mainBox.stroke();
    
    mainBox.lineStyle(1, COLORS.goldBright);
    mainBox.drawRect(this.boxX + 4, this.boxY + 4, this.boxWidth - 8, this.boxHeight - 8);
    mainBox.stroke();
    
    // Corner ornaments
    const cornerSize = 12;
    mainBox.beginFill(COLORS.goldBright);
    // Top left
    mainBox.drawPolygon([
      this.boxX, this.boxY,
      this.boxX + cornerSize * 2, this.boxY,
      this.boxX, this.boxY + cornerSize * 2
    ]);
    // Top right
    mainBox.drawPolygon([
      this.boxX + this.boxWidth, this.boxY,
      this.boxX + this.boxWidth - cornerSize * 2, this.boxY,
      this.boxX + this.boxWidth, this.boxY + cornerSize * 2
    ]);
    // Bottom left
    mainBox.drawPolygon([
      this.boxX, this.boxY + this.boxHeight,
      this.boxX + cornerSize * 2, this.boxY + this.boxHeight,
      this.boxX, this.boxY + this.boxHeight - cornerSize * 2
    ]);
    // Bottom right
    mainBox.drawPolygon([
      this.boxX + this.boxWidth, this.boxY + this.boxHeight,
      this.boxX + this.boxWidth - cornerSize * 2, this.boxY + this.boxHeight,
      this.boxX + this.boxWidth, this.boxY + this.boxHeight - cornerSize * 2
    ]);
    mainBox.endFill();
    
    this.lMain.addChild(mainBox);

    /* ── Tab header section ────────────────────── */
    const tabBg = new PIXI.Graphics();
    tabBg.beginFill(COLORS.headerBg);
    tabBg.drawRect(this.boxX, this.boxY, this.boxWidth, TAB_HEIGHT + 10);
    tabBg.endFill();
    
    // Gold trim on bottom
    tabBg.beginFill(COLORS.goldBorder);
    tabBg.drawRect(this.boxX, this.boxY + TAB_HEIGHT + 7, this.boxWidth, 3);
    tabBg.endFill();
    
    this.lMain.addChild(tabBg);

    /* ── Create tabs ──────────────────────────── */
    const tabWidth = this.boxWidth / TAB_TYPES.length;
    
    function centerCrisp(text, cx, cy, offX = 0, offY = 0) {
      text.roundPixels = true;
      text.anchor.set(0.5, 0.5);
      // keep resolution in sync with renderer to avoid internal scaling blur
      if (text.resolution !== this.app.renderer.resolution) {
        text.resolution = this.app.renderer.resolution;
      }
      const w = Math.round(text.width);
      const h = Math.round(text.height);
      const oddW = (w & 1) === 1;
      const oddH = (h & 1) === 1;
      const baseX = Math.round(cx) + (oddW ? 0.5 : 0);
      const baseY = Math.round(cy) + (oddH ? 0.5 : 0);
      text.position.set(baseX + offX, baseY + offY); // offX/offY must be integers
    }

    const SHADOW_OX = 1, SHADOW_OY = 1;

    TAB_TYPES.forEach((type, idx) => {
      const tabContainer = new PIXI.Container();

      const tabButton = new PIXI.Graphics();
      this.drawTab(tabButton, idx, idx === this.activeTab);
      tabButton.eventMode = 'static';
      tabButton.cursor = 'pointer';

      const cx = Math.round(this.boxX + idx * tabWidth + tabWidth / 2);
      const cy = Math.round(this.boxY + (TAB_HEIGHT + 10) / 2);

      const baseStyle = idx === this.activeTab ? tabStyleActive : tabStyle;

      const label = new PIXI.Text(type.toUpperCase(), baseStyle);
      const shadowLabel = new PIXI.Text(type.toUpperCase(), baseStyle);
      shadowLabel.tint = 0x000000;
      shadowLabel.alpha = 0.12;

      centerCrisp.call(this, label, cx, cy);
      centerCrisp.call(this, shadowLabel, cx, cy, SHADOW_OX, SHADOW_OY);

      this.tabButtons[idx] = { bg: tabButton, label, shadowLabel, type, index: idx, container: tabContainer };

      tabButton.on('pointerover', () => {
        if (idx !== this.activeTab) {
          this.drawTab(tabButton, idx, false, true);
          label.style = tabStyleActive;
          shadowLabel.style = tabStyleActive;
          label.alpha = 0.85;
          centerCrisp.call(this, label, cx, cy);
          centerCrisp.call(this, shadowLabel, cx, cy, SHADOW_OX, SHADOW_OY);
        }
      });

      tabButton.on('pointerout', () => {
        if (idx !== this.activeTab) {
          this.drawTab(tabButton, idx, false);
          label.style = tabStyle;
          shadowLabel.style = tabStyle;
          label.alpha = 1.0;
          centerCrisp.call(this, label, cx, cy);
          centerCrisp.call(this, shadowLabel, cx, cy, SHADOW_OX, SHADOW_OY);
        }
      });

      tabButton.on('pointerdown', () => this.switchTab(idx));

      tabContainer.addChild(tabButton, shadowLabel, label);
      this.lTabs.addChild(tabContainer);
    });

    /* ── Content area decorations ─────────────── */
    const contentDecor = new PIXI.Graphics();
    
    const flourishY = this.boxY + TAB_HEIGHT + 25;
    contentDecor.lineStyle(2, COLORS.goldDark, 0.5);
    contentDecor.moveTo(this.boxX + this.boxWidth / 2 - 100, flourishY);
    contentDecor.lineTo(this.boxX + this.boxWidth / 2 - 20, flourishY);
    contentDecor.stroke();
    contentDecor.moveTo(this.boxX + this.boxWidth / 2 + 20, flourishY);
    contentDecor.lineTo(this.boxX + this.boxWidth / 2 + 100, flourishY);
    contentDecor.stroke();
    
    // Center diamond
    contentDecor.beginFill(COLORS.goldDark);
    contentDecor.drawPolygon([
      this.boxX + this.boxWidth / 2, flourishY - 4,
      this.boxX + this.boxWidth / 2 - 6, flourishY,
      this.boxX + this.boxWidth / 2, flourishY + 4,
      this.boxX + this.boxWidth / 2 + 6, flourishY
    ]);
    contentDecor.endFill();
    
    this.lMain.addChild(contentDecor);
  }

  /* =========================================================
       TAB DRAWING
     ========================================================= */
  drawTab(graphics, index, isActive, isHover = false) {
    const tabWidth = Math.round(this.boxWidth / TAB_TYPES.length);
    const x = Math.round(this.boxX + index * tabWidth);
    const y = Math.round(this.boxY);
    const h = TAB_HEIGHT + 10;

    graphics.clear();

    if (isActive) {
      graphics.beginFill(COLORS.headerBg, 1);
      graphics.drawRect(x, y, tabWidth, h);
      graphics.endFill();

      graphics.beginFill(COLORS.goldBright, 0.3);
      graphics.drawRect(x, y, tabWidth, 3);
      graphics.endFill();

      if (index > 0) {
        graphics.lineStyle(1, COLORS.goldDark);
        graphics.moveTo(x, y + 5);
        graphics.lineTo(x, y + h - 5);
        graphics.stroke();
      }
      if (index < TAB_TYPES.length - 1) {
        graphics.lineStyle(1, COLORS.goldDark);
        graphics.moveTo(x + tabWidth, y + 5);
        graphics.lineTo(x + tabWidth, y + h - 5);
        graphics.stroke();
      }
    } else {
      graphics.beginFill(COLORS.tabInactive, 1.0);
      graphics.drawRect(x, y, tabWidth, h);
      graphics.endFill();

      if (isHover) {
        graphics.beginFill(COLORS.stoneLight, 0.08);
        graphics.drawRect(x, y, tabWidth, h);
        graphics.endFill();
      }

      if (index < TAB_TYPES.length - 1) {
        graphics.lineStyle(1, COLORS.divider, 0.35);
        graphics.moveTo(x + tabWidth, y + 10);
        graphics.lineTo(x + tabWidth, y + h - 10);
        graphics.stroke();
      }
    }

    // Ensure whole area is clickable
    graphics.beginFill(0x000000, 0.001);
    graphics.drawRect(x, y, tabWidth, h);
    graphics.endFill();
  }

  /* =========================================================
       TAB SWITCHING
     ========================================================= */
  switchTab(index) {
    if (index === this.activeTab) return;
    
    const prevTab = this.tabButtons[this.activeTab];
    this.drawTab(prevTab.bg, this.activeTab, false);
    prevTab.label.style = tabStyle;
    prevTab.shadowLabel.style = tabStyle;
    
    this.activeTab = index;
    const newTab = this.tabButtons[index];
    this.drawTab(newTab.bg, index, true);
    newTab.label.style = tabStyleActive;
    newTab.shadowLabel.style = tabStyleActive;
    
    this.redrawContent();
  }

  /* =========================================================
       CONTENT RENDERING
     ========================================================= */
  redrawContent() {
    this.lContent.removeChildren();
    
    const tabType = TAB_TYPES[this.activeTab];
    
    // Content header
    const headerY = this.boxY + TAB_HEIGHT + 45;

    function centerCrispXTop(text, centerX, topY) {
      text.roundPixels = true;
      text.anchor.set(0.5, 0); // center horizontally, top vertically
      // match renderer resolution (prevents internal resampling)
      if (text.resolution !== this.app.renderer.resolution) {
        text.resolution = this.app.renderer.resolution;
      }
      const w = Math.round(text.width);
      const oddW = (w & 1) === 1;
      text.position.set(Math.round(centerX) + (oddW ? 0.5 : 0), Math.round(topY));
    }
    
    const contentHeader = new PIXI.Text(
      `${tabType.toUpperCase()} · TURN ${this.turn}`,
      contentHeaderStyle
    );
    //centerCrispXTop.call(this, contentHeader, this.boxX + this.boxWidth / 2, headerY);
    //contentHeader.roundPixels = true;
    contentHeader.anchor.set(0, 0);
    const centerX = Math.round(this.boxX + this.boxWidth / 2);
    contentHeader.x = Math.round(centerX - contentHeader.width / 2);
    contentHeader.y = headerY;
    this.lContent.addChild(contentHeader);
    
    // Content area bounds
    const contentX = this.boxX + 40;
    const contentY = this.boxY + TAB_HEIGHT + 90;
    const contentW = this.boxWidth - 80;
    const contentH = this.boxHeight - TAB_HEIGHT - 120;
    
    // Render appropriate content
    if (this.activeTab === 0) {
      this.renderTradeRoutes(contentX, contentY, contentW, contentH);
    } else if (this.activeTab === 1) {
      this.renderTradeDeals(contentX, contentY, contentW, contentH);
    } else {
      this.renderResources(contentX, contentY, contentW, contentH);
    }
  }

  /* =========================================================
       TRADE ROUTES RENDERER
     ========================================================= */
  renderTradeRoutes(x, y, w, h) {
    const container = new PIXI.Container();
    
    // Extract active trade routes from data
    const routes = [];
    
    if (this.unitsTradeYields && this.unitsTradeYields[this.turn]) {
      const tradeYields = this.unitsTradeYields[this.turn];
      const tradeCityFrom = this.unitsTradeCityFrom?.[this.turn] || [];
      const tradeCityTo = this.unitsTradeCityTo?.[this.turn] || [];
      const tradePlayerTo = this.unitsTradePlayerTo?.[this.turn] || [];

      
      // Iterate through all players
      for (let playerIdx = 0; playerIdx < 6; playerIdx++) {
        if (!tradeYields[playerIdx]) continue;
        
        // Iterate through all units for this player
        for (let unitIdx = 0; unitIdx < 30; unitIdx++) {
          if (!tradeYields[playerIdx][unitIdx]) continue;

         // Check if this unit has non-zero trade yields (either sender or receiver yields)
         const senderYields = tradeYields[playerIdx][unitIdx][0];
         const receiverYields = tradeYields[playerIdx][unitIdx][1];
         
         if (!senderYields || !Array.isArray(senderYields)) continue;
         if (!receiverYields || !Array.isArray(receiverYields)) continue;
         
         // Check if any yields are non-zero (either sender OR receiver)
         const hasSenderYields = senderYields.some(val => val !== 0);
         const hasReceiverYields = receiverYields.some(val => val !== 0);
         const hasActiveRoute = hasSenderYields || hasReceiverYields;
         
         if (!hasActiveRoute) continue;
          // Get sender city
          const senderCity = tradeCityFrom[playerIdx][unitIdx] + 1;

          if (senderCity === 1) {
            var sender = `Player ${playerIdx + 1} / Capital`;
          } else  {
            var sender = `Player ${playerIdx + 1} / City ${senderCity}`;
          }
          
          // Get receiver info
          const receiverPlayerIdx = tradePlayerTo[playerIdx]?.[unitIdx];
          let receiver = 'Unknown';
          
          if (receiverPlayerIdx !== undefined) {
            if (receiverPlayerIdx < 6) {
              // Trading with another player
              const receiverCity = tradeCityTo[playerIdx][unitIdx] + 1;

              if (receiverCity === 1) {
                receiver = `Player ${receiverPlayerIdx + 1} / Capital`;
              } else {
                receiver = `Player ${receiverPlayerIdx + 1} / City ${receiverCity}`;
              }
            } else {
              // Trading with a city-state (idx >= 6)
              const csIdx = receiverPlayerIdx - 6;
              receiver = constants.csNames?.[csIdx] || `City-State ${csIdx + 1}`;
            }
          }

          
          routes.push({
            sender,
            receiver,
            turnsRemaining: this.unitsEngaged[this.turn][playerIdx][unitIdx],
            senderYields: senderYields,
            receiverYields: receiverYields,
          });
        }
      }
    }
    
    // Summary section
    const summaryBox = new PIXI.Graphics();
    summaryBox.beginFill(COLORS.stoneBg, 0.3);
    summaryBox.drawRect(x, y, w, 60);
    summaryBox.endFill();
    summaryBox.lineStyle(1, COLORS.goldDark);
    summaryBox.drawRect(x, y, w, 60);
    summaryBox.stroke();
    container.addChild(summaryBox);
    
    // Summary stats
    const activeRoutes = routes.length;
    const summaryText = new PIXI.Text(`ACTIVE TRADE ROUTES: ${activeRoutes}`, {
      ...requirementStyle,
      fontSize: 18
    });
    summaryText.anchor.set(0.5, 0.5);
    summaryText.position.set(x + w/2, y + 30);
    container.addChild(summaryText);
    
    // Routes list header
    let listY = y + 80;
    const routesTitle = new PIXI.Text('TRADE ROUTE DETAILS', requirementStyle);
    routesTitle.position.set(x, listY);
    container.addChild(routesTitle);
    
    // Column headers - modified to show S/R sub-columns
    listY += 35;
    
    // Main headers
    const mainHeaders = [
      { text: 'SENDER \u27A1\uFE0E', x: x + 10 },
      { text: 'RECEIVER \u2B05\uFE0E', x: x + 200 },
      { text: 'TURNS', x: x + 380 }
    ];
    
    mainHeaders.forEach(header => {
      const headerText = new PIXI.Text(header.text, {
        ...statusStyle,
        fontSize: 12,
        fill: COLORS.goldDark
      });
      headerText.position.set(header.x, listY);
      container.addChild(headerText);
    });
    
    // Yield headers with S/R sub-columns
    const yieldHeaders = [
      { text: 'FOOD', x: x + 470 },
      { text: 'PROD', x: x + 550 },
      { text: 'GOLD', x: x + 630 },
      { text: 'TOURISM', x: x + 720 },
      { text: 'INFLUENCE', x: x + 830 },
      { text: 'RELIGUOUS PRESSURE', x: x + 950 }
    ];
    const S_ARROW = '\u27A1\uFE0E'; // ➡︎ text-style
    const R_ARROW = '\u2B05\uFE0E'; // ⬅︎ text-style
    const ARROW_Y = listY + 3;
    const GAP = 6;                  // gap between arrow and divider
    const DIVIDER_W = 1;            // visual width of the 1px divider

    yieldHeaders.forEach(header => {
      // Header label
      const headerText = new PIXI.Text(header.text, {
        ...statusStyle,
        fontSize: 11,
        fill: COLORS.goldDark
      });
      headerText.position.set(header.x, listY - 12);
      container.addChild(headerText);

      // Create arrows and measure widths
      const sText = new PIXI.Text(S_ARROW, { ...statusStyle, fontSize: 10, fill: COLORS.goldDark });
      const rText = new PIXI.Text(R_ARROW, { ...statusStyle, fontSize: 10, fill: COLORS.goldDark });

      // Build a mini-group so we can center as one unit
      const group = new PIXI.Container();
      group.addChild(sText);
      group.addChild(rText);

      // Compute header’s visual center (based on rendered text width)
      const headerCenterX = headerText.x + headerText.width / 2;

      // Compute group layout
      const sW = sText.width;
      const rW = rText.width;
      const groupWidth = sW + GAP + DIVIDER_W + GAP + rW;

      // Position S (right) arrow first; everything else derives from it
      sText.position.set(0, 0); // temporary; we'll position the group as a whole
      // Divider graphics (placed between S and R)
      const divider = new PIXI.Graphics();
      divider.lineStyle(1, COLORS.divider, 0.3);
      // Divider x relative to group’s left
      const dividerX = sW + GAP;
      divider.moveTo(dividerX, 1);          // small offset for crispness
      divider.lineTo(dividerX, 14);
      divider.stroke();
      group.addChild(divider);

      // R (left) arrow sits after divider + gap
      rText.position.set(dividerX + DIVIDER_W + GAP, 0);

      // Now center the whole group under the header label
      group.position.set(headerCenterX - groupWidth / 2, ARROW_Y);
      container.addChild(group);
    });
    
    // Header underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.5);
    headerLine.moveTo(x, listY + 20);
    headerLine.lineTo(x + w, listY + 20);
    headerLine.stroke();
    container.addChild(headerLine);
    
    // Routes list with scrolling
    const rowHeight = 25;
    const visibleHeight = h - (listY - y) - 60;
    const maxVisibleRows = Math.floor(visibleHeight / rowHeight);
    const actualVisibleHeight = maxVisibleRows * rowHeight;
    const totalContentHeight = Math.max(routes.length * rowHeight, rowHeight);
    const needsScroll = totalContentHeight > actualVisibleHeight;
    
    // Create scrollable container
    const rowsContainer = new PIXI.Container();
    
    if (needsScroll) {
      const mask = new PIXI.Graphics();
      mask.beginFill(0xffffff);
      mask.drawRect(x, listY + 30, w - 20, actualVisibleHeight);
      mask.endFill();
      rowsContainer.mask = mask;
      container.addChild(mask);
    }
    
    // Render route rows
    if (routes.length === 0) {
      const noRoutesText = new PIXI.Text('No active trade routes', {
        ...statusStyle,
        fill: COLORS.goldDark,
      });
      noRoutesText.anchor.set(0.0, 0.0);
      noRoutesText.position.set(x + w/2 - 70, listY + 50);
      rowsContainer.addChild(noRoutesText);
      this.tradeRouteRemaining = [];

    } else {
      routes.forEach((route, idx) => {
        
        // TODO: THIS NEEDS TO BE REPLACED WITH NTURNSENGAGED
        //console.log("ENGAGED: ", this.unitsEngaged[this.turn]);
        const turnsRemaining = route.turnsRemaining;
        const rowY = listY + 35 + idx * rowHeight - this.routesScrollY;
        
        // Alternating row background
        if (idx % 2 === 0) {
          const rowBg = new PIXI.Graphics();
          rowBg.beginFill(COLORS.tabInactive, 0.5);
          rowBg.drawRect(x, rowY - 1, w, rowHeight - 2);
          rowBg.endFill();
          rowsContainer.addChild(rowBg);
        }
        
        // Sender
        const senderText = new PIXI.Text(route.sender, {
          ...statusStyle,
          fontSize: 12
        });
        senderText.position.set(x + 10, rowY);
        rowsContainer.addChild(senderText);
        
        // Receiver
        const receiverText = new PIXI.Text(route.receiver, {
          ...statusStyle,
          fontSize: 12
        });
        receiverText.position.set(x + 200, rowY);
        rowsContainer.addChild(receiverText);
        
        // Turns remaining
        const turnsText = new PIXI.Text(turnsRemaining.toString(), {
          ...statusStyle,
          fontSize: 12,
          fill: turnsRemaining < 10 ? COLORS.pending : COLORS.goldDark
        });
        turnsText.position.set(x + 380 + 18, rowY);
        rowsContainer.addChild(turnsText);
        
        // Yields with S/R sub-columns
        const yieldPositions = [
          { idx: 0, x: x + 470, offsets: [-4, -2] }, // Food
          { idx: 1, x: x + 550, offsets: [-4, -2] }, // Production  
          { idx: 2, x: x + 630, offsets: [-2, -2] }, // Gold
          { idx: 7, x: x + 720, offsets: [0, 8] }, // Tourism
          { idx: 8, x: x + 830, offsets: [0, 17] }, // Influence
          { idx: 9, x: x + 950, offsets: [0, 58] }  // Religious Pressure
        ];
        
        yieldPositions.forEach(({ idx, x: xPos, offsets }) => {
          // Sender value (S column)
          const senderValue = route.senderYields[idx] || 0;
          if (senderValue !== 0) {
            const senderText = new PIXI.Text(
              senderValue > 0 ? `${Number(senderValue).toFixed(1)}` : senderValue.toString(), 
              {
                ...statusStyle,
                fontSize: 11,
                fontWeight: 600,
                fill: COLORS.success
              }
            );
            senderText.position.set(xPos + offsets[0], rowY);
            rowsContainer.addChild(senderText);
          }
          
          // Receiver value (R column)
          const receiverValue = route.receiverYields[idx] || 0;
          if (receiverValue !== 0) {
            const receiverText = new PIXI.Text(
              receiverValue > 0 ? `${Number(receiverValue).toFixed(1)}` : receiverValue.toString(),
              {
                ...statusStyle,
                fontSize: 11,
                fontWeight: 600,
                fill: COLORS.pending
              }
            );
            receiverText.position.set(xPos + 30 + offsets[1], rowY);
            rowsContainer.addChild(receiverText);
          }
        });
      });
    }
    
    container.addChild(rowsContainer);
    
    // Add scrollbar if needed
    if (needsScroll) {
      const maxScroll = totalContentHeight - actualVisibleHeight;
      const scrollBarX = x + w - 15;
      const scrollBarHeight = actualVisibleHeight;
      const scrollThumbHeight = Math.max(30, (actualVisibleHeight / totalContentHeight) * scrollBarHeight);
      
      // Scroll track
      const scrollTrack = new PIXI.Graphics();
      scrollTrack.beginFill(COLORS.stoneBg, 0.3);
      scrollTrack.drawRect(scrollBarX, listY + 30, 10, scrollBarHeight);
      scrollTrack.endFill();
      container.addChild(scrollTrack);
      
      // Scroll thumb
      const thumbY = listY + 30 + (this.routesScrollY / maxScroll) * (scrollBarHeight - scrollThumbHeight);
      const scrollThumb = new PIXI.Graphics();
      scrollThumb.beginFill(COLORS.goldDark, 0.6);
      scrollThumb.drawRect(scrollBarX, thumbY, 10, scrollThumbHeight);
      scrollThumb.endFill();
      container.addChild(scrollThumb);
      
      // Scroll interaction
      const scrollArea = new PIXI.Graphics();
      scrollArea.beginFill(0x000000, 0.01);
      scrollArea.drawRect(x, listY + 30, w, actualVisibleHeight);
      scrollArea.endFill();
      scrollArea.eventMode = 'static';
      container.addChild(scrollArea);
      
      scrollArea.on('wheel', (e) => {
        const delta = e.deltaY > 0 ? 30 : -30;
        this.routesScrollY = Math.max(0, Math.min(maxScroll, this.routesScrollY + delta));
        this.redrawContent();
      });
    }
    
    this.lContent.addChild(container);
  }
  /* =========================================================
       TRADE DEALS RENDERER
     ========================================================= */
  renderTradeDeals(x, y, w, h) {
    const container = new PIXI.Container();
    
    // Active deals header
    const dealsTitle = new PIXI.Text('ACTIVE TRADE AGREEMENTS', requirementStyle);
    dealsTitle.position.set(x, y);
    container.addChild(dealsTitle);

    //console.log("LEDGER: ", this.tradeLedger[this.turn]);
    //console.log("LENGTH: ", this.tradeLengthLedger[this.turn]);
    const ledger = this.tradeLedger[this.turn];
    const dealLength = this.tradeLengthLedger[this.turn];
    const deals = [];
    
    // per turn: (6, 6, 10, 2)
    // iterate through all players
    for (let fromPlayerIdx = 0; fromPlayerIdx < 6; fromPlayerIdx++) {
      // Next layer in is the receiving player 
      for (let toPlayerIdx = 0; toPlayerIdx < 6; toPlayerIdx++) {
        for (let dealIdx = 0; dealIdx < ledger[fromPlayerIdx][toPlayerIdx].length; dealIdx++) {
          // Early exit if no deal in this slot
          if (ledger[fromPlayerIdx][toPlayerIdx][dealIdx][0] === 0) continue;
          //console.log("from/to: ", fromPlayerIdx, toPlayerIdx, dealIdx);
          //console.log(ledger[fromPlayerIdx][toPlayerIdx]);

          deals.push({
            party1: `Player ${fromPlayerIdx + 1}`,
            party2: `Player ${toPlayerIdx + 1}`,
            party1Gives: ledger[fromPlayerIdx][toPlayerIdx][dealIdx][0],
            party2Gives: ledger[fromPlayerIdx][toPlayerIdx][dealIdx][1],
            turnsRemaining: `Turns Remaining: ${dealLength[fromPlayerIdx][dealIdx]}`,
          });
        }
      }
    }
    // Deal cards setup
  const headerHeight = 35;
  const cardHeight = 100;
  const cardGap = 10;
  const S_ARROW = '\u27A1\uFE0E'; // ➡︎ text-style
  const R_ARROW = '\u2B05\uFE0E'; // ⬅︎ text-style
  
  // Calculate scrolling parameters
  const visibleHeight = h - headerHeight - 20; // Leave some padding
  const totalContentHeight = deals.length > 0 ? deals.length * (cardHeight + cardGap) - cardGap : 150;
  const needsScroll = totalContentHeight > visibleHeight;
  
  // Create scrollable container for deal cards
  const dealsContainer = new PIXI.Container();
  
  // Apply mask if scrolling is needed
  if (needsScroll) {
    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff);
    mask.drawRect(x, y + headerHeight, w - 20, visibleHeight);
    mask.endFill();
    dealsContainer.mask = mask;
    container.addChild(mask);
  }

  if (deals.length === 0) {
    const noRoutesText = new PIXI.Text('No active trade deals', {
      ...statusStyle,
      fill: COLORS.goldDark,
    });
    noRoutesText.anchor.set(0.0, 0.0);
    noRoutesText.position.set(x + w/2 - 70, y + headerHeight + 50 - this.dealsScrollY);
    dealsContainer.addChild(noRoutesText);

  } else {
    deals.forEach((deal, idx) => {
      // Calculate card position with scroll offset
      const cardY = y + headerHeight + idx * (cardHeight + cardGap) - this.dealsScrollY;
      
      // Deal card background
      const dealCard = new PIXI.Graphics();
      const cardColor = COLORS.stoneBg;
      dealCard.beginFill(cardColor, 0.2);
      dealCard.drawRoundedRect(x, cardY, w - (needsScroll ? 20 : 0), cardHeight, 0);
      dealCard.endFill();
      
      // Card border
      const borderColor = COLORS.goldDark;
      dealCard.lineStyle(1, borderColor, 0.5);
      dealCard.drawRoundedRect(x, cardY, w - (needsScroll ? 20 : 0), cardHeight, 0);
      dealCard.stroke();
      dealsContainer.addChild(dealCard);
      
      // Trade details columns
      const col1 = x + 20;
      const col2 = x + 120;
      const resXOffset = 9;
      const resScaleMult = 2;
      const detailY = cardY + 40;

      const party1Text = new PIXI.Text(deal.party1, {
        ...requirementStyle,
        fontSize: 18
      });
      party1Text.position.set(col1, cardY + 10);
      dealsContainer.addChild(party1Text);

      const party2Text = new PIXI.Text(deal.party2, {
        ...requirementStyle,
        fontSize: 18
      });
      party2Text.position.set(col2, cardY + 10);
      dealsContainer.addChild(party2Text);

      if (deal.party1Gives === 1) {
        // Embassy
        const party1GivesText = new PIXI.Text("Embassy", {
          ...statusStyle,
          fontSize: 18
        });
        party1GivesText.position.set(col1, cardY + 40);
        dealsContainer.addChild(party1GivesText);
      } else if (deal.party1Gives === 2) {
        // GPT
        const party1GivesText = new PIXI.Text("5 GPT", {
          ...statusStyle,
          fontSize: 18
        });
        party1GivesText.position.set(col1, cardY + 40);
        dealsContainer.addChild(party1GivesText);
      } else if (deal.party1Gives === 3) {
        // Peace deal
        const party1GivesText = new PIXI.Text("Peace Deal", {
          ...statusStyle,
          fontSize: 18
        });
        party1GivesText.position.set(col1, cardY + 40);
        dealsContainer.addChild(party1GivesText);
      } else {
        // Resource. -4 here as idx 0 is null
        const resourceIdx = deal.party1Gives - 3;
        const def = constants.resourceDefs[resourceIdx];
        const tex = resTextures[resourceIdx];
        const spr = new PIXI.Sprite(tex);
        spr.scale.set(def.s * resScaleMult);
        spr.x = col1 + resXOffset - 5;
        spr.y = cardY + 40;
        dealsContainer.addChild(spr);
      }
      
      if (deal.party2Gives === 1) {
        // Embassy
        const party2GivesText = new PIXI.Text("Embassy", {
          ...statusStyle,
          fontSize: 18
        });
        party2GivesText.position.set(col2, cardY + 40);
        dealsContainer.addChild(party2GivesText);
      } else if (deal.party2Gives === 2) {
        // GPT
        const party2GivesText = new PIXI.Text("5 GPT", {
          ...statusStyle,
          fontSize: 18
        });
        party2GivesText.position.set(col2, cardY + 40);
        dealsContainer.addChild(party2GivesText);
      } else if (deal.party2Gives === 3) {
        // Peace deal
        const party2GivesText = new PIXI.Text("Peace Deal", {
          ...statusStyle,
          fontSize: 18
        });
        party2GivesText.position.set(col2, cardY + 40);
        dealsContainer.addChild(party2GivesText);
      } else {
        // Resource. -3 here as idx 0 is null
        const resourceIdx = deal.party2Gives - 3;
        const def = constants.resourceDefs[resourceIdx];
        const tex = resTextures[resourceIdx];
        const spr = new PIXI.Sprite(tex);
        spr.scale.set(def.s * resScaleMult);
        spr.x = col2 + resXOffset - 5;
        spr.y = cardY + 40;
        dealsContainer.addChild(spr);
      }

      const turnsRemainingText = new PIXI.Text(deal.turnsRemaining, {
        ...requirementStyle,
        fontSize: 18
      });
      turnsRemainingText.position.set(col1 + 200, cardY + 10);
      dealsContainer.addChild(turnsRemainingText);

      
    });
  }
  
  container.addChild(dealsContainer);
  
  // Add scrollbar if needed
  if (needsScroll) {
    const maxScroll = Math.max(0, totalContentHeight - visibleHeight);
    const scrollBarX = x + w - 15;
    const scrollBarY = y + headerHeight;
    const scrollBarHeight = visibleHeight;
    const scrollThumbHeight = Math.max(30, (visibleHeight / totalContentHeight) * scrollBarHeight);
    
    // Scroll track
    const scrollTrack = new PIXI.Graphics();
    scrollTrack.beginFill(COLORS.stoneBg, 0.3);
    scrollTrack.drawRect(scrollBarX, scrollBarY, 10, scrollBarHeight);
    scrollTrack.endFill();
    container.addChild(scrollTrack);
    
    // Scroll thumb
    const thumbY = scrollBarY + (this.dealsScrollY / maxScroll) * (scrollBarHeight - scrollThumbHeight);
    const scrollThumb = new PIXI.Graphics();
    scrollThumb.beginFill(COLORS.goldDark, 0.6);
    scrollThumb.drawRect(scrollBarX, thumbY, 10, scrollThumbHeight);
    scrollThumb.endFill();
    container.addChild(scrollThumb);
    
    // Scroll interaction area (invisible but interactive)
    const scrollArea = new PIXI.Graphics();
    scrollArea.beginFill(0x000000, 0.01);
    scrollArea.drawRect(x, scrollBarY, w, scrollBarHeight);
    scrollArea.endFill();
    scrollArea.eventMode = 'static';
    container.addChild(scrollArea);
    
    // Handle scroll wheel
    scrollArea.on('wheel', (e) => {
      const delta = e.deltaY > 0 ? 30 : -30;
      this.dealsScrollY = Math.max(0, Math.min(maxScroll, this.dealsScrollY + delta));
      this.redrawContent();
    });
  }
  
  this.lContent.addChild(container);
    
  }

/* =========================================================
   RESOURCES RENDERER  (layout-only for now)
   ========================================================= */
  /* =========================================================
   RESOURCES RENDERER — now filled with data
   ========================================================= */
renderResources(x, y, w, h) {
  const container = new PIXI.Container();

  const playerResourceTotals = this.getResourceTotalsForTurn(this.turn); // (6, R)
  // ===== Compute (6, num_resources) totals from (6, max_cities, num_resources) =====
  //const resourcesOwnedLocal = this.resourcesOwned?.[this.turn];
  //let playerResourceTotals = null; // shape (6, R)

  //if (resourcesOwnedLocal && resourcesOwnedLocal.length) {
  //  const P = Math.min(6, resourcesOwnedLocal.length);
  //  // try to infer R safely
  //  const firstPlayerCities = resourcesOwnedLocal[0] || [];
  //  const R = (firstPlayerCities[0]?.length) ?? 0;

  //  playerResourceTotals = new Array(P);
  //  for (let p = 0; p < P; p++) {
  //    const cities = resourcesOwnedLocal[p] || [];
  //    const sums = new Array(R).fill(0);
  //    for (let c = 0; c < cities.length; c++) {
  //      const row = cities[c] || [];
  //      for (let r = 0; r < R; r++) {
  //        sums[r] += row[r] || 0;
  //      }
  //    }
  //    playerResourceTotals[p] = sums;
  //  }
  //}

  // ===== Table metrics & resource rows (skip index 0 per your note) =====
  const headerH = 32;
  const rowH    = 28;
  const leftPad = 10;
  const resColW = Math.max(220, Math.floor(w * 0.32)); // icon + name
  const innerW  = w;

  const resourceIndices = [];
  for (let i = 1; i < (resTextures?.length || 1); i++) resourceIndices.push(i);

  // ===== Header strip =====
  const headerBg = new PIXI.Graphics();
  headerBg.beginFill(COLORS.stoneBg, 0.25).drawRect(x, y, innerW, headerH).endFill();
  headerBg.lineStyle(1, COLORS.goldDark, 0.5).drawRect(x, y, innerW, headerH).stroke();
  container.addChild(headerBg);

  const headerText = (txt, px) => {
    const t = new PIXI.Text(txt, { ...statusStyle, fontSize: 12, fill: COLORS.goldDark });
    t.position.set(px, y + 8);
    container.addChild(t);
    return t;
  };

  headerText('RESOURCE', x + leftPad + 125);

  const playerCols = 6;
  let playerAreaW  = innerW - resColW;
  let colW = Math.floor(playerAreaW / playerCols);

  for (let p = 0; p < playerCols; p++) {
    const cx = x + resColW + p * colW + Math.floor(colW / 2);
    const t = new PIXI.Text(`PLAYER ${p + 1}`, { ...statusStyle, fontSize: 12, fill: COLORS.goldDark });
    t.anchor.set(0.5, 0);
    t.position.set(cx, y + 8);
    container.addChild(t);
  }

  const headLine = new PIXI.Graphics();
  headLine.lineStyle(1, COLORS.goldDark, 0.5).moveTo(x, y + headerH).lineTo(x + innerW, y + headerH).stroke();
  container.addChild(headLine);

  // ===== Scroll area =====
  const listY              = y + headerH;
  const visibleHeight      = h - headerH;
  const totalContentHeight = resourceIndices.length * rowH;
  const needsScroll        = totalContentHeight > visibleHeight;

  const innerWForRows = needsScroll ? (innerW - 20) : innerW;
  playerAreaW = innerWForRows - resColW;
  colW = Math.floor(playerAreaW / playerCols);

  const rows = new PIXI.Container();
  if (needsScroll) {
    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff).drawRect(x, listY, innerWForRows, visibleHeight).endFill();
    rows.mask = mask;
    container.addChild(mask);
  }
function toTitleCase(str) {
  return str.replace(
    /\w\S*/g,
    text => text.charAt(0).toUpperCase() + text.substring(1).toLowerCase()
  );
}
  // Helpers
  const getResourceName = (idx) => {
    return toTitleCase(constants.ALL_RESOURCES[idx])
  };

  const makeResourceSprite = (idx) => {
    const entry = resTextures?.[idx];
    if (!entry) return null;

    let texture = null;
    if (entry.baseTexture) {
      texture = entry;                   // PIXI.Texture
    } else if (entry.path) {
      texture = PIXI.Texture.from(entry.path); // {path, s}
    }
    if (!texture) return null;

    const spr = new PIXI.Sprite(texture);
    const scaleMult = 0.1;
    const s = entry.s != null ? entry.s : 1;
    spr.scale.set(s * scaleMult);
    return spr;
  };

  // ===== Rows with data =====
  resourceIndices.forEach((resIdx, i) => {
    const rowY = listY + i * rowH - this.resourcesScrollY;
    if (rowY > listY + visibleHeight || rowY + rowH < listY) return;

    if (i % 2 === 0) {
      const bg = new PIXI.Graphics();
      bg.beginFill(COLORS.tabInactive, 0.4).drawRect(x, rowY, innerWForRows, rowH).endFill();
      rows.addChild(bg);
    }

    const spr = makeResourceSprite(resIdx);
    if (spr) {
      spr.x = x + leftPad + 130;
      spr.y = rowY + Math.round((rowH - spr.height) / 2);
      rows.addChild(spr);
    }

    const nameX = x + leftPad + (spr ? spr.width + 8 : 26) + 130;
    const nameT = new PIXI.Text(getResourceName(resIdx), { ...statusStyle, fontSize: 14 });
    nameT.position.set(nameX, rowY + Math.round((rowH - nameT.height) / 2));
    rows.addChild(nameT);

    // >>> Fill player cells with totals <<<
    for (let p = 0; p < playerCols; p++) {
      const cx = x + resColW + p * colW + Math.floor(colW / 2);
      const cy = rowY + Math.floor(rowH / 2);

      const val = playerResourceTotals?.[p]?.[resIdx - 1] ?? 0;

      if (val > 0)  {
        const cellText = new PIXI.Text(String(Math.round(val)), { ...statusStyle, fontSize: 14 });
        cellText.anchor.set(0.5, 0.5);
        cellText.position.set(cx, cy);
        rows.addChild(cellText);
      }
    }

    // Column dividers
    const div = new PIXI.Graphics();
    div.lineStyle(1, COLORS.divider, 0.25);
    for (let p = 0; p <= playerCols; p++) {
      const vx = x + resColW + p * colW;
      div.moveTo(vx, rowY);
      div.lineTo(vx, rowY + rowH);
      div.stroke();
    }
    div.moveTo(x + resColW, rowY).lineTo(x + resColW, rowY + rowH).stroke();
    rows.addChild(div);
  });

  container.addChild(rows);

  // ===== Scrollbar =====
  if (needsScroll) {
    const maxScroll = Math.max(0, totalContentHeight - visibleHeight);
    const scrollBarX = x + innerWForRows + 5;
    const scrollBarY = listY;
    const scrollBarH = visibleHeight;
    const thumbH = Math.max(30, (visibleHeight / totalContentHeight) * scrollBarH);

    const track = new PIXI.Graphics();
    track.beginFill(COLORS.stoneBg, 0.3).drawRect(scrollBarX, scrollBarY, 10, scrollBarH).endFill();
    container.addChild(track);

    const thumbY = scrollBarY + (this.resourcesScrollY / maxScroll) * (scrollBarH - thumbH);
    const thumb = new PIXI.Graphics();
    thumb.beginFill(COLORS.goldDark, 0.6).drawRect(scrollBarX, thumbY, 10, thumbH).endFill();
    container.addChild(thumb);

    const scrollArea = new PIXI.Graphics();
    scrollArea.beginFill(0x000000, 0.01).drawRect(x, scrollBarY, innerWForRows, scrollBarH).endFill();
    scrollArea.eventMode = 'static';
    container.addChild(scrollArea);

    scrollArea.on('wheel', (e) => {
      const delta = e.deltaY > 0 ? 30 : -30;
      this.resourcesScrollY = Math.max(0, Math.min(maxScroll, this.resourcesScrollY + delta));
      this.redrawContent();
    });
  }

  this.lContent.addChild(container);
}

}
