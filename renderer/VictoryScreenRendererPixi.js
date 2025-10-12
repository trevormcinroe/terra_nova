import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";
/* =========================================================
   VictoryScreenRendererPixi — Victory conditions overlay
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
const VICTORY_TYPES = ['Cultural', 'Diplomatic', 'Scientific', 'Domination'];
const TAB_HEIGHT = 50;
const Y_OFFSET = 30;

// Civilization-style color palette (matching religion screen)
const COLORS = {
  bg: 0x1a1612,                    // Dark brown/black
  stoneBg: 0x2c2416,               // Stone texture base
  stoneLight: 0x3d3426,            // Stone highlight
  parchment: 0x3d3426,             // Parchment background
  goldBorder: 0xc9b037,            // Gold trim
  goldBright: 0xffd700,            // Bright gold
  goldDark: 0x8b7355,              // Dark gold/bronze
  headerBg: 0x81ADC8,              // Header background
  tabActive: 0x2c2416,             // Active tab background
  tabInactive: 0x1a1612,           // Inactive tab background
  divider: 0x4a3f28,               // Divider lines
  shadow: 0x000000,                // Pure black shadows
  success: 0x16a34a,               // Green for completed
  pending: 0xca8a04,               // Yellow for in-progress
  failed: 0xBB4430,                // Red for not started
};

/* ========================================================= */
export class VictoryScreenRendererPixi {
  constructor(app) {
    /* ------------- scene graph ---------------- */
    this.app = app;
    this.stage = new PIXI.Container();
    this.app.stage.addChild(this.stage);
    
    // Ensure the renderer has a transparent background
    this.app.renderer.background.alpha = 0;

    /* ------------- state ---------------------- */
    this.activeTab = 0;  // Default to Cultural
    this.victoryData = null;
    this.turn = 0;


    /* ------------- geometry ------------------- */
    const { width: W, height: H } = app.renderer;
    this.boxWidth = Math.min(W * 0.85, 1200);
    this.boxHeight = Math.min(H * 0.85, 700);
    this.boxX = (W - this.boxWidth) / 2;
    this.boxY = (H - this.boxHeight) / 2;

    /* ------------- layers --------------------- */
    this.lBackground = new PIXI.Container();  // background effects
    this.lMain = new PIXI.Container();        // main box
    this.lTabs = new PIXI.Container();        // tab buttons
    this.lContent = new PIXI.Container();     // tab content
    this.stage.addChild(this.lBackground, this.lMain, this.lTabs, this.lContent);

    /* ------------- interaction ---------------- */
    this.tabButtons = [];
    this.buildInterface();
  }

  /* =========================================================
       PUBLIC API
     ========================================================= */
  setVictoryData(data) { 
    this.victoryData = data; 
    this.redrawContent();
  }
  
  setTurn(turn) { 
    this.turn = turn; 
    this.redrawContent();
  }
  /** Precompute per-turn Great Works counts by summing over cities.
  *  Input: this.gwSlots shape (T, P, C, 4)
  *  Output: this._gwByTurn shape (T, P, 4)
  *  Note: index 3 in the last axis is unused per spec.
  */
  _buildGreatWorksIndex() {
    const gw = this.gwSlots;
    if (!gw || !gw.length) {
      this._gwByTurn = [];
      return;
    }
    const T = gw.length;
    const P = gw[0]?.length ?? 0;

    this._gwByTurn = new Array(T);
    for (let t = 0; t < T; t++) {
      const perPlayer = new Array(P);
      const gw_t = gw[t];
      for (let p = 0; p < P; p++) {
        const cities = gw_t[p] || [];
        let w = 0, a = 0, m = 0, u = 0; // writing, art, music, unused
        for (let c = 0; c < cities.length; c++) {
          const slot = cities[c] || [0,0,0,0];
          w += slot[0] || 0;
          a += slot[1] || 0;
          m += slot[2] || 0;
          u += slot[3] || 0;
        }
        perPlayer[p] = [w, a, m, u];
      }
      this._gwByTurn[t] = perPlayer;
    }
  }

  setData(csReligiousPopulation, csRelationships, csInfluence, csType, csQuest, csCultureTracker, csFaithTracker, csTechTracker, csTradeTracker, csReligionTracker, csWonderTracker, csResourceTracker, playerTechs, haveMet, tradeLedger, atWar, unitType, unitMilitary, unitHealth, unitRowcol, unitCombatAccel, hasSacked, cultureTotal, tourismTotal, gpps, gpThreshold, gwSlots, numDelegates, playerBuildings) {
    /* ------------- data ----------------------- */
    this.csReligiousPopulation = csReligiousPopulation;
    this.csRelationships = csRelationships;
    this.csInfluence = csInfluence;
    this.csType = csType;
    this.csQuest = csQuest;
    this.csCultureTracker = csCultureTracker;
    this.csFaithTracker = csFaithTracker;
    this.csTechTracker = csTechTracker;
    this.csTradeTracker = csTradeTracker;
    this.csReligionTracker = csReligionTracker;
    this.csWonderTracker = csWonderTracker;
    this.csResourceTracker = csResourceTracker;
    this.playerTechs = playerTechs;
    this.haveMet = haveMet;
    this.tradeLedger = tradeLedger;
    this.atWar = atWar;
    this.unitType = unitType;
    this.unitMilitary = unitMilitary;
    this.unitHealth = unitHealth;
    this.unitRowcol = unitRowcol;
    this.unitCombatAccel = unitCombatAccel;
    this.hasSacked = hasSacked;
    this.cultureTotal = cultureTotal;
    this.tourismTotal = tourismTotal;
    this.gpps = gpps;
    this.gpThreshold = gpThreshold;
    this.gwSlots = gwSlots;
    this.numDelegates = numDelegates;
    this.playerBuildings = playerBuildings;
    this._buildGreatWorksIndex();
  }

  /** Return (P, 4) array for a given turn, building the index on first use. */
  _getGWCountsForTurn(turn) {
    if (!this._gwByTurn) this._buildGreatWorksIndex();
    return (this._gwByTurn && this._gwByTurn[turn]) ? this._gwByTurn[turn] : [];
  }
  
  start() { 
    this.stage.visible = true;
    this.redrawContent(); 
  }
  
  stop() { 
    this.stage.visible = false;
  }
  
  getDominationUnitRows = (playerIndex) => {
    // return [{ type, loc, hp, mult, xp }, ...] for that player
    const unitTypeLocal = this.unitType[this.turn][playerIndex];
    const unitHealthLocal = this.unitHealth[this.turn][playerIndex];
    const unitRowcolLocal = this.unitRowcol[this.turn][playerIndex];
    const unitCombatAccelLocal = this.unitCombatAccel[this.turn][playerIndex];
    
    const unitSummary = [];

    for (let i = 0; i < unitTypeLocal.length; i++) {
      if (unitTypeLocal[i] === 0) continue;
      unitSummary.push({
        type: constants.unitNames[unitTypeLocal[i] - 1],
        loc: `${unitRowcolLocal[i][0]}, ${unitRowcolLocal[i][1]}`,
        hp: Number(unitHealthLocal[i]).toFixed(1),
        mult: Number(unitCombatAccelLocal[i]).toFixed(2),
        xp: 0
      });
    }
    return unitSummary
  };

  /* =========================================================
       BUILD INTERFACE
     ========================================================= */
  sumRowsFast(matrix) {
    const out = new Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
      let s = 0;
      const row = matrix[i];
      for (let j = 0; j < row.length; j++) s += row[j];
      out[i] = s;
    }
    return out;
  }
  subNew(a, b) {
    if (a.length !== b.length) throw new Error('length mismatch');
    const out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
    return out;
  }

  _centerCrisp(text, cx, cy) {
    text.roundPixels = true;
    text.anchor.set(0.5, 0.5);
    // force metrics read so width/height are current for the style
    const w = Math.round(text.width);
    const h = Math.round(text.height);
    const oddW = (w % 2) === 1;
    const oddH = (h % 2) === 1;
    text.position.set(Math.round(cx) + (oddW ? 0.5 : 0), Math.round(cy) + (oddH ? 0.5 : 0));
  }

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
    
    // Ornate border with double lines
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
    const tabWidth = this.boxWidth / VICTORY_TYPES.length;
    
    VICTORY_TYPES.forEach((type, idx) => {
      const tabContainer = new PIXI.Container();

      // Tab button graphics
      const tabButton = new PIXI.Graphics();
      this.drawTab(tabButton, idx, idx === this.activeTab);

      // Make tab interactive
      tabButton.eventMode = 'static';
      tabButton.cursor = 'pointer';

      // Tab center (snap to ints)
      const cx = Math.round(this.boxX + idx * tabWidth + tabWidth / 2);
      const cy = Math.round(this.boxY + (TAB_HEIGHT + 10) / 2);

      // Create main label
      const label = new PIXI.Text(
        type.toUpperCase(),
        idx === this.activeTab ? tabStyleActive : tabStyle
      );
      label.roundPixels = true;
      label.anchor.set(0.5, 0.5);

      // Create shadow label
      const shadowLabel = new PIXI.Text(
        type.toUpperCase(),
        idx === this.activeTab ? tabStyleActive : tabStyle
      );
      shadowLabel.roundPixels = true;
      shadowLabel.anchor.set(0.5, 0.5);
      shadowLabel.tint = 0x000000;
      shadowLabel.alpha = 0.1;

      // Helper to crisp-center a PIXI.Text at (cx, cy) with odd-size nudge
      const crispCenter = (txt) => {
        // force metrics read so width/height reflect current style
        const w = Math.round(txt.width);
        const h = Math.round(txt.height);
        const oddW = (w % 2) === 1;
        const oddH = (h % 2) === 1;
        txt.position.set(cx + (oddW ? 0.5 : 0), cy + (oddH ? 0.5 : 0));
      };

      // Initial positioning
      crispCenter(label);
      // Shadow sits +1,+1 from main label (keeps the same fractional alignment)
      shadowLabel.position.set(label.x + 1, label.y + 1);

      // Store references
      this.tabButtons[idx] = {
        bg: tabButton,
        label,
        shadowLabel,
        type,
        index: idx,
        container: tabContainer
      };

      // Hover effect (re-center after style change)
      tabButton.on('pointerover', () => {
        if (idx !== this.activeTab) {
          this.drawTab(tabButton, idx, false, true);
          label.style = tabStyleActive;
          label.alpha = 0.8;
          crispCenter(label);
          shadowLabel.position.set(label.x + 1, label.y + 1);
        }
      });

      tabButton.on('pointerout', () => {
        if (idx !== this.activeTab) {
          this.drawTab(tabButton, idx, false);
          label.style = tabStyle;
          label.alpha = 1;
          crispCenter(label);
          shadowLabel.position.set(label.x + 1, label.y + 1);
        }
      });

      // Click handler
      tabButton.on('pointerdown', () => {
        this.switchTab(idx);
      });

      tabContainer.addChild(tabButton, shadowLabel, label);
      this.lTabs.addChild(tabContainer);
    });

    /* ── Content area decorations ─────────────── */
    const contentDecor = new PIXI.Graphics();
    
    // Top flourish
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
    const tabWidth = Math.round(this.boxWidth / VICTORY_TYPES.length);
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

      // Side dividers for active tab
      if (index > 0) {
        graphics.lineStyle(1, COLORS.goldDark);
        graphics.moveTo(x, y + 5);
        graphics.lineTo(x, y + h - 5);
        graphics.stroke();
      }
      if (index < VICTORY_TYPES.length - 1) {
        graphics.lineStyle(1, COLORS.goldDark);
        graphics.moveTo(x + tabWidth, y + 5);
        graphics.lineTo(x + tabWidth, y + h - 5);
        graphics.stroke();
      }

    } else {
      graphics.beginFill(COLORS.tabInactive, 1.0);
      graphics.drawRect(x, y, tabWidth, h);
      graphics.endFill();

      // Subtle hover tint on top (keeps the gray base visible)
      if (isHover) {
        graphics.beginFill(COLORS.stoneLight, 0.08);
        graphics.drawRect(x, y, tabWidth, h);
        graphics.endFill();
      }

      // Right-hand divider to separate tabs
      if (index < VICTORY_TYPES.length - 1) {
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
    
    // Update previous active tab
    const prevTab = this.tabButtons[this.activeTab];
    this.drawTab(prevTab.bg, this.activeTab, false);
    prevTab.label.style = tabStyle;
    prevTab.shadowLabel.style = tabStyle;
    
    // Update new active tab
    this.activeTab = index;
    const newTab = this.tabButtons[index];
    this.drawTab(newTab.bg, index, true);
    newTab.label.style = tabStyleActive;
    newTab.shadowLabel.style = tabStyleActive;
    
    // Redraw content
    this.redrawContent();
  }

  /* =========================================================
       CONTENT RENDERING
     ========================================================= */
  redrawContent() {
    // Clear existing content
    this.lContent.removeChildren();
    
    const victoryType = VICTORY_TYPES[this.activeTab];
    
    // Content header with shadow
    const headerY = this.boxY + TAB_HEIGHT + 45;
    
    const contentHeader = new PIXI.Text(
      `${victoryType.toUpperCase()} VICTORY · TURN ${this.turn}`,
      contentHeaderStyle
    );
    contentHeader.roundPixels = true;  // per-object rounding
    contentHeader.anchor.set(0, 0);    // avoid the 0.5 anchor blur trap
    const centerX = Math.round(this.boxX + this.boxWidth / 2);
    contentHeader.x = Math.round(centerX - contentHeader.width / 2);
    contentHeader.y = headerY;
    this.lContent.addChild(contentHeader);
    
    // Content area bounds
    const contentX = this.boxX + 40;
    const contentY = this.boxY + TAB_HEIGHT + 90;
    const contentW = this.boxWidth - 80;
    const contentH = this.boxHeight - TAB_HEIGHT - 120;
    
    // Create content based on victory type
    switch(victoryType) {
      case 'Cultural':
        this.renderCulturalVictory(contentX, contentY, contentW, contentH);
        break;
      case 'Diplomatic':
        this.renderDiplomaticVictory(contentX, contentY, contentW, contentH);
        break;
      case 'Scientific':
        this.renderScientificVictory(contentX, contentY, contentW, contentH);
        break;
      case 'Domination':
        this.renderDominationVictory(contentX, contentY, contentW, contentH);
        break;
    }
  }

  /* =========================================================
       VICTORY TYPE RENDERERS
     ========================================================= */
  renderCulturalVictory(x, y, w, h) {
    const container = new PIXI.Container();

    // ---- Description ----
    const desc = new PIXI.Text(
      'Achieve victory through cultural influence over other civilizations.',
      descStyle
    );
    desc.style.wordWrapWidth = w;
    desc.position.set(x, y);
    container.addChild(desc);

    // ---- Requirements ----
    const reqTitle = new PIXI.Text('REQUIREMENTS', requirementStyle);
    reqTitle.position.set(x, y + 40);
    container.addChild(reqTitle);

    const requirements = [
      '• Generate more tourism than the culture of every other civilization.'
    ];

    let yOffset = y + 70;
    requirements.forEach(req => {
      const reqText = new PIXI.Text(req, statusStyle);
      reqText.position.set(x + 20, yOffset);
      container.addChild(reqText);
      yOffset += 25;
    });

    // ---- Divider ----
    const divider = new PIXI.Graphics();
    divider.lineStyle(1, COLORS.goldDark, 0.5);
    divider.moveTo(x, yOffset + 15);
    divider.lineTo(x + w, yOffset + 15);
    divider.stroke();
    container.addChild(divider);

    yOffset += 35;

    // ========== SUB-TABS ==========
    if (this.culturalSubTab === undefined) {
      // 0 = Overview, 1 = Great Works, 2 = Great People
      this.culturalSubTab = 0;
    }

    const subTabLabels = ['OVERVIEW', 'GREAT WORKS', 'GREAT PEOPLE'];
    const subTabWidth  = (w - 40) / subTabLabels.length;
    const subTabX      = x + 20;
    const subTabY      = yOffset;
    const subTabHeight = 35;

    // Background strip
    const subTabBg = new PIXI.Graphics();
    subTabBg.beginFill(COLORS.stoneBg, 0.2);
    subTabBg.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBg.endFill();
    container.addChild(subTabBg);

    // Tabs
    subTabLabels.forEach((label, idx) => {
      const tabG = new PIXI.Graphics();
      const tabX = subTabX + idx * subTabWidth;

      if (idx === this.culturalSubTab) {
        // Active tab
        tabG.beginFill(COLORS.goldDark, 0.3);
        tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
        tabG.endFill();

        // Gold top accent
        tabG.beginFill(COLORS.goldBright);
        tabG.drawRect(tabX, subTabY, subTabWidth, 2);
        tabG.endFill();

        // Side borders
        tabG.lineStyle(1, COLORS.goldDark);
        tabG.moveTo(tabX, subTabY);
        tabG.lineTo(tabX, subTabY + subTabHeight);
        tabG.moveTo(tabX + subTabWidth, subTabY);
        tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight);
        tabG.stroke();
      } else {
        // Inactive separator
        if (idx < subTabLabels.length - 1) {
          tabG.lineStyle(1, COLORS.divider, 0.5);
          tabG.moveTo(tabX + subTabWidth, subTabY + 5);
          tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight - 5);
          tabG.stroke();
        }
      }

      // Hit area
      tabG.eventMode = 'static';
      tabG.cursor = 'pointer';
      tabG.beginFill(0x000000, 0.01);
      tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
      tabG.endFill();

      // Label
      const subTabStyle = idx === this.culturalSubTab
        ? { ...requirementStyle, fill: COLORS.goldBright }
        : { ...statusStyle, fill: COLORS.goldDark };

      const subTabText = new PIXI.Text(label, subTabStyle);
      subTabText.roundPixels = true;
      subTabText.anchor.set(0.5, 0.5);

      const cx = Math.round(tabX + subTabWidth / 2);
      const cy = Math.round(subTabY + subTabHeight / 2);
      const oddW = (Math.round(subTabText.width)  % 2) === 1;
      const oddH = (Math.round(subTabText.height) % 2) === 1;
      subTabText.position.set(cx + (oddW ? 0.5 : 0), cy + (oddH ? 0.5 : 0));

      // Hover
      tabG.on('pointerover', () => {
        if (idx !== this.culturalSubTab) {
          subTabText.style.fill = COLORS.goldBright;
          subTabText.alpha = 0.7;
        }
      });
      tabG.on('pointerout', () => {
        if (idx !== this.culturalSubTab) {
          subTabText.style.fill = COLORS.goldDark;
          subTabText.alpha = 1;
        }
      });

      // Click
      tabG.on('pointerdown', () => {
        this.culturalSubTab = idx;
        this.redrawContent();
      });

      container.addChild(tabG);
      container.addChild(subTabText);
    });

    // Bottom border
    const subTabBorder = new PIXI.Graphics();
    subTabBorder.lineStyle(1, COLORS.goldDark);
    subTabBorder.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBorder.stroke();
    container.addChild(subTabBorder);

    // ========== CONTENT AREA ==========
    const contentY = subTabY + subTabHeight + 20;
    const contentH = h - (contentY - y) - 20;

    switch (this.culturalSubTab) {
      case 0:
        this.renderCulturalOverview(container, x, contentY, w, contentH);
        break;
      case 1:
        this.renderGreatWorks(container, x, contentY, w, contentH);
        break;
      case 2:
        this.renderGreatPeople(container, x, contentY, w, contentH);
        break;
    }

    this.lContent.addChild(container);
  }

  /* ========================================
     Cultural Overview — 6×6 Player Matrix
     ======================================== */
  renderCulturalOverview(container, x, y, w, h) {

    const top = Math.round(y + 30);
    const leftPad = 20, rightPad = 20;
    const innerW = Math.max(0, w - leftPad - rightPad);

    // Table geometry
    const rowLabels = ['PLAYER 1','PLAYER 2','PLAYER 3','PLAYER 4','PLAYER 5','PLAYER 6'];
    const colLabels = rowLabels;
    const headerH = 26;
    const rowH = 30;
    const headerW = 140;
    const cellW = Math.floor((innerW - headerW) / 6);
    const tableW = headerW + cellW * 6;
    const tableH = headerH + rowH * 6;

    const tableX = Math.round(x + leftPad + Math.floor((innerW - tableW) / 2));
    const tableY = top;

    // Data (assumes setData() already ran)
    const tourism = this.tourismTotal[this.turn];
    const culture = this.cultureTotal[this.turn];

    // Table background + border
    const box = new PIXI.Graphics();
    box.beginFill(COLORS.stoneBg, 0.20);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.endFill();
    box.lineStyle(1, COLORS.goldDark, 0.5);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.stroke();
    container.addChild(box);

    // Column headers
    colLabels.forEach((lbl, j) => {
      const tx = new PIXI.Text(lbl, statusStyle);
      tx.roundPixels = true;
      tx.anchor.set(0.5, 0.5);
      const cx = Math.round(tableX + headerW + j * cellW + cellW / 2);
      const cy = Math.round(tableY + headerH / 2);
      tx.position.set(cx, cy);
      container.addChild(tx);
    });
    
    // underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.5);
    headerLine.moveTo(Math.round(x + leftPad + 2), Math.round(top + headerH - 2));
    headerLine.lineTo(Math.round(x + leftPad + tableW + 2), Math.round(top + headerH - 2));
    headerLine.stroke();
    container.addChild(headerLine);

    // Rows
    for (let i = 0; i < 6; i++) {
      // Row header
      const rh = new PIXI.Text(rowLabels[i], statusStyle);
      rh.roundPixels = true;
      rh.anchor.set(0, 0.5);
      const ry = Math.round(tableY + headerH + i * rowH + rowH / 2);
      rh.position.set(Math.round(tableX + 32), ry);
      container.addChild(rh);

      for (let j = 0; j < 6; j++) {
        const cellX = Math.round(tableX + headerW + j * cellW);
        const cellY = Math.round(tableY + headerH + i * rowH);

        // Off-diagonal: progress bar like the City-States view
        if (i !== j) {
          const num = tourism[i][j];
          const den = culture[j];

          // Compute ratio and percentage text
          const progress = den > 0 ? (num / den) : 0; // may exceed 1
          const pct = Math.round(progress * 100);

          // Bar geometry inside the cell
          const insetX = 6;
          const insetY = 5;
          const barX = cellX + insetX;
          const barY = cellY + insetY;
          const barW = cellW - insetX * 2;
          const barH = rowH - insetY * 2;

          // Background
          const bar = new PIXI.Graphics();
          bar.beginFill(COLORS.divider, 0.2);
          bar.drawRect(barX, barY, barW, barH);
          bar.endFill();

          const color =
            progress >= 1.0 ? COLORS.success :
            progress >= 0.5 ? COLORS.pending :
                              COLORS.failed;

          // Fill (clamped to width)
          const fillW = Math.max(0, Math.min(barW, Math.floor(barW * progress)));
          if (fillW > 0) {
            bar.beginFill(color, 0.6);
            bar.drawRect(barX, barY, fillW, barH);
            bar.endFill();
          }

          // Border for progress bar
          bar.lineStyle(1, COLORS.goldDark, 0.3);
          bar.drawRect(barX, barY, barW, barH);
          bar.stroke();

          container.addChild(bar);

          // Percentage text overlay (mirrors CS value label style)
          const valueText = new PIXI.Text(`${pct}%`, {
            fontFamily: 'Crimson Text',
            fontSize: 16,
            fontWeight: pct >= 100 ? 'bold' : '600',
            fill: 0xffffff,
            dropShadow: progress >= 0.5,
            dropShadowColor: 0x000000,
            dropShadowDistance: 1,
            dropShadowAlpha: 0.8
          });
          valueText.anchor.set(0.5, 0.5);
          valueText.position.set(Math.round(barX + barW / 2), Math.round(barY + barH / 2));
          container.addChild(valueText);

        } else {
          // Diagonal — show dash to indicate N/A
          const na = new PIXI.Text('—', {
            fontFamily: 'Cinzel',
            fontSize: 14,
            fontWeight: 600,
            fill: COLORS.goldDark
          });
          na.anchor.set(0.5, 0.5);
          na.position.set(Math.round(cellX + cellW / 2), Math.round(cellY + rowH / 2));
          container.addChild(na);
        }
      }
    }

    // Axis hints
    const axisLeft = new PIXI.Text('FROM', { ...statusStyle, fontSize: 18 });
    axisLeft.roundPixels = true;
    axisLeft.anchor.set(0, 0.5);  // left-center as the pivot
    axisLeft.rotation = -Math.PI / 2  // then rotate 90' counter-clockwise
    axisLeft.position.set(Math.round(tableX) - 12, Math.round(tableY - 18) + 140);
    axisLeft.alpha = 0.8;
    container.addChild(axisLeft);

    const axisTop = new PIXI.Text('TO', { ...statusStyle, fontSize: 12 });
    axisTop.roundPixels = true;
    axisTop.anchor.set(1, 0);
    axisTop.position.set(Math.round(tableX + tableW) - 520, Math.round(tableY - 22));
    axisTop.alpha = 0.8;
    container.addChild(axisTop);
  }
  



  /* =======================
     Great Works (stub)
     ======================= */

  renderGreatWorks(container, x, y, w, h) {
    // Build data on demand
    const countsP4 = this._getGWCountsForTurn(this.turn);   // shape (P, 4)
    console.log(countsP4);
    const numPlayers = countsP4.length || 6;

    // Guard: no data yet
    if (!countsP4.length) {
      const msg = new PIXI.Text('No Great Works data available.', statusStyle);
      msg.position.set(Math.round(x), Math.round(y));
      container.addChild(msg);
      return;
    }

    // Labels
    const rowLabels = ['WRITING', 'ART', 'MUSIC']; // ignore index 3 (unused)
    const colLabels = Array.from({ length: numPlayers }, (_, i) => `PLAYER ${i + 1}`);

    // Geometry
    const top = Math.round(y + 30);
    const leftPad = 20, rightPad = 20;
    const innerW = Math.max(0, w - leftPad - rightPad);

    const headerH = 26;
    const rowH = 32;
    const nameColW = 150;
    const cellW = Math.floor((innerW - nameColW) / numPlayers);
    const tableW = nameColW + cellW * numPlayers;
    const tableH = headerH + rowH * rowLabels.length;

    const tableX = Math.round(x + leftPad + Math.floor((innerW - tableW) / 2));
    const tableY = top;

    // Background + frame
    const box = new PIXI.Graphics();
    box.beginFill(COLORS.stoneBg, 0.20);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.endFill();
    box.lineStyle(1, COLORS.goldDark, 0.5);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.stroke();
    container.addChild(box);
    
    // underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.5);
    headerLine.moveTo(Math.round(x + leftPad + 1), Math.round(top + headerH - 1));
    headerLine.lineTo(Math.round(x + leftPad + tableW + 1), Math.round(top + headerH - 1));
    headerLine.stroke();
    container.addChild(headerLine);

    // Column headers
    for (let j = 0; j < numPlayers; j++) {
      //const tx = new PIXI.Text(colLabels[j], { ...requirementStyle, fontSize: 14, fill: COLORS.goldDark });
      const tx = new PIXI.Text(colLabels[j], statusStyle);
      tx.roundPixels = true;
      tx.anchor.set(0.5, 0.5);
      const cx = Math.round(tableX + nameColW + j * cellW + cellW / 2);
      const cy = Math.round(tableY + headerH / 2);
      tx.position.set(cx, cy);
      container.addChild(tx);
    }

    // Rows
    for (let i = 0; i < rowLabels.length; i++) {
      const ry = Math.round(tableY + headerH + i * rowH + rowH / 2);

      // Row header
      //const rh = new PIXI.Text(rowLabels[i], { ...requirementStyle, fontSize: 14, fill: COLORS.goldDark });
      const rh = new PIXI.Text(rowLabels[i], statusStyle);
      rh.roundPixels = true;
      rh.anchor.set(0, 0.5);
      rh.position.set(Math.round(tableX + 40), ry);
      container.addChild(rh);

      // Cells
      for (let j = 0; j < numPlayers; j++) {
        const cellX = Math.round(tableX + nameColW + j * cellW);
        const cellY = Math.round(tableY + headerH + i * rowH);

        // Background (zebra)
        const cellBg = new PIXI.Graphics();
        const baseAlpha = (i % 2 === 0) ? 0.10 : 0.06;
        cellBg.beginFill(COLORS.stoneBg, baseAlpha);
        cellBg.drawRect(cellX, cellY, cellW, rowH);
        cellBg.endFill();

        // Border
        cellBg.lineStyle(1, COLORS.goldDark, 0.25);
        cellBg.drawRect(cellX, cellY, cellW, rowH);
        cellBg.stroke();
        container.addChild(cellBg);

        // Value: countsP4[j][i]
        const v = (countsP4[j] && countsP4[j][i]) ? countsP4[j][i] : 0;
        const txt = new PIXI.Text(String(v), {
          ...statusStyle,
          fontSize: 16,
          fontWeight: 700,
          //fill: v > 0 ? 0xffffff : COLORS.goldDark,
          dropShadow: v > 0,
          dropShadowColor: 0x000000,
          dropShadowDistance: 1,
          dropShadowAlpha: 0.8
        });
        txt.anchor.set(0.5, 0.5);
        txt.position.set(Math.round(cellX + cellW / 2), Math.round(cellY + rowH / 2));
        container.addChild(txt);
      }
    }
  }

  /* =======================
     Great People (stub)
     ======================= */
  renderGreatPeople(container, x, y, w, h) {
    const gppsThisTurn = this.gpps[this.turn];
    const gpThresholdThisTurn = this.gpThreshold[this.turn];

    const gpTypes = ['ARTIST','MUSICIAN','WRITER','ENGINEER','MERCHANT','SCIENTIST'];
    const playerLabels = ['PLAYER 1','PLAYER 2','PLAYER 3','PLAYER 4','PLAYER 5','PLAYER 6'];

    // ---- geometry ----
    const top = Math.round(y + 30);
    const leftPad = 20, rightPad = 20;
    const headerH = 26;
    const rowH = 30;
    const nameColW = 150; // room for "PLAYER X"
    const innerW = Math.max(0, w - leftPad - rightPad);
    const cellW = Math.floor((innerW - nameColW) / gpTypes.length);
    const tableW = nameColW + cellW * gpTypes.length;
    const tableH = headerH + rowH * 6;

    const tableX = Math.round(x + leftPad + Math.floor((innerW - tableW) / 2));
    const tableY = top;
  
    // Table background + border
    const box = new PIXI.Graphics();
    box.beginFill(COLORS.stoneBg, 0.20);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.endFill();
    box.lineStyle(1, COLORS.goldDark, 0.5);
    box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
    box.stroke();
    container.addChild(box);

    gpTypes.forEach((label, j) => {
      const tx = new PIXI.Text(label, {
        ...statusStyle,
        fontSize: 14,
        fill: COLORS.goldDark,
        fontWeight: 700  // avoid faux 600 which blurs
      });
      tx.roundPixels = true;
      tx.anchor.set(0.5, 0);
      const hx = Math.round(x + leftPad + nameColW + j * cellW + cellW / 2);
      tx.position.set(hx, Math.round(top));  // whole pixels
      container.addChild(tx);
    });

    // underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.5);
    headerLine.moveTo(Math.round(x + leftPad), Math.round(top + headerH - 6));
    headerLine.lineTo(Math.round(x + leftPad + tableW), Math.round(top + headerH - 6));
    headerLine.stroke();
    container.addChild(headerLine);

    // ---- rows ----
    const rowsY0 = top + headerH;
    const barHeight = 18;
    const barSidePad = 6;  // inside each cell

    for (let p = 0; p < 6; p++) {
      const rowY = Math.round(rowsY0 + p * rowH);

      // alternating background
      if (p % 2 === 0) {
        const bg = new PIXI.Graphics();
        bg.beginFill(COLORS.stoneBg, 0.08);
        bg.drawRect(
          Math.round(x + leftPad - 5),
          Math.round(rowY - 3),
          Math.round(tableW + 10),
          Math.round(rowH - 2)
        );
        bg.endFill();
        container.addChild(bg);
      }

      // player name
      const nameText = new PIXI.Text(playerLabels[p], statusStyle);
      nameText.position.set(Math.round(x + leftPad + 40), rowY + 3);
      container.addChild(nameText);

      const thresh = Math.max(1, Math.floor(gpThresholdThisTurn?.[p] ?? 1));

      for (let j = 0; j < gpTypes.length; j++) {
        const pts = Math.max(0, Math.floor(gppsThisTurn?.[p]?.[j] ?? 0));
        const ratio = Math.min(1, pts / thresh);

        // cell rect
        const cx = Math.round(x + leftPad + nameColW + j * cellW);
        const barW = Math.max(0, cellW - 2 * barSidePad);
        const barX = cx + barSidePad;
        const barY = Math.round(rowY + (rowH - barHeight) / 2);

        // bar background
        const g = new PIXI.Graphics();
        g.beginFill(COLORS.divider, 0.20);
        g.drawRect(barX, barY, barW, barHeight);
        g.endFill();

        // bar fill color
        const barColor =
          ratio >= 1 ? COLORS.success :
          ratio >= 0.75 ? 0x3498db :      // pushing toward threshold
          COLORS.goldDark;

        // fill
        const fillW = Math.floor(barW * ratio);
        if (fillW > 0) {
          g.beginFill(barColor, 0.65);
          g.drawRect(barX, barY, fillW, barHeight);
          g.endFill();
        }

        // border
        g.lineStyle(1, COLORS.goldDark, 0.35);
        g.drawRect(barX, barY, barW, barHeight);
        g.stroke();
        container.addChild(g);

        //const cy = Math.round(barY + barHeight / 2);
        
        const txt = new PIXI.Text(`${pts}/${thresh}`, {
          fontFamily: 'Crimson Text',
          fontSize: 14,
          fontWeight: ratio >= 1 ? 700 : 600, // prefer 700 over 600 for crisp
          fill: 0xffffff,
          dropShadow: ratio >= 0.75,
          dropShadowColor: 0x000000,
          dropShadowDistance: 1,
          dropShadowAlpha: 0.85
        });
        txt.roundPixels = true;
        txt.anchor.set(0.5, 0.5);
        txt.position.set(Math.round(barX + barW / 2), Math.round(barY + barHeight / 2));                  
        container.addChild(txt);
      }
    }
  }

  renderDiplomaticVictory(x, y, w, h) {
    const container = new PIXI.Container();
    
    const desc = new PIXI.Text(
      'Win through diplomatic dominance in the World Congress.',
      descStyle
    );
    desc.style.wordWrapWidth = w;
    desc.position.set(x, y);
    container.addChild(desc);
    
    // Requirements
    const reqTitle = new PIXI.Text('REQUIREMENTS', requirementStyle);
    reqTitle.position.set(x, y + 40);
    container.addChild(reqTitle);
    
    const requirements = [
      '• Control a majority of delegates during a World Congress Vote.',
    ];
    
    let yOffset = y + 70;
    requirements.forEach(req => {
      const reqText = new PIXI.Text(req, statusStyle);
      reqText.position.set(x + 20, yOffset);
      container.addChild(reqText);
      yOffset += 25;
    });
    
    // Divider
    const divider = new PIXI.Graphics();
    divider.lineStyle(1, COLORS.goldDark, 0.5);
    divider.moveTo(x, yOffset + 15)
    divider.lineTo(x + w, yOffset + 15);
    divider.stroke();
    container.addChild(divider);
    
    yOffset += 35;
    
    // ========== SUB-TABS SYSTEM ==========
    // Initialize diplomatic sub-tab state if not exists
    if (this.diplomaticSubTab === undefined) {
      this.diplomaticSubTab = 0; // 0 = Overview, 1 = City-States, 2 = Quests
    }
    
    const subTabLabels = ['OVERVIEW', 'CITY STATES', 'QUESTS'];
    const subTabWidth = (w - 40) / subTabLabels.length;
    const subTabX = x + 20;
    const subTabY = yOffset;
    const subTabHeight = 35;
    
    // Sub-tab background
    const subTabBg = new PIXI.Graphics();
    subTabBg.beginFill(COLORS.stoneBg, 0.2);
    subTabBg.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBg.endFill();
    container.addChild(subTabBg);
    
    // Create sub-tabs
    subTabLabels.forEach((label, idx) => {
      const tabG = new PIXI.Graphics();
      const tabX = subTabX + idx * subTabWidth;
      
      // Draw sub-tab
      if (idx === this.diplomaticSubTab) {
        // Active sub-tab
        tabG.beginFill(COLORS.goldDark, 0.3);
        tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
        tabG.endFill();
        
        // Gold accent line on top
        tabG.beginFill(COLORS.goldBright);
        tabG.drawRect(tabX, subTabY, subTabWidth, 2);
        tabG.endFill();
        
        // Side borders
        tabG.lineStyle(1, COLORS.goldDark);
        tabG.moveTo(tabX, subTabY);
        tabG.lineTo(tabX, subTabY + subTabHeight);
        tabG.moveTo(tabX + subTabWidth, subTabY);
        tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight);
        tabG.stroke();
      } else {
        // Inactive sub-tab - just divider
        if (idx < subTabLabels.length - 1) {
          tabG.lineStyle(1, COLORS.divider, 0.5);
          tabG.moveTo(tabX + subTabWidth, subTabY + 5);
          tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight - 5);
          tabG.stroke();
        }
      }
      
      // Make interactive
      tabG.eventMode = 'static';
      tabG.cursor = 'pointer';
      tabG.beginFill(0x000000, 0.01);
      tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
      tabG.endFill();
      
      // Sub-tab label
      const subTabStyle = idx === this.diplomaticSubTab ? 
        { ...requirementStyle, fill: COLORS.goldBright } : 
        { ...statusStyle, fill: COLORS.goldDark };
      
      const subTabText = new PIXI.Text(label, subTabStyle);
      subTabText.roundPixels = true;          // per-object pixel snapping
      subTabText.anchor.set(0.5, 0.5);
      // compute the tab center, snapped to whole pixels
      const cx = Math.round(tabX + subTabWidth / 2);
      const cy = Math.round(subTabY + subTabHeight / 2);

      // if the rendered text has odd pixel width/height, nudge by 0.5 to keep edges on whole pixels
      const oddW = (Math.round(subTabText.width)  % 2) === 1;
      const oddH = (Math.round(subTabText.height) % 2) === 1;

      // final position
      subTabText.position.set(cx + (oddW ? 0.5 : 0), cy + (oddH ? 0.5 : 0));
      //subTabText.position.set(tabX + subTabWidth / 2, subTabY + subTabHeight / 2);
      
      // Hover effect
      tabG.on('pointerover', () => {
        if (idx !== this.diplomaticSubTab) {
          subTabText.style.fill = COLORS.goldBright;
          subTabText.alpha = 0.7;
        }
      });
      
      tabG.on('pointerout', () => {
        if (idx !== this.diplomaticSubTab) {
          subTabText.style.fill = COLORS.goldDark;
          subTabText.alpha = 1;
        }
      });
      
      // Click handler
      tabG.on('pointerdown', () => {
        this.diplomaticSubTab = idx;
        this.redrawContent(); // Redraw to show new sub-tab content
      });
      
      container.addChild(tabG);
      container.addChild(subTabText);
    });
    
    // Bottom border for sub-tabs
    const subTabBorder = new PIXI.Graphics();
    subTabBorder.lineStyle(1, COLORS.goldDark);
    subTabBorder.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBorder.stroke();
    container.addChild(subTabBorder);
    
    // ========== SUB-TAB CONTENT ==========
    const contentY = subTabY + subTabHeight + 20;
    const contentH = h - (contentY - y) - 20;
    
    switch(this.diplomaticSubTab) {
      case 0: // Overview
        this.renderDiplomaticOverview(container, x, contentY, w, contentH);
        break;
      case 1: // City-States
        this.renderCityStateRelations(container, x, contentY, w, contentH);
        break;
      case 2: // Quests
        this.renderActiveQuests(container, x, contentY, w, contentH);
        break;
    }
    
    this.lContent.addChild(container);
  }
 
  renderDiplomaticOverview(container, x, y, w, h) {
    // ---------- Data ----------
    const turn = this.turn ?? 0;
    const delegates = (this.numDelegates && this.numDelegates[turn]) ? this.numDelegates[turn] : Array(6).fill(0);
    const csRelTurn = (this.csRelationships && this.csRelationships[turn]) ? this.csRelationships[turn] : []; // shape (numCS, 6)

    // Count city-state allies per player (relationship level === 2)
    const allies = new Array(6).fill(0);
    for (let i = 0; i < csRelTurn.length; i++) {
      const row = csRelTurn[i] || [];
      for (let p = 0; p < 6; p++) {
        if ((row[p] | 0) === 2) allies[p]++;
      }
    }

    // Totals and ratios (cap at 10)
    const TOTAL_CAP = 12;
    const totals = Array.from({ length: 6 }, (_, p) => (Number(delegates[p] || 0) + allies[p]));
    const ratios = totals.map(v => Math.min(1, v / TOTAL_CAP));

    // ---------- Geometry ----------
    const top = Math.round(y);
    const leftPad = 20, rightPad = 20;
    const innerW = Math.max(0, w - leftPad - rightPad);

    const headerH = 28;        // header row height
    const rowH    = 42;        // content row height
    const cols    = 6;
    const cellW   = Math.floor(innerW / cols);
    const tableW  = cellW * cols;
    const tableX  = Math.round(x + leftPad + Math.floor((innerW - tableW) / 2));
    const tableY  = top + 6;

    // ---------- Table background & frame ----------
    const box = new PIXI.Graphics();
    box.beginFill(COLORS.stoneBg, 0.20);
    box.drawRoundedRect(tableX, tableY, tableW, headerH + rowH, 0);
    box.endFill();
    box.lineStyle(1, COLORS.goldDark, 0.5);
    box.drawRoundedRect(tableX, tableY, tableW, headerH + rowH, 0);
    box.stroke();
    container.addChild(box);

    // Header underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.35);
    headerLine.moveTo(tableX, Math.round(tableY + headerH));
    headerLine.lineTo(tableX + tableW, Math.round(tableY + headerH));
    headerLine.stroke();
    container.addChild(headerLine);

    // ---------- Column headers ----------
    for (let j = 0; j < cols; j++) {
      //const tx = new PIXI.Text(`PLAYER ${j + 1}`, { ...requirementStyle, fontSize: 14, fill: COLORS.goldDark });
      const tx = new PIXI.Text(`PLAYER ${j + 1}`, statusStyle);
      tx.roundPixels = true;
      tx.anchor.set(0.5, 0.5);
      const cx = Math.round(tableX + j * cellW + cellW / 2);
      const cy = Math.round(tableY + headerH / 2);
      tx.position.set(cx, cy);
      container.addChild(tx);
    }

    // ---------- Content row: progress bars ----------
    for (let j = 0; j < cols; j++) {
      const cellX = Math.round(tableX + j * cellW);
      const cellY = Math.round(tableY + headerH);

      // Cell background (subtle zebra not needed with one row; keep a faint base)
      const cellBg = new PIXI.Graphics();
      cellBg.beginFill(COLORS.stoneBg, 0.08);
      cellBg.drawRect(cellX, cellY, cellW, rowH);
      cellBg.endFill();
      cellBg.lineStyle(1, COLORS.goldDark, 0.25);
      cellBg.drawRect(cellX, cellY, cellW, rowH);
      cellBg.stroke();
      container.addChild(cellBg);

      // Bar geometry inside the cell
      const insetX = 8;
      const insetY = 8;
      const barX = cellX + insetX;
      const barY = cellY + insetY;
      const barW = cellW - insetX * 2;
      const barH = rowH - insetY * 2;

      // Background
      const bar = new PIXI.Graphics();
      bar.beginFill(COLORS.divider, 0.22);
      bar.drawRect(barX, barY, barW, barH);
      bar.endFill();

      // Fill color thresholds (mirror CS style)
      const r = ratios[j];
      const color =
        r >= 1.0 ? COLORS.success :
        r >= 0.5 ? COLORS.pending :
                   COLORS.failed;

      // Fill
      const fillW = Math.max(0, Math.min(barW, Math.floor(barW * r)));
      if (fillW > 0) {
        bar.beginFill(color, 0.65);
        bar.drawRect(barX, barY, fillW, barH);
        bar.endFill();
      }

      // Border
      bar.lineStyle(1, COLORS.goldDark, 0.35);
      bar.drawRect(barX, barY, barW, barH);
      bar.stroke();
      container.addChild(bar);

      // Value text overlay: "<total>/10  (D + A)"
      const D = Number(delegates[j] || 0);
      const A = allies[j];
      const valText = new PIXI.Text(`${totals[j]}/${TOTAL_CAP}  (${D} + ${A})`, {
        fontFamily: 'Crimson Text',
        fontSize: 16,
        fontWeight: r >= 1 ? 'bold' : '600',
        fill: 0xffffff,
        dropShadow: r >= 0.5,
        dropShadowColor: 0x000000,
        dropShadowDistance: 1,
        dropShadowAlpha: 0.85
      });
      valText.anchor.set(0.5, 0.5);
      valText.position.set(Math.round(barX + barW / 2), Math.round(barY + barH / 2));
      container.addChild(valText);
    }
  }

  renderCityStateRelations(container, x, y, w, h) {
    const title = new PIXI.Text('CITY STATE RELATIONSHIPS', requirementStyle);
    title.position.set(x, y);
    container.addChild(title);
    
    // Map integer types to string names
    const typeMapping = {
      0: 'Cultural',
      1: 'Agricultural', 
      2: 'Mercantile',
      3: 'Religious',
      4: 'Scientific',
      5: 'Militaristic'
    };
    
    // Build city-states array from actual data
    const cityStates = [];
    const csTypes = this.csType?.[this.turn] || [];
    
    // Import actual CS names from constants
    const csNames = constants.csNames
    
    // Create city-state objects with actual data
    for (let i = 0; i < Math.min(12, csNames.length); i++) {
      cityStates.push({
        name: csNames[i],
        type: typeMapping[csTypes[i]] || 'Unknown',
        influence: 60, // Will be replaced with actual data
        status: 'Ally', // Will be replaced with actual data
        leader: 'Player I' // Will be replaced with actual data
      });
    }
    
    let yOffset = y + 35;
    // Headers (remain fixed, not scrolled)
    const headers = [
      { text: 'CITY STATE', x: x + 20 },
      { text: 'TYPE', x: x + 180 },
      { text: 'PLAYER 1', x: x + 330 },
      { text: 'PLAYER 2', x: x + 460 },
      { text: 'PLAYER 3', x: x + 590 },
      { text: 'PLAYER 4', x: x + 720 },
      { text: 'PLAYER 5', x: x + 840 },
      { text: 'PLAYER 6', x: x + 970 },
    ];
    
    headers.forEach(header => {
      const headerText = new PIXI.Text(header.text, {
        ...statusStyle,
        fontSize: 14,
        fill: COLORS.goldDark
      });
      headerText.position.set(header.x, yOffset);
      container.addChild(headerText);
    });
    
    // Header underline
    const headerLine = new PIXI.Graphics();
    headerLine.lineStyle(1, COLORS.goldDark, 0.5);
    headerLine.moveTo(x + 20, yOffset + 20);
    headerLine.lineTo(x + w - 40, yOffset + 20);
    headerLine.stroke();
    container.addChild(headerLine);
    
    yOffset += 35;
    
    // Calculate visible area
    const rowHeight = 30;
    const visibleHeight = h - (yOffset - y) - 20;
    const maxVisibleRows = Math.floor(visibleHeight / rowHeight);
    const actualVisibleHeight = maxVisibleRows * rowHeight;
    const totalContentHeight = cityStates.length * rowHeight;
    const needsScroll = totalContentHeight > actualVisibleHeight;
    const maxScroll = Math.max(0, totalContentHeight - actualVisibleHeight);
    
    // Initialize scroll position if not exists
    if (this.csScrollY === undefined) {
      this.csScrollY = 0;
    }
    
    // Ensure scroll is within bounds
    this.csScrollY = Math.max(0, Math.min(maxScroll, this.csScrollY));
    
    // Create container for rows and apply mask
    const rowsContainer = new PIXI.Container();
    
    // Create and apply mask
    const mask = new PIXI.Graphics();
    mask.beginFill(0xffffff);
    mask.drawRect(x, yOffset - 5, w - (needsScroll ? 30 : 0), actualVisibleHeight + 10);
    mask.endFill();
    rowsContainer.mask = mask;
    container.addChild(mask);
    // Get relationship data
    const csRelationships = this.csInfluence?.[this.turn] || [];
    const csRelLevel = this.csRelationships?.[this.turn] || [];

    // Create all rows (they'll be masked if outside visible area)
    cityStates.forEach((cs, csIdx) => {
      const rowY = yOffset + (csIdx * rowHeight) - this.csScrollY;
      
      // Alternating row background
      if (csIdx % 2 === 0) {
        const rowBg = new PIXI.Graphics();
        rowBg.beginFill(COLORS.stoneBg, 0.1);
        rowBg.drawRect(x + 15, rowY - 5, w - 50, 25);
        rowBg.endFill();
        rowsContainer.addChild(rowBg);
      }
      
      // City-state name
      const nameText = new PIXI.Text(cs.name, statusStyle);
      nameText.position.set(x + 20, rowY);
      rowsContainer.addChild(nameText);
      
      // Type with color coding
      const typeColors = {
        'Cultural': 0x9b59b6,
        'Agricultural': 0x27ae60,
        'Mercantile': 0xf39c12,
        'Religious': 0xffffff,
        'Scientific': 0x3498db,
        'Militaristic': 0xe74c3c,
        'Unknown': 0x7f8c8d
      };
      
      const typeText = new PIXI.Text(cs.type, {
        ...statusStyle,
        fill: typeColors[cs.type] || COLORS.goldDark
      });
      typeText.position.set(x + 180, rowY);
      rowsContainer.addChild(typeText);
      
      // Get relationships for this city-state
      const csRelations = csRelationships[csIdx] || [];
      const csRelLevelInner = csRelLevel[csIdx] || [];
      
      // Display relationship values for each player
      const playerXPositions = [330, 460, 590, 720, 840, 970];
      
      for (let playerIdx = 0; playerIdx < 6; playerIdx++) {
        const relationValue = csRelations[playerIdx] || 0;
        const relationLevel = csRelLevelInner[playerIdx] || 0;
        const xPos = x + playerXPositions[playerIdx] + 33;
        
        // Progress bar settings
        const barWidth = 80;
        const barHeight = 18;
        const maxValue = 60; // Max value for progress bar
        
        // Draw progress bar background
        const barBg = new PIXI.Graphics();
        barBg.beginFill(COLORS.divider, 0.2);
        barBg.drawRect(xPos - barWidth/2, rowY - 2, barWidth, barHeight);
        barBg.endFill();
        
        // Determine color based on relationship value
        let barColor;
        if (relationLevel === 2) {
          barColor = 0x3498db; // Blue for Ally
        } else if (relationLevel === 1) {
          barColor = COLORS.success; // Green for Friend
        } else {
          barColor = COLORS.goldDark; // Gold/neutral
        } 
        // Draw progress bar fill
        if (relationValue !== 0) {
          // Calculate fill width (capped at maxValue of 60)
          const clampedValue = Math.min(Math.abs(relationValue), maxValue);
          const fillWidth = (clampedValue / maxValue) * barWidth;
          
          barBg.beginFill(barColor, 0.6);
          if (relationValue > 0) {
            // Positive values - fill from left
            barBg.drawRect(xPos - barWidth/2, rowY - 2, fillWidth, barHeight);
          } else {
            // Negative values - fill entire bar in red (since negative means hostile)
            const negFillWidth = Math.min(barWidth, (Math.abs(relationValue) / 30) * barWidth);
            barBg.drawRect(xPos - barWidth/2, rowY - 2, negFillWidth, barHeight);
          }
          barBg.endFill();
        }
        
        // Draw border around progress bar
        barBg.lineStyle(1, COLORS.goldDark, 0.3);
        barBg.drawRect(xPos - barWidth/2, rowY - 2, barWidth, barHeight);
        barBg.stroke();
        
        rowsContainer.addChild(barBg);
        
        // Display the relationship value on top of the bar
        const relationText = new PIXI.Text(Math.floor(relationValue).toString(), {
          fontFamily: 'Crimson Text',
          fontSize: 16,
          fontWeight: relationValue >= 60 ? 'bold' : '600',
          fill: relationValue >= 30 ? 0xffffff : 0x0f0f1e,
          dropShadow: relationValue >= 30,
          dropShadowColor: 0x000000,
          dropShadowDistance: 1,
          dropShadowAlpha: 0.8
        });
        relationText.anchor.set(0.5, 0.5);
        relationText.position.set(xPos, rowY + 7);
        rowsContainer.addChild(relationText);
      }
    });
    
    container.addChild(rowsContainer);
    
    // Add scroll bar if needed
    if (needsScroll) {
      const scrollBarX = x + w - 25;
      const scrollBarHeight = actualVisibleHeight;
      const scrollThumbHeight = Math.max(30, (actualVisibleHeight / totalContentHeight) * scrollBarHeight);
      
      // Scroll track
      const scrollTrack = new PIXI.Graphics();
      scrollTrack.beginFill(COLORS.stoneBg, 0.3);
      scrollTrack.drawRect(scrollBarX, yOffset, 10, scrollBarHeight);
      scrollTrack.endFill();
      scrollTrack.lineStyle(1, COLORS.goldDark, 0.3);
      scrollTrack.drawRect(scrollBarX, yOffset, 10, scrollBarHeight);
      scrollTrack.stroke();
      container.addChild(scrollTrack);
      
      // Scroll thumb
      const thumbY = yOffset + (this.csScrollY / maxScroll) * (scrollBarHeight - scrollThumbHeight);
      const scrollThumb = new PIXI.Graphics();
      scrollThumb.beginFill(COLORS.goldDark, 0.6);
      scrollThumb.drawRect(scrollBarX, thumbY, 10, scrollThumbHeight);
      scrollThumb.endFill();
      scrollThumb.lineStyle(1, COLORS.goldBright, 0.5);
      scrollThumb.drawRect(scrollBarX, thumbY, 10, scrollThumbHeight);
      scrollThumb.stroke();
      container.addChild(scrollThumb);
      
      // Make scrollable area interactive for mouse wheel
      const scrollArea = new PIXI.Graphics();
      scrollArea.beginFill(0x000000, 0.01);
      scrollArea.drawRect(x, yOffset, w - 30, actualVisibleHeight);
      scrollArea.endFill();
      scrollArea.eventMode = 'static';
      container.addChild(scrollArea);
      
      // Simple mouse wheel handler
      scrollArea.on('wheel', (e) => {
        //e.preventDefault();
        const delta = e.deltaY > 0 ? 30 : -30;
        this.csScrollY = Math.max(0, Math.min(maxScroll, this.csScrollY + delta));
        this.redrawContent(); // Now it's okay since we're not doing it continuously
      });
    }
  }
  
  renderActiveQuests(container, x, y, w, h) {
    // ---------- Header left ----------
    const title = new PIXI.Text('ACTIVE CITY-STATE QUESTS', requirementStyle);
    title.roundPixels = true;
    title.position.set(Math.round(x), Math.round(y));
    container.addChild(title);

    // ---------- Header right ----------
    const computedTurnsRemaining = 30 - ((this.turn - 1) % 30);
    const turnsRemaining = new PIXI.Text(`TURNS REMAINING: ${computedTurnsRemaining}`, requirementStyle);
    turnsRemaining.roundPixels = true;
    turnsRemaining.anchor.set(1, 0); // right-align
    const rightPad = 20;
    turnsRemaining.position.set(Math.round(x + w - rightPad), Math.round(y));
    container.addChild(turnsRemaining);

    // ---------- Data ----------
    const csNames = constants.csNames || [];
    const questsForTurn = (this.csQuest && this.csQuest[this.turn]) || [];
    const QUEST_TYPES = ['None', 'Culture', 'Faith', 'Techs', 'Trade Routes', 'Religion', 'Wonders', 'Resources'];

    // Trackers in the order requested for qType 1..7
    const trackers = [
      this.csCultureTracker,
      this.csFaithTracker,
      this.csTechTracker,
      this.csTradeTracker,
      this.csReligionTracker,
      this.csWonderTracker,
      this.csResourceTracker
    ];

    const haveMetThisTurn = this.haveMet[this.turn];

    const count = Math.min(12, csNames.length || 12); // default to 12 if names missing

    // ---------- Layout ----------
    const listTop = Math.round(y + 35);
    const leftPad = 20;
    const innerWidth = w - leftPad - rightPad;
    const cardHeight = 80;
    const cardGap = 10;
    const rowHeight = cardHeight + cardGap;

    // Visible area & scrolling params
    const visibleHeight = Math.max(0, h - (listTop - y) - 20);
    const totalContentHeight = count * rowHeight;
    const needsScroll = totalContentHeight > visibleHeight;
    const scrollBarWidth = needsScroll ? 12 : 0;

    // Mask width leaves space for scrollbar if present
    const maskWidth = innerWidth - scrollBarWidth;

    // Maintain scroll state across redraws
    if (this._questsScrollY == null) this._questsScrollY = 0;

    // ---------- Rows container (created once per render; scrolled by y-only updates) ----------
    const rowsContainer = new PIXI.Container();
    rowsContainer.position.set(Math.round(x + leftPad), Math.round(listTop));
    rowsContainer.roundPixels = true;
    container.addChild(rowsContainer);

    // Mask to clip rows
    const maskG = new PIXI.Graphics();
    maskG.beginFill(0xffffff);
    maskG.drawRect(Math.round(x + leftPad), Math.round(listTop) - 1, Math.round(maskWidth), Math.round(visibleHeight) + 2);
    maskG.endFill();
    container.addChild(maskG);
    rowsContainer.mask = maskG;

    // ---------- Build cards (once) ----------
    for (let i = 0; i < count; i++) {
      const rowY = i * rowHeight;

      // Card background
      const card = new PIXI.Graphics();
      card.beginFill(COLORS.stoneBg, 0.20);
      card.drawRoundedRect(0, rowY, maskWidth, cardHeight, 0);
      card.endFill();
      card.lineStyle(1, COLORS.goldDark, 0.5);
      card.drawRoundedRect(0, rowY, maskWidth, cardHeight, 0);
      card.stroke();
      rowsContainer.addChild(card);

      // City-state name
      const name = new PIXI.Text(csNames[i] || `City-State ${i + 1}`, {
        ...requirementStyle,
        fontSize: 16
      });
      name.roundPixels = true;
      name.position.set(14, rowY + 10);
      rowsContainer.addChild(name);

      // Quest type text (pulled from this.csQuest[this.turn])
      const qIdx = questsForTurn[i] ?? 0;                 // 0..7
      const qType = QUEST_TYPES[qIdx] || 'None';
      const quest = new PIXI.Text(qType, {
        ...statusStyle,
        fontSize: 14,
        fontWeight: 600,
        fill: qIdx === 0 ? COLORS.goldDark : COLORS.goldBright
      });
      quest.roundPixels = true;
      quest.position.set(14, rowY + 36);
      rowsContainer.addChild(quest);
      // ---------- Per-player tiny table (only if qType != None) ----------
          // ---------- Per-player tiny table (only if qType != None) ----------
      if (qIdx >= 1 && qIdx <= 7) {
        const tracker = trackers[qIdx - 1];

        if (qIdx === 3) {
          // For techs, subtract sum of all techs researched from currently stored values
          const techSums = this.sumRowsFast(this.playerTechs[this.turn]);
          var values = this.subNew(techSums, tracker[this.turn]);
        } else if (qIdx === 4) {
          // (6, 12)
          var values = tracker[this.turn].map(row => row[i]);
        } else {
          var values = tracker[this.turn];
        }

        // Wider table on the right for more breathing room
        const tableWidth = Math.min(420, Math.floor(maskWidth * 0.72));
        const tableX = Math.round(maskWidth - tableWidth - 12); // 12px inset from right
        const padX = 10;
        const innerW = tableWidth - padX * 2;
        const step = innerW / 6;

        // Header band to make top row stand out
        const headerBand = new PIXI.Graphics();
        headerBand.beginFill(COLORS.stoneBg, 0.28);
        headerBand.drawRoundedRect(tableX, rowY + 8, tableWidth, 20, 4);
        headerBand.endFill();
        headerBand.lineStyle(1, COLORS.goldDark, 0.35);
        headerBand.drawRoundedRect(tableX, rowY + 8, tableWidth, 20, 4);
        headerBand.stroke();
        rowsContainer.addChild(headerBand);

        // Column centers (with left/right padding)
        const centers = Array.from({ length: 6 }, (_, k) =>
          Math.round(tableX + padX + step * (k + 0.5))
        );

        // Largest value for highlight
        // Here we do not highlight values for (cs, player) pairs that 
        // have not met yet. For this, we just mask out (*0)
        let maxVal = Number.NEGATIVE_INFINITY;
        for (let p = 0; p < 6; p++) {
          const vp = (values[p] * haveMetThisTurn[p][i + 6])?? Number.NEGATIVE_INFINITY;
          if (vp > maxVal) maxVal = vp;
        }

        // Header row: "Player N"
        for (let p = 0; p < 6; p++) {
          const hdr = new PIXI.Text(`Player ${p + 1}`, statusStyle);
          hdr.roundPixels = true;
          hdr.anchor.set(0.5, 0.5);
          hdr.position.set(centers[p], Math.round(rowY + 18)); // in header band
          rowsContainer.addChild(hdr);
        }

        // Values row (below header), highlight the largest
        for (let p = 0; p < 6; p++) {
          const v = values[p] ?? 0;
          const isLeader = (v === maxVal && maxVal !== Number.NEGATIVE_INFINITY && haveMetThisTurn[p][i + 6] === 1);

          const valText = new PIXI.Text(String(Math.round(v)), {
            ...statusStyle,
            fontSize: 14,
            fontWeight: isLeader ? 800 : 600,
            fill: isLeader ? COLORS.goldBright : COLORS.goldDark
          });
          valText.roundPixels = true;
          valText.anchor.set(0.5, 0.5);
          valText.position.set(centers[p], Math.round(rowY + 44));
          rowsContainer.addChild(valText);

          if (isLeader) {
            // subtle underline to reinforce the highlight
            const underline = new PIXI.Graphics();
            underline.lineStyle(1, COLORS.goldBright, 0.8);
            underline.moveTo(centers[p] - Math.floor(step * 0.35), Math.round(rowY + 52));
            underline.lineTo(centers[p] + Math.floor(step * 0.35), Math.round(rowY + 52));
            underline.stroke();
            rowsContainer.addChild(underline);
          }
        }
      }
  }

  // ---------- Scrollbar & wheel handling (position updates only) ----------
  if (needsScroll) {
    const trackX = Math.round(x + leftPad + maskWidth + 6);
    const trackY = Math.round(listTop);
    const trackH = Math.round(visibleHeight);

    // Track
    const track = new PIXI.Graphics();
    track.beginFill(COLORS.stoneBg, 0.3);
    track.drawRect(trackX, trackY, scrollBarWidth, trackH);
    track.endFill();
    track.lineStyle(1, COLORS.goldDark, 0.3);
    track.drawRect(trackX, trackY, scrollBarWidth, trackH);
    track.stroke();
    container.addChild(track);

    // Thumb
    const thumbMin = 28;
    const thumbH = Math.max(thumbMin, (visibleHeight / totalContentHeight) * trackH);
    const thumb = new PIXI.Graphics();
    container.addChild(thumb);

    const updatePositions = () => {
      // Clamp scroll
      const maxScroll = totalContentHeight - visibleHeight;
      this._questsScrollY = Math.max(0, Math.min(maxScroll, this._questsScrollY));

      // Move rows by y only (no re-creation)
      rowsContainer.y = Math.round(listTop - this._questsScrollY);

      // Thumb position
      const free = trackH - thumbH;
      const ratio = (maxScroll > 0) ? (this._questsScrollY / maxScroll) : 0;
      const thumbY = Math.round(trackY + ratio * free);

      thumb.clear();
      thumb.beginFill(COLORS.goldDark, 0.6);
      thumb.drawRect(trackX, thumbY, scrollBarWidth, thumbH);
      thumb.endFill();
      thumb.lineStyle(1, COLORS.goldBright, 0.5);
      thumb.drawRect(trackX, thumbY, scrollBarWidth, thumbH);
      thumb.stroke();
    };

    // Invisible scrollable hit area covering the masked list
    const scrollArea = new PIXI.Graphics();
    scrollArea.beginFill(0x000000, 0.001);
    scrollArea.drawRect(Math.round(x + leftPad), Math.round(listTop), Math.round(maskWidth), Math.round(visibleHeight));
    scrollArea.endFill();
    scrollArea.eventMode = 'static';
    container.addChild(scrollArea);

    // Wheel handler: update Y only
    scrollArea.on('wheel', (e) => {
      //e.preventDefault();
      const delta = e.deltaY > 0 ? 40 : -40;
      this._questsScrollY += delta;
      updatePositions();
    });

    // Click track to jump
    track.eventMode = 'static';
    track.on('pointerdown', (e) => {
      const localY = e.global.y - trackY;
      const maxScroll = totalContentHeight - visibleHeight;
      const free = trackH - thumbH;
      const ratio = free > 0 ? (localY - thumbH / 2) / free : 0;
      this._questsScrollY = Math.round(Math.max(0, Math.min(1, ratio)) * maxScroll);
      updatePositions();
    });

    updatePositions(); // initial layout with current scroll
  } else {
    // No scrolling needed — ensure container is aligned
    rowsContainer.y = Math.round(listTop);
  }
  }

  renderScientificVictory(x, y, w, h) {
    const container = new PIXI.Container();
    
    const desc = new PIXI.Text(
      'Achieve victory through scientific advancement and space exploration.',
      descStyle
    );
    desc.style.wordWrapWidth = w;
    desc.position.set(x, y);
    container.addChild(desc);
    
    // Requirements
    const reqTitle = new PIXI.Text('REQUIREMENTS', requirementStyle);
    reqTitle.position.set(x, y + 40);
    container.addChild(reqTitle);
    
    const requirements = [
      '• Research all required technologies.',
      '• Build and launch all spaceship parts.'
    ];
    
    let yOffset = y + 70;
    requirements.forEach(req => {
      const reqText = new PIXI.Text(req, statusStyle);
      reqText.position.set(x + 20, yOffset);
      container.addChild(reqText);
      yOffset += 25;
    });
    
    // Divider
    const divider = new PIXI.Graphics();
    divider.lineStyle(1, COLORS.goldDark, 0.5);
    divider.moveTo(x, yOffset + 15);
    divider.lineTo(x + w, yOffset + 15);
    divider.stroke();
    container.addChild(divider);
    
    // ---------- Data ----------
  //const turn = this.turn ?? 0;
  //const pb = (this.playerBuildings && this.playerBuildings[turn]) || []; // (6, max_num_cities, num_buildings)
  const pb = this.playerBuildings[this.turn];
  const rows = [
    { name: 'Apollo Program', idx: 139 },
    { name: 'Booster 1',      idx: 140 },
    { name: 'Booster 2',      idx: 141 },
    { name: 'Booster 3',      idx: 142 },
    { name: 'Engine',         idx: 143 },
    { name: 'Cockpit',        idx: 144 },
    { name: 'Stasis Chamber', idx: 145 },
  ];
  const numPlayers = 6;

  // Compute completion matrix: rows.length x numPlayers -> boolean
  const complete = rows.map(({ idx }) => {
    const row = new Array(numPlayers).fill(false);
    for (let p = 0; p < numPlayers; p++) {
      const cities = pb[p] || [];
      let hasIt = false;
      for (let c = 0; c < cities.length && !hasIt; c++) {
        const slot = cities[c] || [];
        hasIt = (slot[idx] | 0) === 1;
      }
      row[p] = hasIt;
    }
    return row;
  });

  // ---------- Geometry ----------
  const top = Math.round(y + 4 + yOffset / 2.5);
  const leftPad = 24, rightPad = 24;
  const innerW = Math.max(0, w - leftPad - rightPad);

  const headerH = 26;
  const rowH = 32;
  const nameColW = 200;
  const colW = Math.floor((innerW - nameColW) / numPlayers);
  const tableW = nameColW + colW * numPlayers;
  const tableH = headerH + rowH * rows.length;

  const tableX = Math.round(x + leftPad + Math.floor((innerW - tableW) / 2));
  const tableY = top;

  // ---------- Background & frame ----------
  const box = new PIXI.Graphics();
  box.beginFill(COLORS.stoneBg, 0.20);
  box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
  box.endFill();
  box.lineStyle(1, COLORS.goldDark, 0.5);
  box.drawRoundedRect(tableX, tableY, tableW, tableH, 0);
  box.stroke();
  container.addChild(box);

  // ---------- Column headers ----------
  for (let j = 0; j < numPlayers; j++) {
    const tx = new PIXI.Text(`PLAYER ${j + 1}`, statusStyle);
    tx.roundPixels = true;
    tx.anchor.set(0.5, 0.5);
    const cx = Math.round(tableX + nameColW + j * colW + colW / 2);
    const cy = Math.round(tableY + headerH / 2);
    tx.position.set(cx, cy);
    container.addChild(tx);

    // Vertical dividers between player columns
    if (j < numPlayers - 1) {
      const vd = new PIXI.Graphics();
      vd.lineStyle(1, COLORS.divider, 0.35);
      vd.moveTo(Math.round(tableX + nameColW + (j + 1) * colW), Math.round(tableY + 4));
      vd.lineTo(Math.round(tableX + nameColW + (j + 1) * colW), Math.round(tableY + tableH - 4));
      vd.stroke();
      container.addChild(vd);
    }
  }

  // Header underline
  const headerLine = new PIXI.Graphics();
  headerLine.lineStyle(1, COLORS.goldDark, 0.35);
  headerLine.moveTo(tableX, Math.round(tableY + headerH));
  headerLine.lineTo(tableX + tableW, Math.round(tableY + headerH));
  headerLine.stroke();
  container.addChild(headerLine);

  // ---------- Rows ----------
  for (let i = 0; i < rows.length; i++) {
    const rowY = Math.round(tableY + headerH + i * rowH);

    // Row label
    const label = new PIXI.Text(rows[i].name, statusStyle);
    label.roundPixels = true;
    label.anchor.set(0, 0.5);
    label.position.set(Math.round(tableX + 10), Math.round(rowY + rowH / 2));
    container.addChild(label);

    // Zebra background under full row
    if (i % 2 === 0) {
      const zebra = new PIXI.Graphics();
      zebra.beginFill(COLORS.stoneBg, 0.08);
      zebra.drawRect(Math.round(tableX + nameColW), rowY, Math.round(colW * numPlayers), rowH);
      zebra.endFill();
      container.addChild(zebra);
    }

    // Cells
    for (let j = 0; j < numPlayers; j++) {
      const cellX = Math.round(tableX + nameColW + j * colW);
      const cellY = rowY;

      const ok = complete[i][j];

      const cell = new PIXI.Graphics();
      if (ok) {
        cell.beginFill(COLORS.success, 0.40);
        cell.lineStyle(2, COLORS.success);
      } else {
        cell.beginFill(COLORS.failed, 0.20);
        cell.lineStyle(1, COLORS.failed, 0.65);
      }
      const pad = 4;
      cell.drawRect(
        cellX + pad,
        cellY + pad,
        colW - pad * 2,
        rowH - pad * 2
      );
      cell.endFill();
      cell.stroke();
      container.addChild(cell);

      // Icon: check for yes, X for no (styled like Domination grid)
      const icon = new PIXI.Graphics();
      icon.lineStyle(2, ok ? COLORS.success : COLORS.failed);

      const ix0 = cellX + colW * 0.30, iy0 = cellY + rowH * 0.50;
      const ix1 = cellX + colW * 0.45, iy1 = cellY + rowH * 0.65;
      const ix2 = cellX + colW * 0.70, iy2 = cellY + rowH * 0.35;

      if (ok) {
        // checkmark
        icon.moveTo(ix0, iy0);
        icon.lineTo(ix1, iy1);
        icon.lineTo(ix2, iy2);
      } else {
        // X
        const xa = cellX + colW * 0.30, ya = cellY + rowH * 0.30;
        const xb = cellX + colW * 0.70, yb = cellY + rowH * 0.70;
        const xc = cellX + colW * 0.70, yc = cellY + rowH * 0.30;
        const xd = cellX + colW * 0.30, yd = cellY + rowH * 0.70;
        icon.moveTo(xa, ya);
        icon.lineTo(xb, yb);
        icon.moveTo(xc, yc);
        icon.lineTo(xd, yd);
      }
      icon.stroke();
      container.addChild(icon);
    }
  }
  this.lContent.addChild(container);
  }

  renderDominationVictory(x, y, w, h) {
    const container = new PIXI.Container();
    
    const desc = new PIXI.Text(
      'Win through military conquest and territorial control.',
      descStyle
    );
    desc.style.wordWrapWidth = w;
    desc.position.set(x, y);
    container.addChild(desc);
    
    // Requirements
    const reqTitle = new PIXI.Text('REQUIREMENTS', requirementStyle);
    reqTitle.position.set(x, y + 40);
    container.addChild(reqTitle);
    
    const requirements = [
      '• Sack every capital city in the game.',
    ];
    
    let yOffset = y + 70;
    requirements.forEach(req => {
      const reqText = new PIXI.Text(req, statusStyle);
      reqText.position.set(x + 20, yOffset);
      container.addChild(reqText);
      yOffset += 25;
    });
    
    // Divider
    const divider = new PIXI.Graphics();
    divider.lineStyle(1, COLORS.goldDark, 0.5);
    divider.moveTo(x, yOffset + 15);
    divider.lineTo(x + w, yOffset + 15);
    divider.stroke();
    container.addChild(divider);
    
    yOffset += 35;
    
    // ========== SUB-TABS SYSTEM ==========
    // Initialize domination sub-tab state if not exists
    if (this.dominationSubTab === undefined) {
      this.dominationSubTab = 0; // 0 = Overview, 1 = Relationships
    }
    
    const subTabLabels = ['OVERVIEW', 'RELATIONSHIPS', 'UNITS'];
    const subTabWidth = (w - 40) / subTabLabels.length;
    const subTabX = x + 20;
    const subTabY = yOffset;
    const subTabHeight = 35;
    
    // Sub-tab background
    const subTabBg = new PIXI.Graphics();
    subTabBg.beginFill(COLORS.stoneBg, 0.2);
    subTabBg.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBg.endFill();
    container.addChild(subTabBg);
    
    // Create sub-tabs
    subTabLabels.forEach((label, idx) => {
      const tabG = new PIXI.Graphics();
      const tabX = subTabX + idx * subTabWidth;
      
      // Draw sub-tab
      if (idx === this.dominationSubTab) {
        // Active sub-tab
        tabG.beginFill(COLORS.goldDark, 0.3);
        tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
        tabG.endFill();
        
        // Gold accent line on top
        tabG.beginFill(COLORS.goldBright);
        tabG.drawRect(tabX, subTabY, subTabWidth, 2);
        tabG.endFill();
        
        // Side borders
        tabG.lineStyle(1, COLORS.goldDark);
        tabG.moveTo(tabX, subTabY);
        tabG.lineTo(tabX, subTabY + subTabHeight);
        tabG.moveTo(tabX + subTabWidth, subTabY);
        tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight);
        tabG.stroke();
      } else {
        // Inactive sub-tab - just divider
        if (idx < subTabLabels.length - 1) {
          tabG.lineStyle(1, COLORS.divider, 0.5);
          tabG.moveTo(tabX + subTabWidth, subTabY + 5);
          tabG.lineTo(tabX + subTabWidth, subTabY + subTabHeight - 5);
          tabG.stroke();
        }
      }
      
      // Make interactive
      tabG.eventMode = 'static';
      tabG.cursor = 'pointer';
      tabG.beginFill(0x000000, 0.01);
      tabG.drawRect(tabX, subTabY, subTabWidth, subTabHeight);
      tabG.endFill();
      
      // Sub-tab label
      const subTabStyle = idx === this.dominationSubTab ? 
        { ...requirementStyle, fill: COLORS.goldBright } : 
        { ...statusStyle, fill: COLORS.goldDark };
      
      const subTabText = new PIXI.Text(label, subTabStyle);
      subTabText.roundPixels = true;
      subTabText.anchor.set(0.5, 0.5);
      const cx = Math.round(tabX + subTabWidth / 2);
      const cy = Math.round(subTabY + subTabHeight / 2);
      const oddW = (Math.round(subTabText.width) % 2) === 1;
      const oddH = (Math.round(subTabText.height) % 2) === 1;
      subTabText.position.set(cx + (oddW ? 0.5 : 0), cy + (oddH ? 0.5 : 0));
      
      // Hover effect
      tabG.on('pointerover', () => {
        if (idx !== this.dominationSubTab) {
          subTabText.style.fill = COLORS.goldBright;
          subTabText.alpha = 0.7;
        }
      });
      
      tabG.on('pointerout', () => {
        if (idx !== this.dominationSubTab) {
          subTabText.style.fill = COLORS.goldDark;
          subTabText.alpha = 1;
        }
      });
      
      // Click handler
      tabG.on('pointerdown', () => {
        this.dominationSubTab = idx;
        this.redrawContent(); // Redraw to show new sub-tab content
      });
      
      container.addChild(tabG);
      container.addChild(subTabText);
    });
    
    // Bottom border for sub-tabs
    const subTabBorder = new PIXI.Graphics();
    subTabBorder.lineStyle(1, COLORS.goldDark);
    subTabBorder.drawRect(subTabX, subTabY, w - 40, subTabHeight);
    subTabBorder.stroke();
    container.addChild(subTabBorder);
    
    // ========== SUB-TAB CONTENT ==========
    const contentY = subTabY + subTabHeight + 20;
    const contentH = h - (contentY - y) - 20;
    
    switch(this.dominationSubTab) {
      case 0: // Overview
        this.renderDominationOverview(container, x, contentY, w, contentH);
        break;
      case 1: // Relationships
        this.renderDominationRelationships(container, x, contentY, w, contentH);
        break;
      case 2: // Units
        this.renderDominationUnits(container, x, contentY, w, contentH);
        break;
    }
    
    this.lContent.addChild(container);
  }

  renderDominationOverview(container, x, y, w, h) {
    let yOffset = y + 1;
    let xOffset = x + 20;
    
    // Count how many capitals controlled (sum of all green cells)
    let controlledCount = 0;
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        if (i !== j && this.hasSacked[this.turn][i][j] === 1) {
          controlledCount++;
        }
      }
    }
    
    // 6x6 Grid
    const gridStartX = x + 60 + xOffset;
    const gridStartY = yOffset + 30;
    const cellSize = Math.min((w - 120) / 6, 50); // Responsive sizing
    const cellPadding = 2;
    
    // Column headers
    for (let col = 0; col < 6; col++) {
      const colLabel = new PIXI.Text(`P${col + 1}`, {
        fontFamily: 'Cinzel',
        fontSize: 14,
        fontWeight: 600,
        fill: COLORS.goldDark
      });
      colLabel.anchor.set(0.5, 1);
      colLabel.position.set(
        gridStartX + col * cellSize + cellSize/2,
        gridStartY - 5
      );
      container.addChild(colLabel);
    }
    
    // Row headers and grid cells
    for (let row = 0; row < 6; row++) {
      // Row label
      const rowLabel = new PIXI.Text(`Player ${row + 1}`, {
        fontFamily: 'Cinzel',
        fontSize: 14,
        fontWeight: 600,
        fill: COLORS.goldDark
      });
      rowLabel.anchor.set(1, 0.5);
      rowLabel.position.set(
        gridStartX - 10,
        gridStartY + row * cellSize + cellSize/2
      );
      container.addChild(rowLabel);
      
      // Grid cells
      for (let col = 0; col < 6; col++) {
        const cellX = gridStartX + col * cellSize;
        const cellY = gridStartY + row * cellSize;
        
        const cell = new PIXI.Graphics();
        
        // Determine cell appearance
        if (row === col) {
          // Diagonal - player's own capital
          cell.beginFill(COLORS.stoneBg, 0.3);
          cell.lineStyle(1, COLORS.goldDark, 0.5);
        } else if (this.hasSacked[this.turn][row][col] === 1) {
          // Captured
          cell.beginFill(COLORS.success, 0.4);
          cell.lineStyle(2, COLORS.success);
        } else {
          // Not captured
          cell.beginFill(COLORS.failed, 0.2);
          cell.lineStyle(1, COLORS.failed, 0.6);
        }
        
        cell.drawRect(
          cellX + cellPadding,
          cellY + cellPadding,
          cellSize - cellPadding * 2,
          cellSize - cellPadding * 2
        );
        cell.endFill();
        cell.stroke();
        container.addChild(cell);
        
        // Add icon in cell
        if (row !== col) {
          const icon = new PIXI.Graphics();
          icon.lineStyle(2, this.hasSacked[this.turn][row][col] === 1 ? COLORS.success : COLORS.failed);
          
          if (this.hasSacked[this.turn][row][col] === 1) {
            // Checkmark for captured
            icon.moveTo(cellX + cellSize * 0.3, cellY + cellSize * 0.5);
            icon.lineTo(cellX + cellSize * 0.45, cellY + cellSize * 0.65);
            icon.lineTo(cellX + cellSize * 0.7, cellY + cellSize * 0.35);
          } else {
            // X for not captured
            icon.moveTo(cellX + cellSize * 0.3, cellY + cellSize * 0.3);
            icon.lineTo(cellX + cellSize * 0.7, cellY + cellSize * 0.7);
            icon.moveTo(cellX + cellSize * 0.7, cellY + cellSize * 0.3);
            icon.lineTo(cellX + cellSize * 0.3, cellY + cellSize * 0.7);
          }
          icon.stroke();
          container.addChild(icon);
        } else {
          // Add dash or "N/A" text for diagonal
          const naText = new PIXI.Text('—', {
            fontFamily: 'Cinzel',
            fontSize: 18,
            fill: COLORS.goldDark,
            fontWeight: 600
          });
          naText.anchor.set(0.5, 0.5);
          naText.position.set(
            cellX + cellSize/2,
            cellY + cellSize/2
          );
          container.addChild(naText);
        }
      }
    }
  }

  renderDominationRelationships(container, x, y, w, h) {
    // Calculate center and radius for hexagon
    const centerX = x + w / 2;
    const centerY = y + h / 2 + 10; // Slight offset down to account for title
    const radius = Math.min(w, h - 60) * 0.5; // Leave room for labels
    
    // Calculate hexagon vertices (flat-topped hexagon)
    const players = [];
    for (let i = 0; i < 6; i++) {
      // Start from top and go clockwise
      const angle = (Math.PI / 3) * i - Math.PI / 2; // Start from top
      const px = centerX + radius * Math.cos(angle);
      const py = centerY + radius * Math.sin(angle);
      players.push({ x: px, y: py, index: i });
    }
    
    // Draw connection lines between all players (placeholder)
    const linesGraphics = new PIXI.Graphics();

    const haveMetThisTurn = this.haveMet[this.turn];
    const tradeLedgerThisTurn = this.tradeLedger[this.turn];
    const atWarThisTurn = this.atWar[this.turn];
    
    // Draw all possible connections (15 total for 6 players)
    for (let i = 0; i < 6; i++) {
      for (let j = i + 1; j < 6; j++) {
        // Placeholder: alternate between different relationship types
        const haveMetInner = haveMetThisTurn[i][j];
        const areTrading = (tradeLedgerThisTurn[i][j].flat().reduce((acc, val) => acc + val, 0)) > 0;
        const areAtWar = atWarThisTurn[i][j];
         
        let lineColor, lineAlpha, lineWidth;

        if ((haveMetInner === 1) && !areTrading) {
          if (areAtWar) {
            lineColor = COLORS.failed;
          } else {
            lineColor = COLORS.goldDark;
          }
          lineAlpha = 1;
          lineWidth = 4;
          linesGraphics.lineStyle(lineWidth, lineColor, lineAlpha);
          linesGraphics.moveTo(players[i].x, players[i].y);
          linesGraphics.lineTo(players[j].x, players[j].y);
          linesGraphics.stroke();
        } else if (areTrading) {
          lineColor = COLORS.success;
          lineAlpha = 1;
          lineWidth = 4;
          linesGraphics.lineStyle(lineWidth, lineColor, lineAlpha);
          linesGraphics.moveTo(players[i].x, players[i].y);
          linesGraphics.lineTo(players[j].x, players[j].y);
          linesGraphics.stroke();
        }
      }
    }
    container.addChild(linesGraphics);
    
    // Draw player circles
    players.forEach((player, idx) => {
      // Outer ring (for decoration)
      const outerRing = new PIXI.Graphics();
      outerRing.lineStyle(3, COLORS.goldDark, 0.5);
      outerRing.drawCircle(player.x, player.y, 45);
      outerRing.stroke();
      container.addChild(outerRing);
      
      // Main circle background
      const circle = new PIXI.Graphics();
      circle.beginFill(COLORS.stoneBg, 0.95);
      circle.drawCircle(player.x, player.y, 40);
      circle.endFill();
      
      // Border (different color for player 0 - the human player)
      //const borderColor = idx === 0 ? COLORS.goldBright : COLORS.goldDark;
      const borderColor = COLORS.goldDark;
      circle.lineStyle(2, borderColor);
      circle.drawCircle(player.x, player.y, 40);
      circle.stroke();
      container.addChild(circle);
      
      // Player number in circle
      const playerNum = new PIXI.Text(`${idx + 1}`, {
        fontFamily: 'Cinzel',
        fontSize: 24,
        fontWeight: 600,
        fill: 0xffffff
      });
      playerNum.anchor.set(0.5, 0.5);
      playerNum.position.set(player.x, player.y);
      container.addChild(playerNum);
      
    });
    
    // Legend at bottom
    const legendY = y + h - 140;
    const legendYOffset = 35;
    const legendItems = [
      { color: COLORS.goldDark, label: 'Have met', x: x + w * 0.75, y: legendY - legendYOffset },
      { color: COLORS.success, label: 'Friendly', x: x + w * 0.75, y: legendY },
      { color: COLORS.failed, label: 'At war', x: x + w * 0.75, y: legendY + legendYOffset }
    ];
    
    legendItems.forEach(item => {
      // Color indicator
      const indicator = new PIXI.Graphics();
      indicator.beginFill(item.color);
      indicator.drawRect(item.x - 40, item.y, 40, 3);
      indicator.endFill();
      container.addChild(indicator);
      
      // Label
      const label = new PIXI.Text(item.label.toUpperCase(), {
        ...statusStyle,
        fill: COLORS.goldDark
      });
      label.style.fontSize = 24;
      label.anchor.set(0, 0.5);
      label.position.set(item.x + 15, item.y);
      container.addChild(label);
    });
  }

  renderDominationUnits(container, x, y, w, h) {
    // ---------- State ----------
    if (this.dominationUnitsPlayerIndex == null) this.dominationUnitsPlayerIndex = 0;
    if (!this._unitsScrollYByPlayer) this._unitsScrollYByPlayer = [];

    // Tear down previous instance of this widget only
    if (this._domUnitsRoot) {
      container.removeChild(this._domUnitsRoot);
      this._domUnitsRoot.destroy({ children: true });
    }
    this._domUnitsRoot = new PIXI.Container();
    const root = this._domUnitsRoot;
    container.addChild(root);

    // ---------- Per-player sub-tabs ----------
    const playerLabels = ['PLAYER 1', 'PLAYER 2', 'PLAYER 3', 'PLAYER 4', 'PLAYER 5', 'PLAYER 6'];
    const ptHeight = 32;
    const ptX = x + 20;
    const ptY = y;
    const ptW = w - 40;
    const perTabW = ptW / playerLabels.length;

    const ptBg = new PIXI.Graphics();
    ptBg.beginFill(COLORS.stoneBg, 0.2);
    ptBg.drawRect(ptX, ptY, ptW, ptHeight);
    ptBg.endFill();
    root.addChild(ptBg);

    playerLabels.forEach((label, idx) => {
      const tabG = new PIXI.Graphics();
      const tabX = Math.round(ptX + idx * perTabW);

      if (idx === this.dominationUnitsPlayerIndex) {
        tabG.beginFill(COLORS.goldDark, 0.3);
        tabG.drawRect(tabX, ptY, Math.round(perTabW), ptHeight);
        tabG.endFill();

        tabG.beginFill(COLORS.goldBright);
        tabG.drawRect(tabX, ptY, Math.round(perTabW), 2);
        tabG.endFill();

        tabG.lineStyle(1, COLORS.goldDark);
        tabG.moveTo(tabX, ptY);
        tabG.lineTo(tabX, ptY + ptHeight);
        tabG.moveTo(tabX + perTabW, ptY);
        tabG.lineTo(tabX + perTabW, ptY + ptHeight);
        tabG.stroke();
      } else {
        if (idx < playerLabels.length - 1) {
          tabG.lineStyle(1, COLORS.divider, 0.5);
          tabG.moveTo(tabX + perTabW, ptY + 5);
          tabG.lineTo(tabX + perTabW, ptY + ptHeight - 5);
          tabG.stroke();
        }
      }

      tabG.eventMode = 'static';
      tabG.cursor = 'pointer';
      tabG.beginFill(0x000000, 0.01);
      tabG.drawRect(tabX, ptY, Math.round(perTabW), ptHeight);
      tabG.endFill();

      const txtStyle = idx === this.dominationUnitsPlayerIndex
        ? { ...requirementStyle, fill: COLORS.goldBright, fontSize: 14 }
        : { ...statusStyle,      fill: COLORS.goldDark,   fontSize: 14 };

      const tabText = new PIXI.Text(label, txtStyle);
      tabText.roundPixels = true;
      tabText.anchor.set(0.5, 0.5);
      const cx = Math.round(tabX + perTabW / 2);
      const cy = Math.round(ptY + ptHeight / 2);
      const oddW = (Math.round(tabText.width)  % 2) === 1;
      const oddH = (Math.round(tabText.height) % 2) === 1;
      tabText.position.set(cx + (oddW ? 0.5 : 0), cy + (oddH ? 0.5 : 0));

      tabG.on('pointerover', () => {
        if (idx !== this.dominationUnitsPlayerIndex) {
          tabText.style.fill = COLORS.goldBright;
          tabText.alpha = 0.7;
        }
      });
      tabG.on('pointerout', () => {
        if (idx !== this.dominationUnitsPlayerIndex) {
          tabText.style.fill = COLORS.goldDark;
          tabText.alpha = 1;
        }
      });
      tabG.on('pointerdown', () => {
        this.dominationUnitsPlayerIndex = idx;
        // Rebuild the widget with the newly selected player
        this.renderDominationUnits(container, x, y, w, h);
      });

      root.addChild(tabG, tabText);
    });

    // ---------- Table header ----------
    const tableX = x + 20;
    const tableY = y + ptHeight + 14;
    const tableW = w - 40;

    const headerBand = new PIXI.Graphics();
    headerBand.beginFill(COLORS.stoneBg, 0.28);
    headerBand.drawRoundedRect(Math.round(tableX), Math.round(tableY), Math.round(tableW), 28, 4);
    headerBand.endFill();
    headerBand.lineStyle(1, COLORS.goldDark, 0.35);
    headerBand.drawRoundedRect(Math.round(tableX), Math.round(tableY), Math.round(tableW), 28, 4);
    headerBand.stroke();
    root.addChild(headerBand);

    const cols = [
      { key: 'type', label: 'Unit Type',   width: 220, align: 'left'   },
      { key: 'loc',  label: 'Location',    width: 160, align: 'center' },
      { key: 'hp',   label: 'Health',      width: 140, align: 'right'  },
      { key: 'mult', label: 'Combat Mult', width: 180, align: 'right'  },
    ];

    const colXs = [];
    { let cx = tableX + 12; cols.forEach(c => { colXs.push(cx); cx += c.width; }); }

    cols.forEach((c, i) => {
      const hdr = new PIXI.Text(c.label.toUpperCase(), {
        ...statusStyle,
        fontSize: 12,
        fontWeight: 700,
      });
      hdr.roundPixels = true;

      if (c.align === 'right') {
        hdr.anchor.set(1, 0.5);
        hdr.position.set(Math.round(colXs[i] + c.width - 12), Math.round(tableY + 14));
      } else if (c.align === 'center') {
        hdr.anchor.set(0.5, 0.5);
        hdr.position.set(Math.round(colXs[i] + c.width / 2), Math.round(tableY + 14));
      } else {
        hdr.anchor.set(0, 0.5);
        hdr.position.set(Math.round(colXs[i]), Math.round(tableY + 14));
      }
      root.addChild(hdr);
    });

    // ---------- Rows for selected player ----------
    let rows;
    if (typeof this.getDominationUnitRows === 'function') {
      rows = this.getDominationUnitRows(this.dominationUnitsPlayerIndex) ?? [];
    } else {
      const unitTypes = ['Warrior', 'Archer', 'Spearman', 'Horseman', 'Catapult', 'Scout', 'Swordsman', 'Pikeman'];
      rows = [];
      for (let i = 0; i < 40; i++) {
        const t = unitTypes[i % unitTypes.length];
        const r = (i * 3) % 42;
        const c = (i * 5) % 66;
        const hp = 100 - ((i * 7) % 100);
        const mult = (1 + ((i % 7) * 0.1)).toFixed(2) + 'x';
        const xp = (i * 5) % 75;
        rows.push({ type: t, loc: `(${r}, ${c})`, hp: `${hp}`, mult, xp: `${xp}` });
      }
    }

    // ---------- Scrollable area ----------
    const listTop = tableY + 36;
    const rowH = 28;
    const innerH = h - (listTop - y) - 20;
    const visibleH = Math.max(0, innerH);
    const needsScroll = rows.length * rowH > visibleH;
    const scrollBarW = needsScroll ? 12 : 0;

    const rowsContainer = new PIXI.Container();
    rowsContainer.position.set(Math.round(tableX), Math.round(listTop));
    rowsContainer.roundPixels = true;
    root.addChild(rowsContainer);

    const maskG = new PIXI.Graphics();
    maskG.beginFill(0xffffff);
    maskG.drawRect(Math.round(tableX), Math.round(listTop) - 1, Math.round(tableW - scrollBarW), Math.round(visibleH) + 2);
    maskG.endFill();
    root.addChild(maskG);
    rowsContainer.mask = maskG;

    for (let i = 0; i < rows.length; i++) {
      const yRow = i * rowH;

      if (i % 2 === 0) {
        const bg = new PIXI.Graphics();
        bg.beginFill(COLORS.stoneBg, 0.10);
        bg.drawRect(0, Math.round(yRow), Math.round(tableW - scrollBarW), rowH);
        bg.endFill();
        rowsContainer.addChild(bg);
      }

      cols.forEach((c, ci) => {
        const text = new PIXI.Text(rows[i][c.key], {
          ...statusStyle,
          fontSize: 14,
          fontWeight: 600,
          fill: c.align === 'left' ? COLORS.goldDark : 0xffffff
        });
        text.roundPixels = true;

        if (c.align === 'right') {
          text.anchor.set(1, 0);
          text.position.set(Math.round(colXs[ci] + c.width - 12 - tableX), Math.round(yRow + 6));
        } else if (c.align === 'center') {
          text.anchor.set(0.5, 0);
          text.position.set(Math.round(colXs[ci] - tableX + c.width / 2), Math.round(yRow + 6));
        } else {
          text.anchor.set(0, 0);
          text.position.set(Math.round(colXs[ci] - tableX), Math.round(yRow + 6));
        }
        rowsContainer.addChild(text);
      });
    }

    // ---------- Scrollbar & wheel handling (per-player state) ----------
    const pIdx = this.dominationUnitsPlayerIndex;
    if (needsScroll) {
      if (this._unitsScrollYByPlayer[pIdx] == null) this._unitsScrollYByPlayer[pIdx] = 0;

      const trackX = Math.round(tableX + tableW - scrollBarW);
      const trackY = Math.round(listTop);
      const trackH = Math.round(visibleH);

      const track = new PIXI.Graphics();
      track.beginFill(COLORS.stoneBg, 0.3);
      track.drawRect(trackX, trackY, scrollBarW, trackH);
      track.endFill();
      track.lineStyle(1, COLORS.goldDark, 0.3);
      track.drawRect(trackX, trackY, scrollBarW, trackH);
      track.stroke();
      root.addChild(track);

      const thumb = new PIXI.Graphics();
      root.addChild(thumb);

      const getScroll = () => this._unitsScrollYByPlayer[pIdx] || 0;
      const setScroll = (v) => { this._unitsScrollYByPlayer[pIdx] = v; };

      const updatePositions = () => {
        const maxScroll = rows.length * rowH - visibleH;
        setScroll(Math.max(0, Math.min(maxScroll, getScroll())));

        rowsContainer.y = Math.round(listTop - getScroll());

        const thumbH = Math.max(28, (visibleH / (rows.length * rowH)) * trackH);
        const free = trackH - thumbH;
        const ratio = (maxScroll > 0) ? (getScroll() / maxScroll) : 0;
        const ty = Math.round(trackY + ratio * free);

        thumb.clear();
        thumb.beginFill(COLORS.goldDark, 0.6);
        thumb.drawRect(trackX, ty, scrollBarW, thumbH);
        thumb.endFill();
        thumb.lineStyle(1, COLORS.goldBright, 0.5);
        thumb.drawRect(trackX, ty, scrollBarW, thumbH);
        thumb.stroke();
      };

      const scrollArea = new PIXI.Graphics();
      scrollArea.beginFill(0x000000, 0.001);
      scrollArea.drawRect(Math.round(tableX), Math.round(listTop), Math.round(tableW - scrollBarW), Math.round(visibleH));
      scrollArea.endFill();
      scrollArea.eventMode = 'static';
      root.addChild(scrollArea);

      scrollArea.on('wheel', (e) => {
        const delta = e.deltaY > 0 ? 40 : -40;
        setScroll(getScroll() + delta);
        updatePositions();
      });

      track.eventMode = 'static';
      track.on('pointerdown', (e) => {
        const localY = e.global.y - trackY;
        const maxScroll = rows.length * rowH - visibleH;
        const thumbH = Math.max(28, (visibleH / (rows.length * rowH)) * trackH);
        const free = trackH - thumbH;
        const ratio = free > 0 ? (localY - thumbH / 2) / free : 0;
        setScroll(Math.round(Math.max(0, Math.min(1, ratio)) * maxScroll));
        updatePositions();
      });

      updatePositions();
    } else {
      rowsContainer.y = Math.round(listTop);
    }
  }
}
