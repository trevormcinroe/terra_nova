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
  fontSize   : 26,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 24,
  fontWeight : 600,
  fill       : 0xffffff,
  align      : "center",
  wordWrap   : true,
});

const detailStyleTenet = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 20,
  fontWeight : 600,
  fill       : 0xBB4430,
  align      : "right",
  //wordWrap   : true,
});

const detailStylePID = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 600,
  fill       : 0x0f0f1e,
});

let religionIcons = await constants.loadReligionIcons();

/* =========================================================
   ReligionScreenRendererPixi — GPU overlay (Pixi v8)
   ========================================================= */

/* ───────────────────────── helpers ───────────────────────── */
const TENET_LABELS  = ['Pantheon','Founder','Follower','Enhancer','Reformation'];
const ICON_SIZE     = 48;
const ICON_PAD      = 12;
const ROW_HEIGHT    = 42;
const COL_PAD       = 18;
const HEADER_H      = 45;
const Y_OFFSET      = 30;

// Civilization-style color palette
const COLORS = {
  bg: 0x1a1612,                    // Dark brown/black
  stoneBg: 0x2c2416,               // Stone texture base
  stoneLight: 0x3d3426,            // Stone highlight
  parchment: 0x3d3426,             // Parchment background
  goldBorder: 0xc9b037,            // Gold trim
  goldBright: 0xffd700,            // Bright gold
  goldDark: 0x8b7355,              // Dark gold/bronze
  headerBg: 0x81ADC8,              // Terracotta red header
  iconBg: 0x1a1612,                // Icon background
  labelText: 0xd4af37,             // Gold text
  categoryText: 0xa89968,          // Muted gold text
  tenetText: 0xffd700,             // Bright gold for tenets
  divider: 0x4a3f28,               // Divider lines
  shadow: 0x000000,                // Pure black shadows
  playerColors: [0x1e3a8a, 0x7e22ce, 0x0891b2, 0x16a34a, 0x94a3b8, 0xca8a04]
};

/* ========================================================= */
export class ReligionScreenRendererPixi {
  constructor(app, tenetNames, tenetCategories) {
    /* ------------- scene graph ---------------- */
    this.app   = app;
    this.stage = new PIXI.Container();
    this.app.stage.addChild(this.stage);

    this.tenetNames      = tenetNames;
    this.tenetCategories = tenetCategories;
    this.players         = [0, 1, 2, 3, 4, 5];

    /* ------------- geometry ------------------- */
    const { width: W, height: H } = app.renderer;
    this.rows    = 2;
    this.cols    = 3;
    this.padding = 20;
    this.cellW   = W / this.cols;
    this.cellH   = H / this.rows / 1.25;

    /* ------------- layers --------------------- */
    this.lBackground = new PIXI.Container();  // background effects
    this.lStatic  = new PIXI.Container();     // boxes, headers, labels
    this.lIcons   = new PIXI.Container();
    this.lDynamic = new PIXI.Container();     // per-turn tenets
    this.stage.addChild(this.lBackground, this.lStatic, this.lDynamic, this.lIcons);

    this.buildBackground();
    this.buildStaticLayout();
  }

  /* =========================================================
       PUBLIC API
     ========================================================= */
  setPlayerReligion(rel) { this.playerReligion = rel; }
  setTurn(turn)         { this.turn = turn; this.redraw(); }
  start()               { this.redraw(); }
  stop()                { /* nothing to detach */ }

  /* =========================================================
       BACKGROUND
     ========================================================= */
  buildBackground() {
    const { width: W, height: H } = this.app.renderer;
    const bg = new PIXI.Graphics();
    
    // Dark stone-like background
    bg.beginFill(COLORS.bg, 0.8);
    bg.drawRect(0 - 15, 0 + Y_OFFSET - 10, W + 25, H - 175);
    bg.endFill();
    
    // Add stone texture effect with rectangles
    for (let i = 0; i < 50; i++) {
      const x = Math.random() * W;
      const y = Math.random() * H;
      const w = 50 + Math.random() * 100;
      const h = 50 + Math.random() * 100;
      bg.beginFill(COLORS.stoneBg, Math.random() * 0.1);
      bg.drawRect(x, y, w, h);
      bg.endFill();
    }
    
    this.lBackground.addChild(bg);
  }

  /* =========================================================
       STATIC LAYOUT (draw once)
     ========================================================= */
  buildStaticLayout() {
    const W = this.cellW;
    const H = this.cellH;
    const pad = this.padding;

    /* iterate over the six player boxes */
    for (let i = 0; i < this.players.length; i++) {
      const row = Math.floor(i / this.cols);
      const col = i % this.cols;

      const xCell = col * W;
      const yCell = row * H;

      /* inner box coordinates */
      const boxX = xCell + pad / 2;
      const boxY = yCell + pad / 2 + Y_OFFSET;
      const boxW = W - pad;
      const boxH = H - pad;
      
      const iconX = boxX + ICON_PAD + ICON_SIZE / 2;
      const iconY = boxY + HEADER_H + ICON_SIZE / 2 + 15;
    
      // Inserting religion icons  
      const relIcon = new PIXI.Sprite(religionIcons[i]);
      relIcon.anchor.set(0.5, 0.5);
      relIcon.scale.set(constants.religionConfigs[i].s);
      relIcon.position.set(iconX - 35 + boxW / 2, iconY + 20);
      this.lIcons.addChild(relIcon);


      /* make a graphics node per cell */
      const G = new PIXI.Graphics();

      /* ── drop shadow ─────────────────────────── */
      G.beginFill(COLORS.shadow, 0.5);
      G.drawRect(boxX + 3, boxY + 3, boxW, boxH);
      G.endFill();

      /* ── main panel - stone/parchment look ────── */
      // Base fill
      G.beginFill(COLORS.parchment);
      G.drawRect(boxX, boxY, boxW, boxH);
      G.endFill();

      // Add some texture variation
      for (let t = 0; t < 3; t++) {
        G.beginFill(COLORS.stoneLight, 0.1);
        G.drawRect(
          boxX + Math.random() * boxW * 0.8,
          boxY + Math.random() * boxH * 0.8,
          20 + Math.random() * 40,
          20 + Math.random() * 40
        );
        G.endFill();
      }

      /* ── header section with ornate border ─────── */
      // Header background
      G.beginFill(COLORS.headerBg);
      G.drawRect(boxX, boxY, boxW, HEADER_H);
      G.endFill();

      // Gold trim on header bottom
      G.beginFill(COLORS.goldBorder);
      G.drawRect(boxX, boxY + HEADER_H - 3, boxW, 3);
      G.endFill();

      // Ornamental corner pieces
      const cornerSize = 8;
      G.beginFill(COLORS.goldBright);
      // Top corners
      G.drawPolygon([
        boxX, boxY,
        boxX + cornerSize, boxY,
        boxX, boxY + cornerSize
      ]);
      G.drawPolygon([
        boxX + boxW, boxY,
        boxX + boxW - cornerSize, boxY,
        boxX + boxW, boxY + cornerSize
      ]);
      // Bottom corners
      G.drawPolygon([
        boxX, boxY + boxH,
        boxX + cornerSize, boxY + boxH,
        boxX, boxY + boxH - cornerSize
      ]);
      G.drawPolygon([
        boxX + boxW, boxY + boxH,
        boxX + boxW - cornerSize, boxY + boxH,
        boxX + boxW, boxY + boxH - cornerSize
      ]);
      G.endFill();

      /* ── main border - double line effect ─────── */
      // Outer gold border
      G.lineStyle(3, COLORS.goldDark);
      G.drawRect(boxX, boxY, boxW, boxH);
      G.stroke();

      // Inner gold border
      G.lineStyle(1, COLORS.goldBright);
      G.drawRect(boxX + 4, boxY + 4, boxW - 8, boxH - 8);
      G.stroke();

      /* ── icon frame - circular style ──────────── */

      // Draw circular frame with double border
      const circleRadius = (ICON_SIZE + 50) / 2;
      
      // Outer circle - filled
      G.beginFill(0x0f0f1e);
      G.lineStyle(3, COLORS.goldBorder);
      G.drawCircle(iconX  - 35 + boxW / 2, iconY + 20, circleRadius);
      G.endFill();
      G.stroke();

      // Inner decorative circle
      //G.lineStyle(2, COLORS.playerColors[i]);
      //G.drawCircle(iconX, iconY, circleRadius - 6);
      //G.stroke();
      //
      //// Small inner circle for additional detail
      //G.lineStyle(1, COLORS.goldDark, 0.5);
      //G.drawCircle(iconX, iconY, circleRadius - 10);
      //G.stroke();

      /* ── Roman numeral in icon ───────────────── */
      const romanNumerals = ['1', '2', '3', '4', '5', '6'];
      const pNum = new PIXI.Text(romanNumerals[i], detailStyle);
      pNum.anchor.set(0.5);
      pNum.position.set(iconX, iconY);
      this.lStatic.addChild(pNum);

      /* ── tenet rows with decorative dividers ──── */
      const rowBaseY = boxY + HEADER_H + 100;
      
      for (let r = 0; r < TENET_LABELS.length; r++) {
        const lineY = rowBaseY + r * ROW_HEIGHT + ROW_HEIGHT - 6;
        
        // Main divider line
        G.lineStyle(1, COLORS.divider);
        G.moveTo(boxX + COL_PAD + ICON_SIZE + 25, lineY);
        G.lineTo(boxX + boxW - COL_PAD, lineY);
        G.stroke();
        
        // Decorative diamond shapes
        const diamondSize = 3;
        G.beginFill(COLORS.goldDark);
        G.lineStyle(0);
        // Left diamond
        //G.drawPolygon([
        //  boxX + COL_PAD + ICON_SIZE + 20, lineY,
        //  boxX + COL_PAD + ICON_SIZE + 20 - diamondSize, lineY - diamondSize,
        //  boxX + COL_PAD + ICON_SIZE + 20, lineY - diamondSize * 2,
        //  boxX + COL_PAD + ICON_SIZE + 20 + diamondSize, lineY - diamondSize
        //]);
        // Right diamond
        G.drawPolygon([
          boxX + boxW - COL_PAD + 5, lineY,
          boxX + boxW - COL_PAD + 5 - diamondSize, lineY - diamondSize,
          boxX + boxW - COL_PAD + 5, lineY - diamondSize * 2,
          boxX + boxW - COL_PAD + 5 + diamondSize, lineY - diamondSize
        ]);
        G.endFill();
      }

      /* add graphics to static layer */
      this.lStatic.addChild(G);

      /* ── static text labels ───────────────────── */
      
      // Player name with engraved effect
      const shadowText = new PIXI.Text(`PLAYER ${romanNumerals[i]}`, headerStyle);
      shadowText.anchor.set(0.5);
      shadowText.position.set(xCell + W / 2 + 1, boxY + HEADER_H / 2 + 1);
      this.lStatic.addChild(shadowText);

      const pLab = new PIXI.Text(`PLAYER ${romanNumerals[i]}`, headerStyle);
      pLab.anchor.set(0.5);
      pLab.position.set(xCell + W / 2, boxY + HEADER_H / 2);
      this.lStatic.addChild(pLab);

      // Category labels with small icon markers
      for (let r = 0; r < TENET_LABELS.length; r++) {
        const cy = rowBaseY + r * ROW_HEIGHT + ROW_HEIGHT / 2;

        // Small decorative square
        const markerG = new PIXI.Graphics();
        markerG.beginFill(COLORS.goldDark);
        markerG.drawRect(boxX + COL_PAD + 4, cy - 4, 8, 8);
        markerG.endFill();
        markerG.lineStyle(1, COLORS.goldBorder);
        markerG.drawRect(boxX + COL_PAD + 4, cy - 4, 8, 8);
        markerG.stroke();
        this.lStatic.addChild(markerG);

        // Category label
        const lbl = new PIXI.Text(TENET_LABELS[r], detailStyle);
        lbl.anchor.set(0, 0.5);
        lbl.position.set(boxX + COL_PAD + 18, cy);
        this.lStatic.addChild(lbl);

        // Empty state indicator
        const emptyText = new PIXI.Text('—', {
          fontSize: 14,
          fill: 0x5a4f38,
        });
        emptyText.anchor.set(1, 0.5);
        emptyText.position.set(boxX + boxW - COL_PAD, cy);
        this.lStatic.addChild(emptyText);
      }

      // Add decorative flourish at bottom
      const flourishY = boxY + boxH - 15;
      const flourishG = new PIXI.Graphics();
      flourishG.lineStyle(1, COLORS.goldDark, 0.5);
      flourishG.moveTo(boxX + boxW / 2 - 30, flourishY);
      flourishG.lineTo(boxX + boxW / 2 - 10, flourishY);
      flourishG.stroke();
      flourishG.moveTo(boxX + boxW / 2 + 10, flourishY);
      flourishG.lineTo(boxX + boxW / 2 + 30, flourishY);
      flourishG.stroke();
      
      flourishG.beginFill(COLORS.goldDark, 0.8);
      flourishG.drawPolygon([
        boxX + boxW / 2, flourishY - 2,
        boxX + boxW / 2 - 4, flourishY + 2,
        boxX + boxW / 2, flourishY + 6,
        boxX + boxW / 2 + 4, flourishY + 2
      ]);
      flourishG.endFill();
      this.lStatic.addChild(flourishG);
    }
  }

  /* =========================================================
       DYNAMIC TENET TEXT (per turn)
     ========================================================= */
  redraw() {
    this.lDynamic.removeChildren();
    if (!this.playerReligion?.length) return;

    const turnVec = this.playerReligion[this.turn] ?? [];

    for (let pIdx = 0; pIdx < this.players.length; pIdx++) {
      const row = Math.floor(pIdx / this.cols);
      const col = pIdx % this.cols;

      const cellX = col * this.cellW + this.padding / 2;
      const cellY = row * this.cellH + this.padding / 2 + HEADER_H + 10;
      const boxW  = this.cellW - this.padding;

      /* gather tenets by category */
      const entries = [[], [], [], [], []];
      this.tenetNames.forEach((name, id) => {
        if (turnVec[pIdx]?.[id]) {
          const cat = this.tenetCategories[id];
          const dst = cat === 'pantheon'   ? 0
                    : cat === 'founder'    ? 1
                    : cat === 'follower'   ? 2
                    : cat === 'enhancer'   ? 3
                    : cat === 'reformation'? 4 : -1;
          if (dst >= 0) entries[dst].push(name);
        }
      });

      /* create Text nodes for each row */
      for (let r = 0; r < TENET_LABELS.length; r++) {
        if (!entries[r].length) continue;

        // Text with shadow for depth
        //const shadowTxt = new PIXI.Text(entries[r].join(' · '), detailStyleTenet)
        ////const shadowTxt = new PIXI.Text(entries[r].join(' · '), {
        ////  fontSize: 12,
        ////  fill: COLORS.shadow,
        ////  fontWeight: '500',
        ////  align: 'right',
        ////  wordWrap: true,
        ////  wordWrapWidth: boxW - COL_PAD * 2 - 120,
        ////});
        //shadowTxt.anchor.set(1, 0.5);
        //shadowTxt.position.set(
        //  cellX + boxW - COL_PAD + 1,
        //  cellY + r * ROW_HEIGHT + ROW_HEIGHT / 2 + 1
        //);
        //this.lDynamic.addChild(shadowTxt);

        const txt = new PIXI.Text(entries[r].join(', '), detailStyleTenet);
        txt.anchor.set(1, 0.5);
        txt.position.set(
          cellX + boxW - COL_PAD - 8,
          cellY + 125 + r * ROW_HEIGHT + ROW_HEIGHT / 2
        );
        this.lDynamic.addChild(txt);
      }
    }
  }
}
