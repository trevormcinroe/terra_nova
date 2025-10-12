import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";
let cityYieldsTextures = await constants.loadCityYieldsTexture();

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
  return document.fonts.ready;  // resolves when fonts are usable
}

await loadGoogleFonts();
const headerStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 64,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 18,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyleSmall = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 600,
  fill       : 0x81ADC8,
});

const detailStylePID = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 600,
  fill       : 0x0f0f1e,
});

function toTitleCase(str) {
  return str.replace(
    /\w\S*/g,
    function(txt){
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    }
  );
}

/****************************************************************
 * 2️⃣  TECH-TREE RENDERER  (Pixi v8)                           *
 ****************************************************************/
export class TechTreeRendererPixi {
  constructor(app, techArray) {
    this.app = app;
    this.stage = new PIXI.Container();  // isolate entire tree
    this.app.stage.addChild(this.stage);

    /* data -------------------------------------------------- */
    this.techs = techArray;                   // static tech list
    this.turn = 0;
    this.playerTechs = [];                    // injected later

    /* constants (same as old) ------------------------------- */
    this.TILE_W = 220;
    this.TILE_H = 60;
    this.COL_GAP = 60;
    this.ROW_GAP = 1;
    this.RADIUS = 5;
    this.BORDER = 5;
    this.PLAYER_X_OFF=[85,110,135,160,185,210];
    this.PLAYER_Y_OFF=50;
    this.PLAYER_COLORS = constants.playerColorsScreens;

    /* camera ----------------------------------------------- */
    this.offsetX = 0; this.offsetY = 0;
    this.dragging = false;
    this.dragStart= {x:0,y:0};
    this.stage.interactive = true;
    this.stage.on('pointerdown', this.onDown.bind(this));
    this.stage.on('pointerup', this.onUp.bind(this));
    this.stage.on('pointerupoutside', this.onUp.bind(this));
    this.stage.on('pointermove', this.onMove.bind(this));
    this.stage.on('wheel', this.onWheel.bind(this));

    /* pre-compute node positions ---------------------------- */
    this.nodePos = new Map();
    this.maxX = 0;
    this.maxY = 0;
    for (const t of this.techs) {
      const [row,col] = t.grid;
      const x = col * (this.TILE_W+this.COL_GAP);
      const y = row * (this.TILE_H+this.ROW_GAP) + 180;
      this.nodePos.set(t.id,{x,y});
      this.maxX = Math.max(this.maxX, x + this.TILE_W);
      this.maxY = Math.max(this.maxY, y + this.TILE_H);
    }

    /* layers: order = background, background strip, dividers, edges, nodes, player dots */
    this.lBackground = new PIXI.Graphics();
    this.lEraBG = new PIXI.Container();
    this.lDividers = new PIXI.Graphics();
    this.lEdges = new PIXI.Graphics();
    this.lNodes = new PIXI.Container();
    this.lPlayers = new PIXI.Container();
    this.stage.addChild(this.lBackground,this.lEraBG,this.lDividers,this.lEdges,
                        this.lNodes,this.lPlayers);

    /* build static visuals once ----------------------------- */
    this.drawBackground();
    this.drawEraColumns();
    this.drawEraDividers();
    this.drawEdges();
    this.drawNodes();
  }

  /* BACKGROUND --------------------------------------------- */
  drawBackground() {
    const g = this.lBackground;
    // Draw a large background rectangle in the tech tree color
    // Make it much wider than content to ensure full coverage when scrolling
    const bgWidth = Math.max(this.maxX + 2000, this.app.renderer.width * 3);
    g.beginFill(0x242119, 0.8)
     .drawRect(-1000, -1000, bgWidth, this.maxY + 2000)
     .endFill();
  }

  /* SCROLLBAR ---------------------------------------------- */
  createScrollbar() {
    const viewWidth = this.app.renderer.width;
    const viewHeight = this.app.renderer.height;
    const contentWidth = this.maxX + 100; // Add some padding
    
    // Only show scrollbar if content is wider than view
    if (contentWidth <= viewWidth) return;
    
    const scrollbarHeight = 20;
    const scrollbarY = viewHeight - scrollbarHeight - 10;
    
    // Scrollbar track
    this.scrollTrack = new PIXI.Graphics();
    this.scrollTrack.beginFill(0x1a1a1a, 0.7)
                    .drawRoundedRect(10, scrollbarY, viewWidth - 20, scrollbarHeight, 10)
                    .endFill();
    
    // Scrollbar thumb
    const thumbWidth = Math.max(50, (viewWidth / contentWidth) * (viewWidth - 20));
    this.scrollThumb = new PIXI.Graphics();
    this.scrollThumb.beginFill(0x6a6a6a)
                    .drawRoundedRect(0, 0, thumbWidth, scrollbarHeight, 10)
                    .endFill();
    this.scrollThumb.position.set(10, scrollbarY);
    
    // Make thumb interactive
    this.scrollThumb.eventMode = 'static';
    this.scrollThumb.cursor = 'pointer';
    
    // Thumb dragging
    let thumbDragging = false;
    let thumbDragStart = 0;
    
    this.scrollThumb.on('pointerdown', (e) => {
      thumbDragging = true;
      thumbDragStart = e.data.global.x - this.scrollThumb.x;
    });
    
    const onThumbMove = (e) => {
      if (!thumbDragging) return;
      
      const newX = e.data.global.x - thumbDragStart;
      const minX = 10;
      const maxX = viewWidth - 10 - thumbWidth;
      
      const clampedX = Math.max(minX, Math.min(maxX, newX));
      this.scrollThumb.x = clampedX;
      
      // Update stage position
      const scrollPercent = (clampedX - minX) / (maxX - minX);
      this.offsetX = scrollPercent * (contentWidth - viewWidth);
      this.stage.position.x = -this.offsetX;
    };
    
    const onThumbUp = () => {
      thumbDragging = false;
    };
    
    this.app.stage.on('pointermove', onThumbMove);
    this.app.stage.on('pointerup', onThumbUp);
    this.app.stage.on('pointerupoutside', onThumbUp);
    
    // Add click on track to jump
    this.scrollTrack.eventMode = 'static';
    this.scrollTrack.cursor = 'pointer';
    this.scrollTrack.on('pointerdown', (e) => {
      if (e.target === this.scrollThumb) return;
      
      const clickX = e.data.global.x;
      const targetX = clickX - thumbWidth / 2;
      const minX = 10;
      const maxX = viewWidth - 10 - thumbWidth;
      
      const clampedX = Math.max(minX, Math.min(maxX, targetX));
      this.scrollThumb.x = clampedX;
      
      const scrollPercent = (clampedX - minX) / (maxX - minX);
      this.offsetX = scrollPercent * (contentWidth - viewWidth);
      this.stage.position.x = -this.offsetX;
    });
    
    this.scrollbarContainer.addChild(this.scrollTrack, this.scrollThumb);
    
    // Store values for updating thumb position when dragging the main view
    this.scrollbarData = {
      thumbWidth,
      viewWidth,
      contentWidth,
      minX: 10,
      maxX: viewWidth - 10 - thumbWidth
    };
  }

  /* Update scrollbar thumb position when main view is dragged */
  updateScrollbarPosition() {
    if (!this.scrollbarData || !this.scrollThumb) return;
    
    const { minX, maxX, viewWidth, contentWidth } = this.scrollbarData;
    const scrollPercent = this.offsetX / (contentWidth - viewWidth);
    const thumbX = minX + scrollPercent * (maxX - minX);
    this.scrollThumb.x = Math.max(minX, Math.min(maxX, thumbX));
  }

  /* EVENT HANDLERS ----------------------------------------- */
  onDown(e){
    this.dragging=true;
    const p=e.data.global;
    this.dragStart={ x:p.x+this.offsetX, y:p.y+this.offsetY };
  }
  
  onUp(){ this.dragging=false; }
  
  onMove(e){
    if(!this.dragging)return;
    const p=e.data.global;
    
    // Only allow horizontal movement
    const newOffsetX = this.dragStart.x - p.x;
    
    // Clamp horizontal movement - ensure we can scroll to see all content
    const viewWidth = this.app.renderer.width;
    const contentWidth = this.maxX + 550; // Add padding to ensure rightmost techs are fully visible
    const maxOffset = Math.max(0, contentWidth - viewWidth);
    this.offsetX = Math.max(0, Math.min(maxOffset, newOffsetX));
    
    // Keep Y locked at 0
    this.offsetY = 0;
    this.stage.position.set(-this.offsetX, 0);
  }
  
  onWheel(e){
    // Only allow horizontal scrolling with mouse wheel
    const scrollAmount = e.deltaY * 0.5; // Reduce scroll speed
    const viewWidth = this.app.renderer.width;
    const contentWidth = this.maxX + 300; // Add padding to ensure rightmost techs are fully visible
    const maxOffset = Math.max(0, contentWidth - viewWidth);
    
    this.offsetX = Math.max(0, Math.min(maxOffset, this.offsetX + scrollAmount));
    this.stage.position.x = -this.offsetX;
    
    e.preventDefault();
  }

  /* PUBLIC API --------------------------------------------- */
  start(){ this.app.ticker.add(this.drawPlayers,this); }
  stop(){  this.app.ticker.remove(this.drawPlayers,this); }

  setTurn(n){ this.turn=n; }
  setPlayerTechs(arr){ this.playerTechs=arr; }

  /* STATIC LAYERS (draw once) ------------------------------ */
  drawEraColumns(){
    const eraGroups={};
    for(const t of this.techs)(eraGroups[t.era]??=[]).push(t);

    const eraOrder=['ancient','classical','medieval','renaissance',
                    'industrial','modern','postmodern','future'];
    const colW=this.TILE_W+this.COL_GAP;

    for(const era of eraOrder){
      const list = eraGroups[era]; 
      if (!list) continue;
      const min = Math.min(...list.map(t=>t.grid[1]));
      const max = Math.max(...list.map(t=>t.grid[1]));
      const x0 = min * colW - this.COL_GAP / 2;
      const w = (max + 1) * colW - this.COL_GAP / 2 - x0;

      const g=new PIXI.Graphics();
      g.beginFill(0xffffff, 0.08).drawRect(x0, 0, w, 2000).endFill();

      const label=new PIXI.Text(era.toUpperCase(), detailStyle);
      label.anchor.set(0.5, 0);
      label.position.set(x0 + w / 2, 125);
      this.lEraBG.addChild(g,label);
    }
  }

  drawEraDividers(){
    const eraGroups={};
    for (const t of this.techs) (eraGroups[t.era]??=[]).push(t);

    const eraOrder = ['ancient','classical','medieval','renaissance',
                      'industrial','modern','postmodern','future'];
    const colW = this.TILE_W + this.COL_GAP;

    const g = this.lDividers;
    g.lineStyle(4, 0xffffff, 0.25);
    for(let i = 0; i < eraOrder.length - 1; i++){
      const list = eraGroups[eraOrder[i]]; 
      if (!list) continue;
      const max = Math.max(...list.map(t=>t.grid[1]));
      const x = (max + 1) * colW - this.COL_GAP / 2;
      g.moveTo(x, 0).lineTo(x, 2000);
      g.stroke();
    }
  }

  drawEdges(){
    const g = this.lEdges;
    g.lineStyle(3, 0xc1c1c1);
    for (const t of this.techs) {
      const {x: x1, y: y1} = this.nodePos.get(t.id);
      const midY = y1 + this.TILE_H / 2;

      for (const p of t.prereq) {
        const {x: x0, y: y0} = this.nodePos.get(p);
        if(x0 === x1){
          g.moveTo(x0 + this.TILE_W - 2, y0 + this.TILE_H / 2)
           .lineTo(x1 - 6, midY);
        }else{
          const elbow = x1 - this.COL_GAP / 2;
          g.moveTo(x0 + this.TILE_W, y0 + this.TILE_H / 2)
           .lineTo(elbow, y0+this.TILE_H/2)
           .lineTo(elbow, midY)
           .lineTo(x1 - 6, midY);
          g.stroke();
        }
      }
    }
  }

  drawNodes(){
    for (const t of this.techs) {
      const {x, y} = this.nodePos.get(t.id);
      const g = new PIXI.Graphics();
      g.lineStyle(this.BORDER * 0.5, "#B68F40").beginFill("#0f0f1e", 0.95);
      g.drawRoundedRect(x, y, this.TILE_W, this.TILE_H, this.RADIUS * 0).endFill();
      
      const techName = toTitleCase(t.name.replace(/_/g,' '));
      const name = new PIXI.Text(techName, detailStyle);
      name.position.set(x + 8,y + 8);

      const icon = new PIXI.Sprite(cityYieldsTextures[6]);
      icon.anchor.set(0.5);
      icon.scale.set(constants.cityYieldsDef[6].s / 1.5);
      icon.position.set(x + 15, y + 37);

      const cost=new PIXI.Text(`${t.cost}`, detailStyleSmall);
      cost.position.set(x + 25, y + 28);
      this.lNodes.addChild(g, name, cost, icon);
    }
  }

  /* DYNAMIC layer: player overlays ------------------------- */
  drawPlayers(){
    this.lPlayers.removeChildren();

    if (!this.playerTechs.length) return;
    const turnVec = this.playerTechs[this.turn] ?? [];

    for (let pid=0; pid < turnVec.length; pid++) {
      const techOwned = turnVec[pid];
      const color = this.PLAYER_COLORS[pid];

      for (const t of this.techs) {
        if (techOwned[t.id] !== 1) continue;
        const {x, y} = this.nodePos.get(t.id);
        const badge = new PIXI.Graphics();
        badge.lineStyle(this.BORDER, color)
             .beginFill(color)
             .drawRoundedRect(
               x + this.PLAYER_X_OFF[pid],
               y + this.PLAYER_Y_OFF,
               20, 
               20,
               0).endFill();

        const txt = new PIXI.Text(String(pid + 1), detailStylePID);
        txt.anchor.set(0.5);
        txt.position.set(
          x + this.PLAYER_X_OFF[pid]+10,
          y + this.PLAYER_Y_OFF+10);

        this.lPlayers.addChild(badge,txt);
      }
    }
  }
}
