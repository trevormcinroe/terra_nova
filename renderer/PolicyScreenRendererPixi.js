/* =========================================================
   PolicyScreenRendererPixi – GPU replacement (Pixi v8)
   ========================================================= */
import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";

function toTitleCase(str){
  return str.replace(/\w\S*/g,
    txt => txt.charAt(0).toUpperCase()+txt.substr(1).toLowerCase());
}

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
  fontSize   : 14,
  fontWeight : 600,
  fill       : 0xffffff,
  align      : "center",
  wordWrap   : true,
});


const detailStylePID = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 16,
  fontWeight : 600,
  fill       : 0x0f0f1e,
});


export class PolicyScreenRendererPixi {
  constructor(app, policyArray){
    /* ---------- Pixi plumbing ---------- */
    this.app  = app;
    this.stage= new PIXI.Container();
    this.app.stage.addChild(this.stage);

    /* ---------- constants ---------- */
    this.pols   = policyArray;
    this.CARD_W = 290;  this.CARD_H = 380;
    this.SLOT_W =  90;  this.SLOT_H =  60;
    this.CARD_GAP = 18; this.ROW_GAP = 20;
    this.HEADER_H = 50;
    this.RADIUS   = 8;  this.BORDER = 3;

    this.PLAYER_X_OFF=[-10, 6, 22, 38, 54, 70];
    this.PLAYER_Y_OFF=130;
    //this.PLAYER_COLORS = ["#FF2A2F", "#FFD700", "#0A63BA", "#286B32", "#FFFFFF", "#91A1D6"]
    this.PLAYER_COLORS = constants.playerColorsScreens;

    this.CARD_ORDER=[
      ['tradition','liberty','honor','piety','patronage'],
      ['','rationalism','exploration','commerce','aesthetics']
    ];
    this.cardTint={
      tradition:0xf4e7ce, liberty:0xf4ecd6, honor:0xf1e4cf,
      piety:0xf7ead6, patronage:0xf5e8d4, ideology:0xe4d6bb,
      rationalism:0xf0e3ce, exploration:0xecdec8, commerce:0xf6e8d2,
      aesthetics:0xf2e5cf
    };

    /* ---------- calculate total content size and center offset ---------- */
    const totalWidth = 5 * this.CARD_W + 4 * this.CARD_GAP;  // 5 cards per row
    const totalHeight = 2 * this.CARD_H + this.ROW_GAP;       // 2 rows
    const viewWidth = app.renderer.width;
    const viewHeight = app.renderer.height;
    
    // Center horizontally, move up slightly vertically
    this.baseOffsetX = (viewWidth - totalWidth) / 2;
    this.baseOffsetY = (viewHeight - totalHeight) / 2 - 50;  // Move up by 50 pixels

    /* ---------- camera (horizontal pan) ---------- */
    this.offsetX=0; this.drag=false;
    this.stage.interactive=true;
    this.stage.on('pointerdown',e=>{
      this.drag=true; this.dragStart=e.data.global.x+this.offsetX;});
    this.stage.on('pointerup',_=>this.drag=false)
              .on('pointerupoutside',_=>this.drag=false)
              .on('pointermove',e=>{
                if(!this.drag)return;
                this.offsetX=this.dragStart-e.data.global.x;
                this.updateStagePosition();});
    this.stage.on('wheel',e=>{
      this.offsetX+=e.deltaY; 
      this.updateStagePosition(); 
      e.preventDefault();});

    /* ---------- pre-compute slot positions ---------- */
    this.nodePos=new Map();      // id → {x,y,isOpener}
    for(const [rowIdx,row] of this.CARD_ORDER.entries()){
      row.forEach((branch,colIdx)=>{
        const xCard=colIdx*(this.CARD_W+this.CARD_GAP) + this.baseOffsetX;
        const yCard=rowIdx*(this.CARD_H+this.ROW_GAP) + this.baseOffsetY;

        const branchPols=this.pols.filter(p=>p.tree===branch);

        /* rowsUsed = number of diamond rows (gy 1‒3) */
        const maxGy=Math.max(
          0,
          ...branchPols.filter(p=>p.grid[0]>0 && p.grid[0]<9)
                       .map(p=>p.grid[0]));
        const rowsUsed   = maxGy;                  // 0,1,2,3
        const rowGapInner= rowsUsed===3 ? 0 : 20;  // tighter if 3 rows

        branchPols.forEach(p=>{
          const [gy,gx]=p.grid;                // Civ-V coords
          const opener  = gy===0;
          const finisher= gy===9;

          const w=this.SLOT_W+(opener?80:0);
          const h=this.SLOT_H+(opener?-20:0);

          let localX,localY;
          if(opener){
            localX=(this.CARD_W-w)/2+10;
            localY=this.HEADER_H+35;
          }else if(finisher){
            localX=(this.CARD_W-w)/2;
            localY=this.CARD_H-h-18;
          }else{
            const colRel=gx-0.8;                 // 0,0.5,1,2…
            const rowRel=gy-1;                   // 0,1,2
            const colGap=(this.CARD_W-this.SLOT_W*3)/4;
            localX=colGap+colRel/2*(w+colGap);
            localY=this.HEADER_H+rowRel*1.5*(h+rowGapInner)+5;
          }

          this.nodePos.set(p.id,{
            x:xCard+localX,
            y:yCard+localY,
            isOpener:opener
          });
        });
      });
    }

    /* ---------- static layers ---------- */
    this.lBackground = new PIXI.Graphics();  // Background frame
    this.lCards   = new PIXI.Container();
    this.lConnect = new PIXI.Graphics();
    this.lNodes   = new PIXI.Container();
    this.lPlayer  = new PIXI.Container();
    this.stage.addChild(this.lBackground,this.lCards,this.lConnect,this.lNodes,this.lPlayer);

    this.drawBackground(totalWidth, totalHeight);
    this.buildCards();
    this.buildConnectors();
    this.buildNodes();
    
    // Set initial position
    this.updateStagePosition();
  }

  /* =========================================================
       BACKGROUND FRAME
     ========================================================= */
  drawBackground(totalWidth, totalHeight) {
    const padding = 30;  // Padding around the content
    const g = this.lBackground;
    
    // Draw dark semi-transparent background rectangle
    g.beginFill(0x1a1a1a, 0.85)
      .drawRect(
        this.baseOffsetX - padding,
        this.baseOffsetY - padding,
        totalWidth + padding * 2,
        totalHeight + padding * 2
      )
      .endFill();
    
    // Draw border frame
    g.lineStyle(3, 0x4a8a8a)
      .drawRect(
        this.baseOffsetX - padding,
        this.baseOffsetY - padding,
        totalWidth + padding * 2,
        totalHeight + padding * 2
      );
  }

  /* =========================================================
       STAGE POSITION UPDATE
     ========================================================= */
  updateStagePosition() {
    // Apply centering offset minus any user drag offset
    this.stage.x = -this.offsetX;
  }

  /* =========================================================
       PUBLIC API
     ========================================================= */
  start(){ this.app.ticker.add(this.drawPlayers,this); }
  stop(){  this.app.ticker.remove(this.drawPlayers,this); }
  setTurn(n){ this.turn=n; }
  setPlayerPolicies(arr){ this.playerPolicies=arr; }

  /* =========================================================
       STATIC LAYERS
     ========================================================= */
  buildCards(){
    for(const [rowIdx,row] of this.CARD_ORDER.entries()){
      row.forEach((branch,colIdx)=>{
        const xCard=colIdx*(this.CARD_W+this.CARD_GAP) + this.baseOffsetX;
        const yCard=rowIdx*(this.CARD_H+this.ROW_GAP) + this.baseOffsetY;

        const g=new PIXI.Graphics();
        // Draw the card background
        g.beginFill(this.cardTint[branch]??0xf0e6d6)
         .drawRect(xCard,yCard,this.CARD_W,this.CARD_H)
         .endFill();
        
        // Draw the header bar
        g.beginFill(0x81ADC8)
         .drawRect(xCard,yCard,this.CARD_W,this.HEADER_H)
         .endFill();
        
        // Add gold outline around entire card
        g.lineStyle(4, 0xB68F40)
         .drawRect(xCard,yCard,this.CARD_W,this.CARD_H)
          .stroke();
        
        this.lCards.addChild(g);

        const label=new PIXI.Text(branch.toUpperCase(), headerStyle);
        label.anchor.set(0.5);
        label.position.set(xCard+this.CARD_W/2,yCard+this.HEADER_H/2);
        this.lCards.addChild(label);
      });
    }
  }

  buildConnectors(){
    const g=this.lConnect;

    const centre=id=>{
      const n = this.nodePos.get(id);
      const yOff = n.isOpener ? -20 : 75;
      const xOff = n.isOpener ? 39 : 0;
      return{cx: n.x + xOff + this.SLOT_W / 2 - 10,
             cy: n.y + yOff + this.SLOT_H / 2};
    };

    for(const dest of this.pols){
      const {cx:cxD,cy:cyD}=centre(dest.id);
      for(const srcId of dest.prereq){
        if(!this.nodePos.has(srcId))continue;
        const {cx:cxS,cy:cyS}=centre(srcId);
        g.moveTo(cxD,cyD).lineTo(cxS,cyS);
        g.stroke({
          width: 4,
          color: 0x000000,
        });
      }
    }
  }

  buildNodes(){
    for(const p of this.pols){
      const node=this.nodePos.get(p.id);
      const opener = node.isOpener;
      const finisher=p.grid[0]===9;

      const w = this.SLOT_W + (opener ? 80 : 0);
      const h = this.SLOT_H + (opener ? -20 : 0);
      const yAdj = node.y + (opener ? -25 : 75);
      const xAdj = node.x - 10 + (opener ? 0 : 0);

      const g=new PIXI.Graphics()
        .lineStyle(this.BORDER,0xB68F40)
        .beginFill(0x0f0f1e)
        .drawRoundedRect(xAdj,yAdj,w,h,this.RADIUS * 0)
        .endFill();
      this.lNodes.addChild(g);

      const label=new PIXI.Text(toTitleCase(p.name.replace(/_/g,' ')), detailStyle);
      label.wordWrapWidth = w - 8;
      label.anchor.set(0.5);
      label.position.set(xAdj+w/2,yAdj+h/2);
      this.lNodes.addChild(label);
    }
  }

  /* =========================================================
       DYNAMIC PLAYER OVERLAYS
     ========================================================= */
  drawPlayers(){
    this.lPlayer.removeChildren();
    if(!this.playerPolicies?.length)return;

    const turnVec=this.playerPolicies[this.turn]??[];
    for(let pid=0;pid<turnVec.length;pid++){
      const polOwned=turnVec[pid];
      const col=this.PLAYER_COLORS[pid];

      for(const p of this.pols){
        if(polOwned[p.id]!==1)continue;
        const node=this.nodePos.get(p.id);
        const yOff=node.isOpener?8:this.PLAYER_Y_OFF;

        const badge=new PIXI.Graphics();
        badge.beginFill(col).lineStyle(this.BORDER,col)
             .drawRoundedRect(node.x+this.PLAYER_X_OFF[pid],
                              node.y+yOff,12,12,0)
             .endFill();
        this.lPlayer.addChild(badge);

        const txt=new PIXI.Text(String(pid + 1), detailStylePID);
        txt.anchor.set(0.5);
        txt.position.set(node.x+this.PLAYER_X_OFF[pid]+6,
                         node.y+yOff+6);
        this.lPlayer.addChild(txt);
      }
    }
  }
}
