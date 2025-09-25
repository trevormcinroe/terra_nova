import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';


export class PerformanceMonitor {
  constructor(app, options = {}) {
    this.app = app;
    this.options = options;
    this.fps = 0;
    this.lastTime = performance.now();
    this.frames = 0;
    
    this.position = options.position || { x: 10, y: 10 };
    this.style = options.style || {
      fill: 0x00FF00,
      fontSize: 20,
      fontFamily: 'monospace',
      stroke: 0x000000,
      strokeThickness: 3
    };
    
    this.createDisplay();
    this.app.ticker.add(() => this.update());
  }
  
  createDisplay() {
    this.container = new PIXI.Container();
    
    // Use Graphics() for PIXI v8
    this.bg = new PIXI.Graphics();
    this.bg.rect(0, 0, 150, 60);
    this.bg.fill({ color: 0x000000, alpha: 0.7 });
    this.container.addChild(this.bg);
    
    // PIXI v8 Text
    this.text = new PIXI.Text({ 
      text: 'FPS: 0', 
      style: this.style 
    });
    this.text.position.set(8, 8);
    this.container.addChild(this.text);
    
    this.container.position.set(this.position.x, this.position.y);
    //this.app.stage.addChild(this.container);
    const targetContainer = this.options.container || this.app.stage;
    targetContainer.addChild(this.container);
  }
  
  update() {
    this.frames++;
    const currentTime = performance.now();
    
    if (currentTime >= this.lastTime + 1000) {
      this.fps = Math.round((this.frames * 1000) / (currentTime - this.lastTime));
      this.text.text = `FPS: ${this.fps}`;
      
      // Update color
      if (this.fps >= 55) {
        this.text.style.fill = 0x00FF00;
      } else if (this.fps >= 30) {
        this.text.style.fill = 0xFFFF00;
      } else {
        this.text.style.fill = 0xFF0000;
      }
      
      this.frames = 0;
      this.lastTime = currentTime;
    }
  }
}
