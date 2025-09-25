import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';
import * as constants from "./constants.js";
import { TechTreeRendererPixi } from './TechTreeRendererPixi.js';
import { PolicyScreenRendererPixi } from './PolicyScreenRendererPixi.js';
import { ReligionScreenRendererPixi } from './ReligionScreenRendererPixi.js';
import { VictoryScreenRendererPixi } from './VictoryScreenRendererPixi.js';
import { DemographicsScreenRendererPixi } from './DemographicsScreenPixi.js';
import { PerformanceMonitor } from "./PerformanceMonitor.js"
import { TradeScreenRendererPixi } from "./TradeScreenRendererPixi.js"

await document.fonts.load('600 24px "Cinzel"');
await document.fonts.load('600 24px "Crimson Text"');


const headerStyle = new PIXI.TextStyle({
  fontFamily : 'Cinzel',
  fontSize   : 26,
  fontWeight : 600,
  fill       : 0xffffff,
});

const detailStyle = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 22,
  fontWeight : 600,
  fill       : 0xffffff,
  resolution: window.devicePixelRatio * 2
});

const detailStyle2 = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 22,
  fontWeight : 600,
  fill       : 0xffffff,
  resolution: window.devicePixelRatio * 2
});

const detailStyleCenter = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 14,
  fontWeight : 600,
  fill       : 0xffffff,
  align      : "center",
});

const detailStyleRight = new PIXI.TextStyle({
  fontFamily : 'Crimson Text',
  fontSize   : 22,
  fontWeight : 600,
  fill       : 0xffffff,
  align      : "right",
  resolution: window.devicePixelRatio * 2
});
// Create a new application
const app = new PIXI.Application();
// Initialize with options
await app.init({
    width: 1920,           // Canvas width
    height: 1080,          // Canvas height
    backgroundColor: 0x000000, // Background color
    antialias: true,     // Enable antialiasing
    //resolution: 1,       // Resolution / device pixel ratio
    resolution: window.devicePixelRatio || 1,
    preference: 'webgl', // or 'webgpu' // Renresolution: window.devicePixelRatio || 1,
    autoDensity: true,
    resizeTo: document.querySelector('.pixi-container'),
    //roundPixels: true,  
});

const world = new PIXI.Container();

const mapLayer = new PIXI.Container();
mapLayer.interactive = true;       // enables pointer events
mapLayer.cacheAsTexture = true;
//app.stage.addChild(mapLayer);

const elevationLayer = new PIXI.Container();
elevationLayer.sortableChildren = false;      // faster
elevationLayer.cacheAsTexture = true;
//app.stage.addChild(elevationLayer); // on top of mapLayer

const riverLayer = new PIXI.Graphics();
riverLayer.sortableChildren = false;
riverLayer.cacheAsTexture = true;


const featureLayer = new PIXI.Container();
featureLayer.sortableChildren = false;
//featureLayer.cacheAsTexture = true;

const nwLayer = new PIXI.Container();
nwLayer.sortableChildren = false;
nwLayer.cacheAsTexture = true;

const borderLayer = new PIXI.Graphics();

const resourceLayer = new PIXI.Container();
resourceLayer.sortableChildren = false;
resourceLayer.cacheAsTexture = true;

const improvementsLayer = new PIXI.Container();
improvementsLayer.sortableChildren = false;

const yieldLayer = new PIXI.Container();
yieldLayer.sortableChildren = false;

const unitsLayer = new PIXI.Container();
unitsLayer.sortableChildren = false;

const cityLayer = new PIXI.Container();
cityLayer.sortableChildren = false;

const cityUILayer = new PIXI.Container();
// Needs to scale with  zoom, so we have to add to world
const workedTileLayer = new PIXI.Graphics();

const traderouteLayer = new PIXI.Container();
traderouteLayer.sortableChildren = false;

const ownershipLayer = new PIXI.Container();
ownershipLayer.sortableChildren = false;

const cityNameLayer = new PIXI.Container();
cityNameLayer.sortableChildren = false;

const fowLayer = new PIXI.Container();
fowLayer.sortableChildren = false;

const debugLayer = new PIXI.Container();
debugLayer.sortableChildren = false;


world.addChild(
  mapLayer, ownershipLayer, riverLayer, elevationLayer, featureLayer, nwLayer, borderLayer, resourceLayer,
  improvementsLayer, yieldLayer, cityLayer, workedTileLayer, traderouteLayer, cityNameLayer, unitsLayer, fowLayer, debugLayer
);
app.stage.addChild(world);
app.stage.addChild(cityUILayer);

// Create FPS monitor container at the highest level
const fpsLayer = new PIXI.Container();
fpsLayer.zIndex = 10000;
app.stage.addChild(fpsLayer);

// Modify the monitor creation to use this layer
const monitor = new PerformanceMonitor(app, {
  position: { x: 10, y: app.screen.height - 300 },  // bottom-left
  container: fpsLayer  // pass the layer
});

// Add the canvas to your webpage
document.body.appendChild(app.canvas);



const rows = 42;
const cols = 66;

const hexRadius = 40;
const hexWidth = Math.sqrt(3) * hexRadius;
const hexHeight = hexRadius * 2;
const vertSpacing = hexHeight * 0.75;
const horizSpacing = hexWidth;         // horizontal distance between hex centers

// Declare global state variables at the top
let terrainMap, riverMap, lakeMap, elevationMap, featureMap, nwMap;
let unitsType, unitsMilitary, unitsRowCol, unitsTradePlayerTo, unitsTradeCityTo, unitsTradeCityFrom, unitsTradeYield, unitsEngaged, unitHealth;
let cs_cities, csOwnership, playerCities, playerOwnership;
let csOwnershipBorders, playerOwnershipBorders;
let allResourceMap, gtYieldMap, playerYieldMap;
let movementCostMap, ISREPLAY, numDelegates;
let playerTechs, playerPolicies, playerReligion;
let workedSlots, playerYields, playerPops, playerBuildings;
let playerWonderAccel, playerBldgAccel, playerMilitaryBldgAccel;
let playerReligionBldgAccel, playerCultureBldgAccel, playerSeaBldgAccel;
let playerScienceBldgAccel, playerEconBldgAccel, playerCityReligion;
let playerGWSlots, playerUnitAccel, playerYieldAccel, playerBorderAccel;
let playerSpecialistSlots, playerBldgMaintenance, playerUnitXPAdd;
let playerCanTradeFood, playerCanTradeProd, playerDefense, playerHP;
let tradeGoldAddOwner, tradeGoldAddDest, tradeLandDistMod, tradeSeaDistMod;
let playerGPAccel, playerMountedAccel, playerLandUnitAccel;
let playerTechStealReduce, playerSeaUnitAccel, playerGWTourismAccel;
let playerCultureToTourism, playerAirUnitCapacity, playerSpaceshipProdAccel;
let playerNavalMovementAdd, playerNavalSightAdd, playerCityConnectionGoldAccel;
let playerArmoredAccel, improvementMap, roadMap;
let NUMTURNS;
let tourismTotal, cultureTotal, gpps, gpThreshold, goldenAgeTurns;
let isConstructing, prodReserves;
let csReligiousPopulation, csRelationships, csInfluence, csType, csQuest, csCultureTracker, csFaithTracker, csTechTracker, csTradeTracker, csReligionTracker, csWonderTracker, csResourceTracker
let playerReligiousPop, resourcesOwned;
let fow;
let tradeLedger, tradeLengthLedger, tradeGPTAdj, tradeResourceAdj; 
let haveMet, atWar, hasSacked;
let cityHP, cityDefense;
let treasury, happiness;
let unitsCombatBonus;

let terrainTextures = await constants.loadTerrainTextures();
let elevTextures = await constants.loadElevationTextures();
let featureTextures = await constants.loadFeatureTextures();
let nwTextures = await constants.loadNWTextures();
let resTextures = await constants.loadResourceTextures();
let impTextures = await constants.loadImprovementTextures(); 
let yieldTextures = await constants.loadYieldTextures();
let unitBGTextures = await constants.loadUnitBGTextures();
let unitTextures   = await constants.loadUnitTextures();
let cityTexture = await constants.loadCityTexture();
let capTexture = await constants.loadCapTexture();
let cityYieldsTextures = await constants.loadCityYieldsTexture();
let palaceTexture = await constants.loadPalaceTexture();
let csTypesTextures = await constants.loadCsIcons();
let religionIcons = await constants.loadReligionIcons();


// Dropdown menu to select the gamefile to render
function countNestedArrays(arr) {
  if (!Array.isArray(arr)) return 0;

  return arr.reduce((count, item) => {
    return count;
  }, 0);
}


const Loader = (() => {
  let el, msgEl;

  function ensure() {
    if (el) return;

    // Root overlay
    el = document.createElement('div');
    el.id = 'loading-overlay';
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-live', 'polite');
    Object.assign(el.style, {
      position: 'fixed',
      inset: '0',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'rgba(26,22,18,0.82)',
      backdropFilter: 'blur(2px)',
      zIndex: '9999',
    });

    // Panel (parchment / stone)
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      minWidth: '320px',
      maxWidth: '72vw',
      padding: '18px 22px 20px',
      background: 'linear-gradient(180deg,#3d3426 0%, #2c2416 100%)',
      borderRadius: '12px',
      border: '1px solid #8b7355',  // goldDark
      boxShadow: '0 8px 32px rgba(0,0,0,.45)',
      position: 'relative',
      color: '#fff',
      textAlign: 'center',
    });

    // Inner inset border
    const inset = document.createElement('div');
    Object.assign(inset.style, {
      position: 'absolute',
      inset: '6px',
      borderRadius: '8px',
      border: '1px solid #ffd700', // goldBright
      pointerEvents: 'none',
    });
    panel.appendChild(inset);

    // Corner ornaments
    ['tl','tr','bl','br'].forEach((pos) => {
      const c = document.createElement('div');
      Object.assign(c.style, {
        position: 'absolute',
        width: '14px',
        height: '14px',
        background: 'linear-gradient(135deg, #ffd700, #c9b037)',
        clipPath: 'polygon(0 0,100% 0,0 100%)',
        opacity: '.9'
      });
      if (pos === 'tl') { c.style.left='-1px'; c.style.top='-1px'; }
      if (pos === 'tr') { c.style.right='-1px'; c.style.top='-1px'; c.style.transform='scaleX(-1)'; }
      if (pos === 'bl') { c.style.left='-1px'; c.style.bottom='-1px'; c.style.transform='scaleY(-1)'; }
      if (pos === 'br') { c.style.right='-1px'; c.style.bottom='-1px'; c.style.transform='scale(-1)'; }
      panel.appendChild(c);
    });

    // Title (Cinzel)
    const title = document.createElement('div');
    title.textContent = 'Loading Save…';
    Object.assign(title.style, {
      fontFamily: '"Cinzel", serif',
      fontWeight: 600,
      fontSize: '24px',
      letterSpacing: '0.06em',
      marginBottom: '10px',
      textShadow: '0 1px 0 rgba(0,0,0,.6)',
    });

    // === Civ-style progress frieze (replaces spinner) ===
    const frieze = document.createElement('div');
    Object.assign(frieze.style, {
      position:'relative',
      width:'280px',
      height:'14px',
      margin:'8px auto 12px',
      background:'linear-gradient(180deg,#2c2416 0%, #3d3426 100%)',
      border:'1px solid #8b7355',
      borderRadius:'7px',
      overflow:'hidden'
    });

    // shimmering fill
    const fill = document.createElement('div');
    Object.assign(fill.style, {
      position:'absolute',
      inset:'2px',
      borderRadius:'5px',
      background:'linear-gradient(90deg, rgba(201,176,55,.15) 0%, rgba(255,215,0,.45) 50%, rgba(201,176,55,.15) 100%)',
      backgroundSize:'200% 100%',
      animation:'civ-fill 2.2s linear infinite'
    });
    frieze.appendChild(fill);

    // moving chevron marker
    const marker = document.createElement('div');
    Object.assign(marker.style, {
      position:'absolute',
      top:'-3px',
      left:'0',
      width:'0',
      height:'0',
      borderLeft:'6px solid transparent',
      borderRight:'6px solid transparent',
      borderTop:'8px solid #ffd700',
      filter:'drop-shadow(0 1px 0 rgba(0,0,0,.5))',
      animation:'civ-marker 1.8s ease-in-out infinite'
    });
    frieze.appendChild(marker);

    // Message (Crimson Text)
    msgEl = document.createElement('div');
    msgEl.textContent = 'Fetching game state…';
    Object.assign(msgEl.style, {
      fontFamily: '"Crimson Text", serif',
      fontWeight: 600,
      fontSize: '16px',
      letterSpacing: '.3px',
      opacity: .95,
    });

    // Gold shimmer rule
    const rule = document.createElement('div');
    Object.assign(rule.style, {
      height: '2px',
      margin: '12px auto 0',
      width: '160px',
      background: 'linear-gradient(90deg, transparent, #ffd700, transparent)',
      animation: 'civ-shimmer 1.8s linear infinite',
      opacity: .6,
    });

    // Keyframes (single style block)
    const style = document.createElement('style');
    style.textContent = `
      @keyframes civ-shimmer { 
        0% { filter: brightness(0.9); } 
        50% { filter: brightness(1.25); } 
        100% { filter: brightness(0.9); } 
      }
      @keyframes civ-fill { 
        0% { background-position: 0 0; } 
        100% { background-position: 200% 0; } 
      }
      @keyframes civ-marker { 
        0%   { transform: translateX(6px); } 
        50%  { transform: translateX(260px); } 
        100% { transform: translateX(6px); } 
      }
    `;

    panel.appendChild(title);
    panel.appendChild(frieze);
    panel.appendChild(msgEl);
    panel.appendChild(rule);
    el.appendChild(style);
    el.appendChild(panel);
    document.body.appendChild(el);
  }

  return {
    show(msg = 'Loading…') {
      ensure();
      msgEl.textContent = msg;
      el.style.display = 'flex';
      // Warm the fonts so the panel paints in-brand immediately
      Promise.allSettled([
        document.fonts.load('600 24px "Cinzel"'),
        document.fonts.load('600 16px "Crimson Text"')
      ]);
    },
    text(msg) {
      if (!el) ensure();
      msgEl.textContent = msg;
    },
    async hide() {
      if (!el) return;
      el.style.display = 'none';
      // Yield a frame to let the overlay disappear before heavy work resumes
      await new Promise(requestAnimationFrame);
    }
  };
})();


const jsonFolderEndpoint = "/list-saved-games";

function populateDropdown() {
  fetch(jsonFolderEndpoint)
    .then(res => res.json())
    .then(files => {
      const dropdown = document.getElementById("jsonFileDropdown");
      files.forEach(file => {
        const option = document.createElement("option");
        console.log(file);
        option.value = file;
        option.textContent = file.split('/').pop().split('.')[0];
        //option.textContent = file;
        dropdown.appendChild(option);
      });
    })
    .catch(err => console.error("Failed to populate dropdown:", err));
}

// True streaming decompression and parsing - never holds full file in memory
async function fetchNDJSONGzipStreaming(url) {
    const { Decompress, strFromU8 } = await import('https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js');
    
    const response = await fetch(url, { cache: 'no-cache' });
    const reader = response.body.getReader();
    
    // Streaming decompressor
    const decomp = new Decompress((chunk, final) => {
        // This callback receives decompressed chunks as they become available
        // We'll collect them but could also process line-by-line here
    });
    
    const gamestate = {};
    let meta = null;
    let buffer = '';
    const turnData = [];
    
    return new Promise(async (resolve, reject) => {
        // Set up the decompressor callback to process chunks
        decomp.ondata = (chunk, final) => {
            // Convert chunk to string and add to buffer
            buffer += strFromU8(chunk);
            
            // Process complete lines
            let lastNewline = buffer.lastIndexOf('\n');
            if (lastNewline !== -1) {
                const completeLines = buffer.substring(0, lastNewline);
                buffer = buffer.substring(lastNewline + 1);
                
                // Process each complete line immediately
                const lines = completeLines.split('\n');
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const obj = JSON.parse(line);
                        
                        if (obj.meta) {
                            meta = obj.meta;
                            // Pre-allocate arrays
                            for (const key of meta.step_keys) {
                                gamestate[key] = new Array(meta.num_steps);
                            }
                        } else if (typeof obj.turn === 'number') {
                            // Store turn data directly into the gamestate arrays
                            // This avoids keeping a separate turnData array
                            for (const key of meta.step_keys) {
                                if (key in obj) {
                                    gamestate[key][obj.turn] = obj[key];
                                }
                            }
                        } else if (obj.const) {
                            gamestate[obj.const] = obj.value;
                        }
                    } catch (e) {
                        console.error('Failed to parse line:', e, line.substring(0, 100));
                    }
                }
            }
            
            if (final) {
                // Process any remaining data in buffer
                if (buffer.trim()) {
                    try {
                        const obj = JSON.parse(buffer);
                        if (obj.const) {
                            gamestate[obj.const] = obj.value;
                        }
                    } catch (e) {
                        console.error('Failed to parse final buffer:', e);
                    }
                }
                resolve(gamestate);
            }
        };
        
        // Read and decompress in chunks
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    decomp.push(new Uint8Array(0), true);  // Signal end
                    break;
                }
                decomp.push(value);  // Push compressed chunk
            }
        } catch (err) {
            reject(err);
        }
    });
}

// Alternative: If streaming still fails, try this approach that processes in smaller batches
async function fetchNDJSONGzipBatched(url) {
    const { gunzipSync } = await import('https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js');
    
    // First, fetch just the compressed data
    const response = await fetch(url, { cache: 'no-cache' });
    const compressed = new Uint8Array(await response.arrayBuffer());
    
    // Decompress to a Uint8Array (more memory efficient than string)
    const decompressed = gunzipSync(compressed);
    
    // Process in chunks without converting entire array to string
    const gamestate = {};
    let meta = null;
    const decoder = new TextDecoder('utf-8', { stream: true });
    let buffer = '';
    const chunkSize = 1024 * 1024; // Process 1MB at a time
    
    for (let i = 0; i < decompressed.length; i += chunkSize) {
        const chunk = decompressed.subarray(i, Math.min(i + chunkSize, decompressed.length));
        buffer += decoder.decode(chunk, { stream: i + chunkSize < decompressed.length });
        
        // Process complete lines
        let lastNewline = buffer.lastIndexOf('\n');
        if (lastNewline !== -1) {
            const lines = buffer.substring(0, lastNewline).split('\n');
            buffer = buffer.substring(lastNewline + 1);
            
            for (const line of lines) {
                if (!line.trim()) continue;
                const obj = JSON.parse(line);
                
                if (obj.meta) {
                    meta = obj.meta;
                    for (const key of meta.step_keys) {
                        gamestate[key] = new Array(meta.num_steps);
                    }
                } else if (typeof obj.turn === 'number') {
                    for (const key in obj) {
                        if (key !== 'turn' && meta.step_keys.includes(key)) {
                            gamestate[key][obj.turn] = obj[key];
                        }
                    }
                } else if (obj.const) {
                    gamestate[obj.const] = obj.value;
                }
            }
        }
    }
    
    // Handle remaining buffer
    if (buffer.trim()) {
        const obj = JSON.parse(buffer);
        if (obj.const) {
            gamestate[obj.const] = obj.value;
        }
    }
    
    return gamestate;
}

// Your modified dropdown handler with both approaches
document.getElementById("jsonFileDropdown").addEventListener("change", async function () {
    const filename = this.value;
    
    try {
        Loader.show('Fetching game state…');
        
        let data;
        
        // Check if it's an NDJSON file
        if (filename.includes('.ndjson')) {
            Loader.text('Unpacking replay data…');
            
            try {
                // Try true streaming first
                data = await fetchNDJSONGzipStreaming(filename.replace(/\.ndjson$/, '.ndjson.gz'));
            } catch (streamErr) {
                console.warn('Streaming failed, trying batched approach:', streamErr);
                Loader.text('Loading replay data (batched)…');
                data = await fetchNDJSONGzipBatched(filename.replace(/\.ndjson$/, '.ndjson.gz'));
            }
        } else {
            // Your existing code for regular JSON files
            let gzUrl = filename;
            if (!/\.gzip$/i.test(filename)) {
                gzUrl = filename.replace(/\.json$/i, '.json.gzip');
            }
            
            try {
                Loader.text('Downloading compressed data…');
                data = await fetchGzipJSON(gzUrl);
            } catch (gzErr) {
                console.warn('[gzip] failed, falling back to plain JSON', gzErr);
                Loader.text('Downloading JSON…');
                const res = await fetch(filename, { cache: 'no-cache' });
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status} ${res.url}`);
                }
                Loader.text('Parsing JSON…');
                const txt = await res.text();
                data = JSON.parse(txt);
            }
        }
        
        // --- The rest of your code is EXACTLY THE SAME ---
        terrainMap = data.terrain;
        riverMap = data.rivers;
        lakeMap = data.lakes;
        elevationMap = data.elevation;
        featureMap = data.features;
        nwMap = data.nw;
        
        unitsType = data.units_type;
        unitsMilitary = data.units_military;
        unitsRowCol = data.units_rowcol;
        unitsTradePlayerTo = data.units_trade_player_to;
        unitsTradeCityTo = data.units_trade_city_to;
        unitsTradeCityFrom = data.units_trade_city_from;
        unitsTradeYield = data.units_trade_yields;
        unitsEngaged = data.units_engaged;
        unitsCombatBonus = data.combat_bonus_accel;
        
        cs_cities = data.cs_rowcols;
        csOwnership = data.cs_ownership;
        playerCities = data.player_rowcols;
        playerOwnership = data.player_ownership;
        csOwnershipBorders = data.cs_ownership_borders;
        playerOwnershipBorders = data.player_ownership_borders;
        allResourceMap = data.all_resource_map;
        gtYieldMap = data.gt_yield_map;
        playerYieldMap = data.player_yield_map;
        movementCostMap = data.movement_cost_map;
        numDelegates = data.num_delegates;
        ISREPLAY = true;
        playerTechs = data.techs;
        playerPolicies = data.policies;
        playerReligion = data.religious_tenets;
        workedSlots = data.player_worked_slots;
        playerYields = data.player_yields;
        
        playerPops = data.player_pops;
        playerBuildings = data.player_buildings;
        playerWonderAccel = data.wonder_accel;
        playerBldgAccel = data.bldg_accel;
        playerMilitaryBldgAccel = data.military_bldg_accel;
        playerReligionBldgAccel = data.religion_bldg_accel;
        playerCultureBldgAccel = data.culture_bldg_accel;
        playerSeaBldgAccel = data.sea_bldg_accel;
        playerScienceBldgAccel = data.science_bldg_accel;
        playerEconBldgAccel = data.econ_bldg_accel;
        playerCityReligion = data.city_religious_tenets;
        playerGWSlots = data.gw_slots;
        playerUnitAccel = data.unit_accel;
        playerYieldAccel = data.yield_accel;
        playerBorderAccel = data.border_accel;
        playerSpecialistSlots = data.specialist_slots;
        playerBldgMaintenance = data.bldg_maintenance;
        playerUnitXPAdd = data.unit_xp_add;
        playerCanTradeFood = data.can_trade_food;
        playerCanTradeProd = data.can_trade_prod;
        playerDefense = data.defense;
        playerHP = data.hp;
        tradeGoldAddOwner = data.trade_gold_add_owner;
        tradeGoldAddDest = data.trade_gold_add_dest;
        tradeLandDistMod = data.trade_land_dist_mod;
        tradeSeaDistMod = data.trade_sea_dist_mod;
        playerGPAccel = data.gp_accel;
        playerMountedAccel = data.mounted_accel;
        playerLandUnitAccel = data.land_unit_accel;
        playerTechStealReduce = data.tech_steal_reduce_accel;
        playerSeaUnitAccel = data.sea_unit_accel;
        playerGWTourismAccel = data.gw_tourism_accel;
        playerCultureToTourism = data.culture_to_tourism;
        playerAirUnitCapacity = data.air_unit_capacity;
        playerSpaceshipProdAccel = data.spaceship_prod_accel;
        playerNavalMovementAdd = data.naval_movement_add;
        playerNavalSightAdd = data.naval_sight_add;
        playerCityConnectionGoldAccel = data.city_connection_gold_accel;
        playerArmoredAccel = data.armored_accel;
        tourismTotal = data.tourism_total;
        cultureTotal = data.culture_total;
        improvementMap = data.improvement_map;
        roadMap = data.road_map;
        isConstructing = data.is_constructing;
        prodReserves = data.prod_reserves;
        csReligiousPopulation = data.cs_religious_population;
        csRelationships = data.cs_relationships;
        csInfluence = data.cs_influence;
        csType = data.cs_type;
        csQuest = data.cs_quest;
        csCultureTracker = data.cs_culture_tracker;
        csFaithTracker = data.cs_faith_tracker;
        csTechTracker = data.cs_tech_tracker;
        csTradeTracker = data.cs_trade_tracker;
        csReligionTracker = data.cs_religion_tracker;
        csWonderTracker = data.cs_wonder_tracker;
        csResourceTracker = data.cs_resource_tracker;
        
        playerReligiousPop = data.city_religious_pop;
        fow = data.fog_of_war;
        tradeLedger = data.trade_ledger;
        tradeLengthLedger = data.trade_length_ledger;
        tradeGPTAdj = data.trade_gpt_adj;
        tradeResourceAdj = data.trade_resource_adj;
        haveMet = data.have_met;
        atWar = data.at_war;
        unitHealth = data.unit_health;
        
        cityHP = data.hp;
        cityDefense = data.defense;
        hasSacked = data.has_sacked;
        treasury = data.treasury;
        happiness = data.happiness;
        resourcesOwned = data.resources_owned;
        gpps = data.gpps;
        gpThreshold = data.gpp_threshold;
        goldenAgeTurns = data.golden_age_turns;
        
        renderMapStatic();
        await Loader.hide();
        await new Promise(requestAnimationFrame);
        renderMapDynamic();
        
    } catch (err) {
        console.error("Failed to load gamestate:", err);
        alert("Failed to load game state. Check the file or console for details.");
        await Loader.hide();
    }
});

//document.getElementById("jsonFileDropdown").addEventListener("change", async function () {
//  const filename = this.value;
//  try {
//    Loader.show('Fetching game state…');
//    const response = await fetch(filename);
//    Loader.text('Parsing game data…');
//
//    const data = await response.json();
//
//    terrainMap = data.terrain;
//    riverMap = data.rivers;
//    lakeMap = data.lakes;
//    elevationMap = data.elevation;
//    featureMap = data.features;
//    nwMap = data.nw;
//
//    unitsType = data.units_type;
//    unitsMilitary = data.units_military;
//    unitsRowCol = data.units_rowcol;
//    unitsTradePlayerTo = data.units_trade_player_to;
//    unitsTradeCityTo = data.units_trade_city_to;
//    unitsTradeCityFrom = data.units_trade_city_from;
//    unitsTradeYield = data.units_trade_yields;
//    unitsEngaged = data.units_engaged;
//    unitsCombatBonus = data.combat_bonus_accel;
//
//    cs_cities = data.cs_rowcols;
//    csOwnership = data.cs_ownership;
//    playerCities = data.player_rowcols;
//    playerOwnership = data.player_ownership;
//    csOwnershipBorders = data.cs_ownership_borders;
//    playerOwnershipBorders = data.player_ownership_borders;
//    allResourceMap = data.all_resource_map;
//    gtYieldMap = data.gt_yield_map;
//    playerYieldMap = data.player_yield_map;
//    movementCostMap = data.movement_cost_map;
//    numDelegates = data.num_delegates;
//    ISREPLAY = true;
//    playerTechs = data.techs;
//    playerPolicies = data.policies;
//    playerReligion = data.religious_tenets;
//    workedSlots = data.player_worked_slots;
//    playerYields = data.player_yields;
//
//    playerPops = data.player_pops;
//    playerBuildings = data.player_buildings;
//    playerWonderAccel = data.wonder_accel;
//    playerBldgAccel = data.bldg_accel;
//    playerMilitaryBldgAccel = data.military_bldg_accel;
//    playerReligionBldgAccel = data.religion_bldg_accel;
//    playerCultureBldgAccel = data.culture_bldg_accel;
//    playerSeaBldgAccel = data.sea_bldg_accel;
//    playerScienceBldgAccel = data.science_bldg_accel;
//    playerEconBldgAccel = data.econ_bldg_accel;
//    playerCityReligion = data.city_religious_tenets;
//    playerGWSlots = data.gw_slots;
//    playerUnitAccel = data.unit_accel;
//    playerYieldAccel = data.yield_accel;
//    playerBorderAccel = data.border_accel;
//    playerSpecialistSlots = data.specialist_slots;
//    playerBldgMaintenance = data.bldg_maintenance;
//    playerUnitXPAdd = data.unit_xp_add;
//    playerCanTradeFood = data.can_trade_food;
//    playerCanTradeProd = data.can_trade_prod;
//    playerDefense = data.defense;
//    playerHP = data.hp;
//    tradeGoldAddOwner = data.trade_gold_add_owner;
//    tradeGoldAddDest = data.trade_gold_add_dest;
//    tradeLandDistMod = data.trade_land_dist_mod;
//    tradeSeaDistMod = data.trade_sea_dist_mod;
//    playerGPAccel = data.gp_accel;
//    playerMountedAccel = data.mounted_accel;
//    playerLandUnitAccel = data.land_unit_accel;
//    playerTechStealReduce = data.tech_steal_reduce_accel;
//    playerSeaUnitAccel = data.sea_unit_accel;
//    playerGWTourismAccel = data.gw_tourism_accel;
//    playerCultureToTourism = data.culture_to_tourism;
//    playerAirUnitCapacity = data.air_unit_capacity;
//    playerSpaceshipProdAccel = data.spaceship_prod_accel;
//    playerNavalMovementAdd = data.naval_movement_add;
//    playerNavalSightAdd = data.naval_sight_add;
//    playerCityConnectionGoldAccel = data.city_connection_gold_accel;
//    playerArmoredAccel = data.armored_accel;
//    tourismTotal = data.tourism_total;
//    cultureTotal = data.culture_total;
//    improvementMap = data.improvement_map;
//    roadMap = data.road_map;
//    isConstructing = data.is_constructing;
//    prodReserves = data.prod_reserves;
//    csReligiousPopulation = data.cs_religious_population;
//    csRelationships = data.cs_relationships;
//    csInfluence = data.cs_influence;
//    csType = data.cs_type;
//    csQuest = data.cs_quest;
//    csCultureTracker = data.cs_culture_tracker;
//    csFaithTracker = data.cs_faith_tracker;
//    csTechTracker = data.cs_tech_tracker;
//    csTradeTracker = data.cs_trade_tracker;
//    csReligionTracker = data.cs_religion_tracker;
//    csWonderTracker = data.cs_wonder_tracker;
//    csResourceTracker = data.cs_resource_tracker;
//
//    playerReligiousPop = data.city_religious_pop;
//    fow = data.fog_of_war;
//    tradeLedger = data.trade_ledger;
//    tradeLengthLedger = data.trade_length_ledger;
//    tradeGPTAdj = data.trade_gpt_adj;
//    tradeResourceAdj = data.trade_resource_adj;
//    haveMet = data.have_met;
//    atWar = data.at_war;
//    unitHealth = data.unit_health;
//
//    cityHP = data.hp;
//    cityDefense = data.defense;
//    hasSacked = data.has_sacked;
//    treasury = data.treasury;
//    happiness = data.happiness;
//    resourcesOwned = data.resources_owned;
//    gpps = data.gpps;
//    gpThreshold = data.gpp_threshold;
//    goldenAgeTurns = data.golden_age_turns;
//
//    // Heavy work
//    renderMapStatic();
//    // Hide BEFORE heavy rendering, and yield a frame so the overlay disappears
//  // IMPORTANT: wait for the FOW animation to play before heavy rendering
//  await Loader.hide();                        // runs the .45s animation
//  await new Promise(requestAnimationFrame);   // give the browser a paint
//    renderMapDynamic();
//
//  } catch (err) {
//    console.error("Failed to load gamestate:", err);
//    // show a quick error toast
//    alert("Failed to load game state. Check the file or console for details.");
//    await Loader.hide();
//  }
//});
//document.getElementById("jsonFileDropdown").addEventListener("change", function () {
//  const filename = this.value;
//  fetch(filename)
//    .then(response => response.json())
//    .then(data => {
//      terrainMap = data.terrain;
//      riverMap = data.rivers;
//      lakeMap = data.lakes;
//      elevationMap = data.elevation;
//      featureMap = data.features;
//      nwMap = data.nw;
//
//      unitsType = data.units_type;
//      unitsMilitary = data.units_military;
//      unitsRowCol = data.units_rowcol;
//      unitsTradePlayerTo = data.units_trade_player_to;
//      unitsTradeCityTo = data.units_trade_city_to;
//      unitsTradeCityFrom = data.units_trade_city_from;
//      unitsTradeYield = data.units_trade_yields;
//      unitsEngaged = data.units_engaged;
//      unitsCombatBonus = data.combat_bonus_accel;
//
//      //settlers = data.settler_rowcols;
//      cs_cities = data.cs_rowcols;
//      csOwnership = data.cs_ownership;
//      playerCities = data.player_rowcols;
//      playerOwnership = data.player_ownership;
//      csOwnershipBorders = data.cs_ownership_borders;
//      playerOwnershipBorders = data.player_ownership_borders;
//      allResourceMap = data.all_resource_map;
//      gtYieldMap = data.gt_yield_map;
//      playerYieldMap = data.player_yield_map;
//      movementCostMap = data.movement_cost_map;
//      numDelegates = data.num_delegates;
//      //canMoveTo = data.can_move_to;
//      //ISREPLAY = (countNestedArrays(terrainMap) > 43);
//      ISREPLAY = true;
//      playerTechs = data.techs;
//      playerPolicies = data.policies;
//      playerReligion = data.religious_tenets;
//      workedSlots = data.player_worked_slots;
//      playerYields = data.player_yields;
//
//      playerPops = data.player_pops;
//      playerBuildings = data.player_buildings;
//      playerWonderAccel = data.wonder_accel;
//      playerBldgAccel = data.bldg_accel;
//      playerMilitaryBldgAccel = data.military_bldg_accel;
//      playerReligionBldgAccel = data.religion_bldg_accel;
//      playerCultureBldgAccel = data.culture_bldg_accel;
//      playerSeaBldgAccel = data.sea_bldg_accel;
//      playerScienceBldgAccel = data.science_bldg_accel;
//      playerEconBldgAccel = data.econ_bldg_accel;
//      playerCityReligion = data.city_religious_tenets;
//      playerGWSlots = data.gw_slots;
//      playerUnitAccel = data.unit_accel;
//      //playerGoldAccel = data.gold_accel;
//      //playerTourismAccel = data.tourism_accel;
//      //playerScienceAccel = data.science_accel;
//      //playerGrowthAccel = data.growth_accel;
//      playerYieldAccel = data.yield_accel;
//      playerBorderAccel = data.border_accel;
//      playerSpecialistSlots = data.specialist_slots;
//      playerBldgMaintenance = data.bldg_maintenance;
//      playerUnitXPAdd = data.unit_xp_add;
//      playerCanTradeFood = data.can_trade_food;
//      playerCanTradeProd = data.can_trade_prod;
//      playerDefense = data.defense;
//      playerHP = data.hp;
//      tradeGoldAddOwner = data.trade_gold_add_owner;
//      tradeGoldAddDest = data.trade_gold_add_dest;
//      tradeLandDistMod = data.trade_land_dist_mod;
//      tradeSeaDistMod = data.trade_sea_dist_mod;
//      playerGPAccel = data.gp_accel;
//      playerMountedAccel = data.mounted_accel;
//      playerLandUnitAccel = data.land_unit_accel;
//      playerTechStealReduce = data.tech_steal_reduce_accel;
//      playerSeaUnitAccel = data.sea_unit_accel;
//      playerGWTourismAccel = data.gw_tourism_accel;
//      playerCultureToTourism = data.culture_to_tourism;
//      playerAirUnitCapacity = data.air_unit_capacity;
//      playerSpaceshipProdAccel = data.spaceship_prod_accel;
//      playerNavalMovementAdd = data.naval_movement_add;
//      playerNavalSightAdd = data.naval_sight_add;
//      playerCityConnectionGoldAccel = data.city_connection_gold_accel;
//      playerArmoredAccel = data.armored_accel;
//      tourismTotal = data.tourism_total;
//      cultureTotal = data.culture_total;
//      improvementMap = data.improvement_map;
//      roadMap = data.road_map;
//      isConstructing = data.is_constructing;
//      prodReserves = data.prod_reserves;
//      csReligiousPopulation = data.cs_religious_population;
//      csRelationships = data.cs_relationships;
//      csInfluence = data.cs_influence;
//      csType = data.cs_type;
//      csQuest = data.cs_quest;
//      csCultureTracker = data.cs_culture_tracker;
//      csFaithTracker = data.cs_faith_tracker;
//      csTechTracker = data.cs_tech_tracker;
//      csTradeTracker = data.cs_trade_tracker;
//      csReligionTracker = data.cs_religion_tracker;
//      csWonderTracker = data.cs_wonder_tracker;
//      csResourceTracker = data.cs_resource_tracker;
//      
//      playerReligiousPop = data.city_religious_pop,
//      fow = data.fog_of_war;
//      tradeLedger = data.trade_ledger;
//      tradeLengthLedger = data.trade_length_ledger;
//      tradeGPTAdj = data.trade_gpt_adj;
//      tradeResourceAdj = data.trade_resource_adj;
//      haveMet = data.have_met;
//      atWar = data.at_war;
//      unitHealth = data.unit_health;
//
//      cityHP = data.hp;
//      cityDefense = data.defense;
//      hasSacked = data.has_sacked;
//      treasury = data.treasury;
//      happiness = data.happiness;
//      resourcesOwned = data.resources_owned;
//      gpps = data.gpps;
//      gpThreshold = data.gpp_threshold;
//      goldenAgeTurns = data.golden_age_turns;
//      //console.log(playerPops);
//      //console.log("playerCities: ", playerOwnership);
//      //console.log("ISREPLAY check: ", ISREPLAY);
//      //console.log("TERRAINMAP: ", terrainMap);
//      /*
//      if (ISREPLAY) {
//        NUMTURNS = terrainMap.length;
//      } else {
//        NUMTURNS = 1;
//      }
//      */
//      renderMapStatic();
//      renderMapDynamic();
//    })
//    .catch(err => console.error("Failed to load gamestate:", err));
//});

populateDropdown();

// Camera intrinsics
let dragging  = false;
let dragStart = new PIXI.Point();
let worldStart= new PIXI.Point();

app.stage.interactive = true;           // listen on the whole canvas

app.stage
  .on('pointerdown', (e) => {
    dragging   = true;
    dragStart.copyFrom(e.data.global);
    worldStart.set(world.x, world.y);
  })
  .on('pointermove', (e) => {
    if (!dragging) return;
    const now = e.data.global;
    world.position.set(
      worldStart.x + (now.x - dragStart.x),
      worldStart.y + (now.y - dragStart.y)
    );
  })
  .on('pointerup', () => dragging = false)
  .on('pointerupoutside', () => dragging = false);

app.canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  
  if (viewState.cityViewEnabled && isOverBuildPanel(ev.offsetX, ev.offsetY)) {
    const lineHeight = 18;
    const direction = Math.sign(ev.deltaY);   // -1 for up, +1 for down
    viewState.buildPanelScroll += direction * lineHeight;
    viewState.buildPanelScroll = Math.max(viewState.buildPanelScroll, 0); // clamp to topalen(ev.offsetX, ev.offsetY)) {
    drawCityBuildingPanelPixi(viewState.turn, viewState.selectedCityPlayer, viewState.selectedCityNum, viewState.buildPanelScroll);
  } 
  else {
    /* 1️⃣  mouse position in stage coords */
    const mouseStage = new PIXI.Point(ev.offsetX, ev.offsetY);

    /* 2️⃣  same point expressed in current world-local coords */
    const mouseWorld = world.toLocal(mouseStage);

    /* 3️⃣  choose zoom factor */
    const factor   = ev.deltaY > 0 ? 0.9 : 1.1;
    const newScale = world.scale.x * factor;
    if (newScale < 0.3 || newScale > 4) return;

    /* 4️⃣  apply new scale */
    world.scale.set(newScale);

    /* 5️⃣  keep the map pixel under the cursor fixed on screen
            — *pivot* must be subtracted before scaling  */
    world.position.set(
      mouseStage.x - (mouseWorld.x - world.pivot.x) * newScale,
      mouseStage.y - (mouseWorld.y - world.pivot.y) * newScale
    );
  }
});


// Toggle states
// Game-step scrolling logic
let currentTurn = 0;
const maxTurn = 999;  // optionally replace with dynamic upper limit

function updateTurn(newTurn) {
  currentTurn = Math.max(0, Math.min(newTurn, maxTurn));
  document.getElementById("turnInput").value = currentTurn;
  viewState.turn = currentTurn;
  renderMapDynamic(getLayerToggleStates());  // <- pass currentTurn to your rendering logic
}

// Click left arrow
document.getElementById("prevTurn").addEventListener("click", () => {
  updateTurn(currentTurn - 1);
});

// Click right arrow
document.getElementById("nextTurn").addEventListener("click", () => {
  updateTurn(currentTurn + 1);
});

// Manual input + Enter key
document.getElementById("turnInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const val = parseInt(e.target.value);
    if (!isNaN(val)) updateTurn(val);
  }
});


function getLayerToggleStates() {
  return {
    showYieldsBool: document.getElementById('toggle-yields').checked,
    showResourcesBool: document.getElementById('toggle-resources').checked,
    showImprovementsBool: document.getElementById('toggle-improvements').checked,
    turn: currentTurn,
    showTechTreeBool: document.getElementById("tech-tree-button").getAttribute('aria-pressed') === 'true',
    yieldTypeIdx: document.getElementById("yieldTypeDropdown").value,
    playerIdx: document.getElementById("playerDropdown").value,
    showOwnershipBool: document.getElementById("toggle-ownership").checked,
    showFOWBool: document.getElementById("toggle-fow").checked,
    fowPlayerIdx: document.getElementById("fowDropdown").value,
  };
}

// Buttons on the top-left corner
/* click = flip state */
document.getElementById('tech-tree-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('tech-tree-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('tech-tree-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

document.getElementById('social-policy-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('social-policy-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('social-policy-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

document.getElementById('religion-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('religion-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('religion-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

document.getElementById('victory-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('victory-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('victory-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

document.getElementById('trade-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('trade-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('trade-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

document.getElementById('demos-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('demos-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('demos-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
});

function setupLayerControls() {
  const checkboxes = document.querySelectorAll('#toggle-yields, #toggle-resources, #toggle-improvements, #toggle-ownership, #toggle-fow');
  checkboxes.forEach((checkbox) => {
    checkbox.addEventListener('change', () => {
      const layerStates = getLayerToggleStates();
      renderMapDynamic(layerStates);
    });
  });

  const dropdown = document.getElementById("yieldTypeDropdown");
  dropdown.addEventListener('change', () => {
    const layerStates = getLayerToggleStates();
    renderMapDynamic(layerStates);  // live update
  });

  const fowDropdown = document.getElementById("fowDropdown");
  fowDropdown.addEventListener('change', () => {
    const layerStates = getLayerToggleStates();
    renderMapDynamic(layerStates);  // live update
  });

}

// Call this once when page loads
setupLayerControls();

// Util functions
function drawHex(cx, cy, colorHex, texture = null, borderColor = 0x000000) {
  const container = new PIXI.Container();
  container.x = cx;
  container.y = cy;

  const r            = hexRadius - 0.5;
  const angleOffset  = Math.PI / 6;
  const dTheta       = Math.PI / 3;

  /* helper: trace a hex path on a Graphics object */
  function traceHexPath(g) {
    g.moveTo(r * Math.cos(angleOffset), r * Math.sin(angleOffset));
    for (let i = 1; i <= 6; i++) {
      const a = angleOffset + i * dTheta;
      g.lineTo(r * Math.cos(a), r * Math.sin(a));
    }
    g.closePath();
  }

  if (texture) {
    /* ------------ 1️⃣ create sprite ------------ */
    const sprite = new PIXI.Sprite(texture);
    sprite.anchor.set(0.5);
    container.addChild(sprite);

    /* ------------ 2️⃣ create mask (fill only) --- */
    // This mask is important, as it clips the textures to the correct
    // hexagonal shape. 
    const mask = new PIXI.Graphics();
    mask.beginFill(0x000000);
    traceHexPath(mask);
    mask.endFill();
    //mask.visible = false;    // hide mask pixels but keep geometry
    container.addChild(mask);
    sprite.mask = mask;

    /* ------------ 3️⃣ border outline on top ----- */
    const border = new PIXI.Graphics();
    border.lineStyle(1, borderColor, 0.6);
    traceHexPath(border);    // no fill, just stroke
    container.addChild(border);   // added last → rendered on top

  } else {
    /* ---------- solid-colour fallback ---------- */
    const fill = typeof colorHex === 'string'
      ? parseInt(colorHex.slice(1), 16)
      : (colorHex ?? 0x000000);

    const hex = new PIXI.Graphics();
    hex.beginFill(fill);
    hex.lineStyle(1, borderColor, 1.0);
    traceHexPath(hex);
    hex.endFill();
    container.addChild(hex);
  }

  return container;
}

function drawHexOwnership(cx, cy, colorHex, borderColor = 0x000000) {
  const container = new PIXI.Container();
  container.x = cx;
  container.y = cy;

  const r            = hexRadius - 0.5;
  const angleOffset  = Math.PI / 6;
  const dTheta       = Math.PI / 3;

  /* helper: trace a hex path on a Graphics object */
  function traceHexPath(g) {
    g.moveTo(r * Math.cos(angleOffset), r * Math.sin(angleOffset));
    for (let i = 1; i <= 6; i++) {
      const a = angleOffset + i * dTheta;
      g.lineTo(r * Math.cos(a), r * Math.sin(a));
    }
    g.closePath();
  }

  /* ---------- solid-colour fallback ---------- */
  const fill = typeof colorHex === 'string'
    ? parseInt(colorHex.slice(1), 16)
    : (colorHex ?? 0x000000);

  const hex = new PIXI.Graphics();
  hex.beginFill(fill, 0.5);
  hex.lineStyle(1, borderColor, 1.0);
  traceHexPath(hex);
  hex.endFill();
  container.addChild(hex);

  return container;
}

function drawHexOwnershipv2(cx, cy, colorHex, borderColor = 0x000000, bgColor = 0xFFFFFF, stripeWidth = 5, stripeSpacing = 16, bgAlpha = 0.0) {
  const container = new PIXI.Container();
  container.x = cx;
  container.y = cy;

  const r            = hexRadius - 0.5;
  const angleOffset  = Math.PI / 6;
  const dTheta       = Math.PI / 3;

  /* helper: trace a hex path on a Graphics object */
  function traceHexPath(g) {
    g.moveTo(r * Math.cos(angleOffset), r * Math.sin(angleOffset));
    for (let i = 1; i <= 6; i++) {
      const a = angleOffset + i * dTheta;
      g.lineTo(r * Math.cos(a), r * Math.sin(a));
    }
    g.closePath();
  }

  // Parse the stripe color (main color argument)
  const stripeColor = typeof colorHex === 'string'
    ? parseInt(colorHex.slice(1), 16)
    : (colorHex ?? 0x000000);

  /* ---------- Draw background hexagon ---------- */
  const bg = new PIXI.Graphics();
  bg.beginFill(bgColor, bgAlpha);
  traceHexPath(bg);
  bg.endFill();
  container.addChild(bg);

  /* ---------- Draw stripes with clipping ---------- */
  const stripeGraphics = new PIXI.Graphics();
  
  // Set up clipping mask by drawing the hex shape first
  stripeGraphics.beginFill(0x000000, 0); // Invisible fill
  traceHexPath(stripeGraphics);
  stripeGraphics.endFill();
  
  // Now draw the stripes
  stripeGraphics.lineStyle(stripeWidth, stripeColor, 1.0);
  
  // Calculate bounds for stripes
  const bounds = r * 2;
  const numStripes = Math.ceil(bounds * 2 / stripeSpacing);
  
  // Create a temporary mask shape
  const maskShape = new PIXI.Graphics();
  maskShape.beginFill(0xFFFFFF);
  traceHexPath(maskShape);
  maskShape.endFill();
  
  // Draw stripes and apply mask
  for (let i = 0; i < numStripes; i++) {
    const offset = i * stripeSpacing - bounds;
    stripeGraphics.moveTo(offset - bounds, -bounds);
    stripeGraphics.lineTo(offset + bounds, bounds);
  }
  stripeGraphics.stroke();
  
  stripeGraphics.mask = maskShape;
  container.addChild(maskShape); // Add mask to container
  container.addChild(stripeGraphics);

  /* ---------- Draw border ---------- */
  //const border = new PIXI.Graphics();
  //border.lineStyle(1, borderColor, 1.0);
  //traceHexPath(border);
  //border.stroke();
  //container.addChild(border);

  return container;
}
function addElevationSprite(cx, cy, elevID, textures) {
  const cfg  = constants.elevationConfigs[elevID];
  const tex  = textures[elevID];
  if (!cfg || !tex) return;                   // ocean/flat → nothing

  const spr  = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(cfg.scale);
  spr.x = cx;
  spr.y = cy + cfg.yOff;                      // lift mountains a bit
  elevationLayer.addChild(spr);
}


function addFeature(id, onHill, cx, cy, texTable) {
  const def = constants.featureDefs[id];
  if (!def) return;                    // id 0 or unknown

  const variant = onHill && def.hill ? 'hill' : 'normal';
  const tex     = texTable[id]?.[variant];
  const conf    = def[variant];

  if (!tex || !conf) return;

  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(conf.scale);
  spr.x = cx;
  spr.y = cy + conf.yOff;
  featureLayer.addChild(spr);
}

function addNaturalWonder(id, x, y) {
  const def = constants.nwDefs[id];
  const tex = nwTextures[id];
  if (!def || !tex) return;

  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.scale);
  spr.x = x;
  spr.y = y + def.yOff;
  nwLayer.addChild(spr);
}

// angles for the 6 corners (pointy-topped)
const CORNER_ANGLE = Math.PI / 6;        // 30°, start
const D_THETA      = Math.PI / 3;        // 60°


// drawHex starts at  +30°  (pointy top, first corner upper-right)
const CORNER_DEG = [ 330, 30,  90, 150, 210, 270 ];   // ← rotated list
function cornerOffset(idx) {
  const rad = CORNER_DEG[idx] * Math.PI / 180;
  return [ Math.cos(rad) * hexRadius * 1.025,
           Math.sin(rad) * hexRadius * 1.025 ];
}

const EDGE_CORNER = [
  [3, 4],  // 0  north (left side)
  [4, 5],  // 1  northeast  (top-left)
  [3, 2],  // 2  southeast  (bottom-left)
  [0, 1],  // 3  south (right side)
  [1, 2],  // 4  southwest  (bottom-right)
  [0, 5]   // 5  northwest  (top-right)
];

function traceRiverEdge(g, cx, cy, edgeIdx,
                        width = 6, extend = 0.85,
                        color = 0x4292c6, alpha = 0.9)
{
  const [cA, cB]        = EDGE_CORNER[edgeIdx];
  const [ax0, ay0]      = cornerOffset(cA);
  const [bx0, by0]      = cornerOffset(cB);

  let ax = cx + ax0, ay = cy + ay0;
  let bx = cx + bx0, by = cy + by0;

  const dx  = bx - ax,  dy = by - ay;
  const len = Math.hypot(dx, dy);
  const nx  = dx / len, ny = dy / len;

  ax -= nx * extend;     ay -= ny * extend;
  bx += nx * extend;     by += ny * extend;

  g.setStrokeStyle({ width, color, alpha, cap: 'round' });
  g.moveTo(ax, ay);
  g.lineTo(bx, by);
  g.stroke();
}

function hexStringToNumber(str) {
  return new PIXI.Color(str).toNumber();   // "#FF2A2F" → 0xff2a2f
}

function addBorderEdge(g, cx, cy, edgeIdx, color,
                       width = 4.4, extend = 0.1)
{
  const [cA, cB]   = EDGE_CORNER[edgeIdx];
  let [ax, ay]     = cornerOffset(cA);
  let [bx, by]     = cornerOffset(cB);
  ax += cx; ay += cy;
  bx += cx; by += cy;

  // extend a smidge beyond the hex
  const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy);
  const nx = dx / len, ny = dy / len;
  ax -= nx * extend; ay -= ny * extend;
  bx += nx * extend; by += ny * extend;

  g.setStrokeStyle({ width, color, cap: 'round', alpha: 1 });
  g.moveTo(ax, ay);
  g.lineTo(bx, by);
  g.stroke();
}

function addResource(id, cx, cy) {
  const def = constants.resourceDefs[id];
  const tex = resTextures[id];
  if (!def || !tex) return;               // id 0 or missing

  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.s);
  spr.x = cx + 22;                        // shift right like old code
  spr.y = cy;
  resourceLayer.addChild(spr);
}

function drawImprovements(id, cx, cy) {
  const def = constants.improvementDefs[id];
  const tex = impTextures[id];
  if (id === 0 || !tex) return;           // nothing to draw

  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.s);

  // Improvement icons go on the LEFT side of the hex
  spr.x = cx - 22;
  spr.y = cy;

  improvementsLayer.addChild(spr);
}

function drawRoads(cx, cy) {
  const def = constants.improvementDefs[11];
  const tex = impTextures[11];
  
  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.s);

  // Improvement icons go on the LEFT side of the hex
  spr.x = cx - 10;
  spr.y = cy + 22;

  improvementsLayer.addChild(spr);

}


const offsets = [
  [0],                       // 1 icon
  [-7, 7],                   // 2 icons
  [-13, 0, 13],              // 3
  [-19, -6, 6, 19],          // 4  (single row)
  [-13, 0, 13, -7, 7],       // 5  (row1:0-2, row2:3-4)
  [-13, 0, 13, -13, 0, 13],  // 6  (row1:0-2, row2:3-5)
];

function countNonZero(v) { return v.reduce((c,x)=>c+(x!==0),0); }
function nonZeroIdx(v)   { return v.flatMap((x,i)=>x?i:[]);     }

function drawYield(cx, cy, yieldVec) {
  // Clip to first 6 entries (food…science)
  yieldVec = yieldVec.slice(0, 6);

  const count = countNonZero(yieldVec);
  if (count === 0) return;

  const kinds = nonZeroIdx(yieldVec);
  const off   = offsets[count-1];

  /*  create a container so the whole block moves together */
  const group = new PIXI.Container();
  group.x = cx;
  group.y = cy;
  yieldLayer.addChild(group);

  for (let i = 0; i < count; i++) {
    const kind   = kinds[i];                 // 0–5
    const amt    = yieldVec[kind];           // 1–N
    const capped = Math.min(amt, 5);
    const id     = kind*5 + (capped-1) + 1;  // translate to defs index
    const tex    = yieldTextures[id];
    const def    = constants.yieldDefs[id];
    if (!def || !tex) continue;

    /* icon sprite */
    const spr = new PIXI.Sprite(tex);
    spr.anchor.set(0.5);
    spr.scale.set(def.s);

    /* position */
    const dx = off[i];
    const dy = (count >= 5 && i >= 3) ?  7
             : (count >= 5 && i <  3) ? -7
             : 0;
    spr.x = dx;
    spr.y = dy;
    group.addChild(spr);

    /* 5+ badge */
    if (amt > 5) {
      const circ = new PIXI.Sprite(yieldTextures['over5']);
      circ.anchor.set(0.5);
      circ.scale.set(constants.yieldOver5.s);
      circ.x = dx + 3;
      circ.y = dy + 3;
      group.addChild(circ);

      const txt = new PIXI.Text({
        text : String(amt),          // "6", "7", …
        style: {
          //fontFamily: 'sans-serif',
          fontSize  : 8,             // crisper / larger
          //fontWeight: 'bold',
          fill      : 0xffffff,      // solid white
          //align     : 'center',
          //strokeThickness: 0,
          //resolution: 200              // high-DPI so small fonts stay sharp
        }
      });
      //txt.style.fontFamily = 'sans-serif'; 

      txt.anchor.set(0.5);
      txt.x = dx + 5.65;
      txt.y = dy + 6;
      group.addChild(txt);
    }
  }
}


function drawFOW(cx, cy, fowValue) {
  const container = new PIXI.Container();
  container.x = cx;
  container.y = cy;

  const r            = hexRadius - 0.5;
  const angleOffset  = Math.PI / 6;
  const dTheta       = Math.PI / 3;

  /* helper: trace a hex path on a Graphics object */
  function traceHexPath(g) {
    g.moveTo(r * Math.cos(angleOffset), r * Math.sin(angleOffset));
    for (let i = 1; i <= 6; i++) {
      const a = angleOffset + i * dTheta;
      g.lineTo(r * Math.cos(a), r * Math.sin(a));
    }
    g.closePath();
  }

  if (fowValue == 1) {
    const fill = 0x000000

    const hex = new PIXI.Graphics();
    hex.beginFill(fill,  0.6);
    hex.lineStyle(1, fill, 1.0);
    traceHexPath(hex);
    hex.endFill();
    container.addChild(hex);
  } else if (fowValue == 2) {
    const fill = 0x000000;

    const hex = new PIXI.Graphics();
    hex.beginFill(fill);
    hex.lineStyle(1, fill, 1.0);
    traceHexPath(hex);
    hex.endFill();
    container.addChild(hex);
  }
  fowLayer.addChild(container);

}


function drawUnitBadge(civIdx, isMilitary, cx, cy) {
  // shapeIndex 0 = triangle (civilian), 1 = circle (military)
  const badgeID  = isMilitary * 6 + civIdx + 1;   // 1‒12
  const tex      = unitBGTextures[badgeID];
  const def      = constants.unitBGDefs[badgeID];
  //console.log("BADGE: ", civIdx, isMilitary, badgeID,  tex, def);
  if (!tex || !def) return;

  const spr  = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.s);
  spr.x = cx;
  spr.y = isMilitary ? cy - 26 : cy + 26;   // match old offsets
  unitsLayer.addChild(spr);
}

function drawUnitIcon(unitType, isMilitary, cx, cy) {
  if (unitType === 0) return;               // no unit

  const tex = unitTextures[unitType];       // 1 settler, 2 warrior, 3 worker
  const def = constants.unitDefs[unitType];
  if (!tex || !def) return;

  const spr = new PIXI.Sprite(tex);
  spr.anchor.set(0.5);
  spr.scale.set(def.s);
  spr.x = cx;
  spr.y = isMilitary ? cy - 26 : cy + 24;   // match old offsets
  unitsLayer.addChild(spr);
}

function drawUnitHealth(health, cx, cy) {
  // Create container for the health bar
  const healthBar = new PIXI.Container();
  
  // Health bar dimensions
  const barWidth = 3;
  const barHeight = 15;
  const borderThickness = 1;
  
  // Position to the right of the unit
  const offsetX = hexRadius * 0.28; // Adjust based on unit sprite width
  const offsetY = -hexRadius * 0.825; // Center vertically on unit
  
  // Create background/border
  const border = new PIXI.Graphics();
  border.beginFill(0x000000);
  border.drawRect(
    cx + offsetX - borderThickness,
    cy + offsetY - borderThickness,
    barWidth + borderThickness * 2,
    barHeight + borderThickness * 2
  );
  border.endFill();
  
  // Create background fill (dark red)
  const background = new PIXI.Graphics();
  background.beginFill(0x4a0000);
  background.drawRect(
    cx + offsetX,
    cy + offsetY,
    barWidth,
    barHeight
  );
  background.endFill();
  
  // Create health fill (fills from bottom, depletes from top)
  const healthFill = new PIXI.Graphics();
  
  // Color based on health percentage
  let fillColor;
  if (health > 0.6) {
    fillColor = 0x00ff00; // Green
  } else if (health > 0.3) {
    fillColor = 0xffff00; // Yellow
  } else {
    fillColor = 0xff0000; // Red
  }
  
  const depletedHeight = barHeight * (1 - health);
  
  healthFill.beginFill(fillColor);
  healthFill.drawRect(
    cx + offsetX,
    cy + offsetY + depletedHeight, // Start drawing from depleted point
    barWidth,
    barHeight * health // Height based on health percentage
  );
  healthFill.endFill();
  
  // Add components to container
  healthBar.addChild(border);
  healthBar.addChild(background);
  healthBar.addChild(healthFill);
  
  unitsLayer.addChild(healthBar);
}

function drawUnits(civIdx, unitType, isMilitary, health, cx, cy) {
  if (unitType === 0) return;               // nothing to draw
  drawUnitBadge(civIdx, isMilitary, cx, cy);
  drawUnitIcon(unitType, isMilitary, cx, cy);
  if (health < 1) {
    drawUnitHealth(health, cx, cy);
  } 
}

function drawCityHealth(cx, cy, turn, playerID, cityID, health) {
  const maxHPView = globalMapState.cityMaxHPLookup.get(turn);
  const maxHP = globalMapState.cityMaxHPLookup.toMatrix(maxHPView);
  const maxHPForMe = Math.max(maxHP[playerID][cityID], 2);

  //console.log(turn, playerID, cityID, health, maxHPForMe);

  const healthBar = new PIXI.Container();
  
  // Health bar dimensions
  const barWidth = 3;
  const barHeight = 15;
  const borderThickness = 1;
  
  // Position to the right of the unit
  const offsetX = hexRadius * 0.28; // Adjust based on unit sprite width
  const offsetY = -10; // Center vertically on unit
  
  // Create background/border
  const border = new PIXI.Graphics();
  border.beginFill(0x000000);
  border.drawRect(
    cx + offsetX - borderThickness,
    cy + offsetY - borderThickness,
    barWidth + borderThickness * 2,
    barHeight + borderThickness * 2
  );
  border.endFill();
  
  // Create background fill (dark red)
  const background = new PIXI.Graphics();
  background.beginFill(0x4a0000);
  background.drawRect(
    cx + offsetX,
    cy + offsetY,
    barWidth,
    barHeight
  );
  background.endFill();
  
  // Create health fill (fills from bottom, depletes from top)
  const healthFill = new PIXI.Graphics();
  
  // Color based on health percentage
  let fillColor;
  if (health > (maxHPForMe * 0.6)) {
    fillColor = 0x00ff00; // Green
  } else if (health > (maxHPForMe * 0.3)) {
    fillColor = 0xffff00; // Yellow
  } else {
    fillColor = 0xff0000; // Red
  }
  
  const depletedHeight = barHeight * (1 - (health  / maxHPForMe));
  
  healthFill.beginFill(fillColor);
  healthFill.drawRect(
    cx + offsetX,
    cy + offsetY + depletedHeight, // Start drawing from depleted point
    barWidth,
    barHeight * (health / maxHPForMe) // Height based on health percentage
  );
  healthFill.endFill();
  
  // Add components to container
  healthBar.addChild(border);
  healthBar.addChild(background);
  healthBar.addChild(healthFill);
  
  cityLayer.addChild(healthBar);
}

function drawCity(cx, cy, texture, scale) {
  const spr = new PIXI.Sprite(texture);
  spr.anchor.set(0.5);
  spr.scale.set(scale);                   // cityDef.s = 0.05
  spr.x = cx;
  spr.y = cy;
  cityLayer.addChild(spr);
}

function drawCap(cx, cy, texture, scale) {
  const spr = new PIXI.Sprite(texture);
  spr.anchor.set(0.5);
  spr.scale.set(scale);                   // cityDef.s = 0.05
  spr.x = cx - hexRadius / 2;
  spr.y = cy - hexRadius / 2;
  cityLayer.addChild(spr);
}

function drawCityReligion(cx, cy, idx) {
  if (idx >= 0) {
    // Create the badge graphics
    const badge = new PIXI.Graphics();
    
    // Draw the rounded rectangle with fill and stroke
    badge.beginFill(0x1a1a2e);
    badge.lineStyle(1, 0xB68F40);
    badge.drawCircle(0, 0, 7);
    badge.endFill();
    badge.x = cx + hexRadius / 2;
    badge.y = cy - hexRadius / 2;
    cityLayer.addChild(badge);

    const texture = religionIcons[idx];
    const scale = constants.religionConfigs[idx].s / 6.5; 
    const spr = new PIXI.Sprite(texture);
    spr.anchor.set(0.5);
    spr.scale.set(scale);                   // cityDef.s = 0.05
    spr.x = cx + hexRadius / 2;
    spr.y = cy - hexRadius / 2;
    cityLayer.addChild(spr);
  }
}

// Drawing traderoutes
function hexLine(r0, c0, r1, c1) {
  const start = oddr_to_cube([r0, c0]);
  const end   = oddr_to_cube([r1, c1]);
  const N = Math.max(Math.abs(start[0] - end[0]), Math.abs(start[1] - end[1]), Math.abs(start[2] - end[2]));
  const results = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const cube = cube_lerp(start, end, t);
    const rounded = cube_round(cube);
    results.push(cube_to_oddr(rounded));
  }
  return results;
}

function oddr_to_cube([r, q]) {
  const x = q - ((r & 1) === 1 ? (r - 1) / 2 : r / 2);
  const z = r;
  const y = -x - z;
  return [x, y, z];
}

function cube_to_oddr([x, y, z]) {
  const r = z;
  const q = x + ((r & 1) === 1 ? (r - 1) / 2 : r / 2);
  return [r, q];
}

function cube_lerp(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function cube_round([x, y, z]) {
  let rx = Math.round(x);
  let ry = Math.round(y);
  let rz = Math.round(z);

  const dx = Math.abs(rx - x);
  const dy = Math.abs(ry - y);
  const dz = Math.abs(rz - z);

  if (dx > dy && dx > dz) {
    rx = -ry - rz;
  } else if (dy > dz) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }

  return [rx, ry, rz];
}

function drawDashedLine(g, points, dashLength = 8, gapLength = 4) {
  for (let i = 0; i < points.length - 1; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[i + 1];

    const dx = x2 - x1;
    const dy = y2 - y1;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);

    let drawn = 0;
    while (drawn < dist) {
      const dash = Math.min(dashLength, dist - drawn);
      const xStart = x1 + Math.cos(angle) * drawn;
      const yStart = y1 + Math.sin(angle) * drawn;
      const xEnd = x1 + Math.cos(angle) * (drawn + dash);
      const yEnd = y1 + Math.sin(angle) * (drawn + dash);
      g.moveTo(xStart, yStart);
      g.lineTo(xEnd, yEnd);
      drawn += dash + gapLength;
    }
  }
}


function drawTraderoute(cx, cy, playerFrom, cityFrom, playerTo, cityTo, playerCityLocations, csCityLocations) {
  const fromHex = playerCityLocations[playerFrom][cityFrom];
  const toHex = (playerTo > 5)
    ? csCityLocations[playerTo - 6]
    : playerCityLocations[playerTo][cityTo];


  const path = hexLine(...fromHex, ...toHex);

  const points = path.map(([r, c]) => {
    const offsetX = (r % 2) * (hexWidth / 2);
    const x = c * horizSpacing + offsetX + hexRadius;
    const y = r * vertSpacing + hexRadius;
    return [x, y];
  });

  const g = new PIXI.Graphics();

  if (playerFrom === playerTo) {
    // Internal: dashed player-colored line
    g.lineStyle(2, 0xffd700, 0.7);
    drawDashedLine(g, points, 8, 4);
  } else if (playerTo > 5) {
    // To city-state: dotted cyan line
    g.lineStyle(2, 0x00ffff, 0.8);
    drawDashedLine(g, points, 2, 6);
  } else {
    // External: solid gold line
    g.lineStyle(3, 0xffd700, 0.85);
    g.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      g.lineTo(points[i][0], points[i][1]);
    }
  }

  g.stroke(); // don't forget to stroke the line
  traderouteLayer.addChild(g);
}

// Overlay Objects
async function initTechTreeOverlay() {
  /* load tech data ------------------------------------------ */
  const techArray = await fetch('technologies.json').then(r => r.json());

  /* --- create transparent Pixi canvas that sits above the map --- */
  const techCanvas = document.getElementById('techCanvas');

  const techApp = new PIXI.Application();          // ① plain constructor
  await techApp.init({                             // ② initialise
    view: techCanvas,
    width : techCanvas.width,      // 2400
    height: techCanvas.height,     // 1000
    backgroundAlpha: 0,
    antialias: true,
    autoDensity: true,
  });

  /* now techApp.stage and techApp.ticker are ready */
  const techTree = new TechTreeRendererPixi(techApp, techArray);

  /* UI hooks exactly like before ---------------------------- */
  const toggleBtn = document.getElementById('tech-tree-button');
  const turnBox = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');
  
  // helper: true if the button is “on”
  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';

  toggleBtn.addEventListener("click", e => {
    const on = isPressed(toggleBtn);
    techCanvas.style.display = on ? 'block' : 'none';
    if (on) {
      techTree.setPlayerTechs(playerTechs);
      techTree.setTurn(+turnBox.value);
      techTree.start();
    } else {
      techTree.stop();
    }
  });

  turnBox.addEventListener('input', () => {
    const on = isPressed(toggleBtn);
    if (on) techTree.setTurn(+turnBox.value);
  });
  prevTurn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    if (on) techTree.setTurn(+turnBox.value);
  });
  nextTurn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    if (on) techTree.setTurn(+turnBox.value);
  });
}

async function initPolicyOverlay() {
  /* 1️⃣  load JSON ----------------------------------------- */
  const polArray = await fetch('social_policies.json').then(r => r.json());

  /* 2️⃣  create transparent Pixi canvas -------------------- */
  const polCanvas = document.getElementById('policyCanvas');
  const polApp    = new PIXI.Application();
  await polApp.init({
    view           : polCanvas,
    width          : polCanvas.width,     // or use resizeTo: window
    height         : polCanvas.height,
    backgroundAlpha: 0,
    antialias      : true,
    autoDensity    : true,
  });

  // Scale down the entire stage to 80%
  polApp.stage.scale.set(0.8);

  // Optional: Center the scaled content if needed
  polApp.stage.position.set(
    polCanvas.width * 0.1,  // Center horizontally (10% margin on each side)
    polCanvas.height * 0.1  // Center vertically
  );

  /* 3️⃣  build renderer ------------------------------------ */
  const polScreen = new PolicyScreenRendererPixi(polApp, polArray);

  /* 4️⃣  DOM controls (same ids as before) ----------------- */
  const toggleBtn = document.getElementById('social-policy-button');
  const turnBox  = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');
  
  // helper: true if the button is “on”
  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';
  
  toggleBtn.addEventListener("click", e => {
    const on = isPressed(toggleBtn);
    polCanvas.style.display = on ? 'block' : 'none';
    if (on) {
      polScreen.setPlayerPolicies(playerPolicies);
      polScreen.setTurn(+turnBox.value);
      polScreen.start();
    } else {
      polScreen.stop();
    }
  });


  const syncTurn = () => { 
    const on = isPressed(toggleBtn);
    if (on) polScreen.setTurn(+turnBox.value); 
  };
  turnBox .addEventListener('input', syncTurn);
  prevTurn.addEventListener('click', syncTurn);
  nextTurn.addEventListener('click', syncTurn);
}

async function initReligionOverlay () {
  /* ── 1. Pixi renderer bound to <canvas id="religionCanvas"> ───────── */
  const relCanvas = document.getElementById('religionCanvas');

  const relApp = new PIXI.Application();
  await relApp.init({
    view           : relCanvas,
    width          : relCanvas.width,     // or use resizeTo: window
    height         : relCanvas.height,
    backgroundAlpha: 0,
    antialias      : true,
    autoDensity    : true
  });
  
  // Scale down the entire stage to 80%
  relApp.stage.scale.set(0.8);

  // Optional: Center the scaled content if needed
  relApp.stage.position.set(
    relCanvas.width * 0.1,  // Center horizontally (10% margin on each side)
    relCanvas.height * 0.1  // Center vertically
  );

  /* ── 2. Build the screen renderer ─────────────────────────────────── */
  const relScreen = new ReligionScreenRendererPixi(
    relApp,
    constants.religiousTenetNames,
    constants.religiousTenetCategories
  );

  /* ── 3. Hook up the existing UI controls ──────────────────────────── */
  const toggleBtn = document.getElementById('religion-button');
  const turnBox  = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');

  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';

  /** helper: push current turn value into renderer (if overlay visible) */
  const syncTurn = () => {
    const on = isPressed(toggleBtn);
    if (on) relScreen.setTurn(+turnBox.value);
  };

  /* checkbox show / hide */
  toggleBtn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    relCanvas.style.display = on ? 'block' : 'none';

    if (on) {
      relScreen.setPlayerReligion(playerReligion);   // inject data
      relScreen.setTurn(+turnBox.value);
      relScreen.start();
    } else {
      relScreen.stop();
    }
  });

  /* turn controls */
  turnBox.addEventListener('input', syncTurn);
  prevTurn.addEventListener('click', syncTurn);
  nextTurn.addEventListener('click', syncTurn);
}

async function initVictoryOverlay() {
  const vicCanvas = document.getElementById('victoryCanvas');
  
  const vicApp = new PIXI.Application();

  await vicApp.init({
    view: vicCanvas,
    width: vicCanvas.width,
    height: vicCanvas.height,
    backgroundAlpha: 0,  // This is crucial
    antialias: true,
    autoDensity: true
  });
  
  // Scale down the entire stage to 80%
  vicApp.stage.scale.set(1.0);
  
  vicApp.stage.position.set(
    vicCanvas.width * 0.0,  // Center horizontally (10% margin on each side)
    vicCanvas.height * 0.0 - 30  // Center vertically
  );

  const vicScreen = new VictoryScreenRendererPixi(vicApp);
  
  const toggleBtn = document.getElementById('victory-button');
  const turnBox = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');
  
  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';
  
  toggleBtn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    vicCanvas.style.display = on ? 'block' : 'none';
    
    if (on) {
      vicScreen.setData(csReligiousPopulation, csRelationships, csInfluence, csType, csQuest, csCultureTracker, csFaithTracker, csTechTracker, csTradeTracker, csReligionTracker, csWonderTracker, csResourceTracker, playerTechs, haveMet, tradeLedger, atWar, unitsType, unitsMilitary, unitHealth, unitsRowCol, unitsCombatBonus, hasSacked, cultureTotal, tourismTotal, gpps, gpThreshold, playerGWSlots, numDelegates, playerBuildings);
      vicScreen.setTurn(+turnBox.value);
      vicScreen.start();
    } else {
      vicScreen.stop();
    }
  });
  
  const syncTurn = () => { 
    const on = isPressed(toggleBtn);
    if (on) vicScreen.setTurn(+turnBox.value); 
  };

  turnBox .addEventListener('input', syncTurn);
  prevTurn.addEventListener('click', syncTurn);
  nextTurn.addEventListener('click', syncTurn);
  
  // Sync turn changes
  //turnBox.addEventListener('input', () => {
  //  if (chk.checked) vicScreen.setTurn(+turnBox.value);
  //});
}

async function initTradeOverlay() {
  const tradeCanvas = document.getElementById('tradeCanvas');
  
  const tradeApp = new PIXI.Application();
  await tradeApp.init({
    view           : tradeCanvas,
    width          : tradeCanvas.width,     // or use resizeTo: window
    height         : tradeCanvas.height,
    backgroundAlpha: 0,
    antialias      : true,
    autoDensity    : true
  });
  
  // Scale down the entire stage to 80%
  tradeApp.stage.scale.set(1.0);

  // Optional: Center the scaled content if needed
  tradeApp.stage.position.set(
    tradeCanvas.width * 0.0,  // Center horizontally (10% margin on each side)
    tradeCanvas.height * 0.0 + 20  // Center vertically
  );
  
  const tradeScreen = new TradeScreenRendererPixi(
    tradeApp,
  );
  
  /* ── 3. Hook up the existing UI controls ──────────────────────────── */
  const toggleBtn = document.getElementById('trade-button');
  const turnBox  = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');

  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';

  /** helper: push current turn value into renderer (if overlay visible) */
  const syncTurn = () => {
    const on = isPressed(toggleBtn);
    if (on) tradeScreen.setTurn(+turnBox.value);
  };
  
  /* checkbox show / hide */
  toggleBtn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    tradeCanvas.style.display = on ? 'block' : 'none';

    if (on) {
      tradeScreen.setData(    
        tradeLedger,
        tradeLengthLedger,
        tradeGPTAdj,
        tradeResourceAdj,
        unitsMilitary,
        unitsTradePlayerTo,
        unitsTradeCityTo,
        unitsTradeCityFrom,
        unitsTradeYield,
        unitsEngaged,
        resourcesOwned,
      );
      tradeScreen.setTurn(+turnBox.value);
      tradeScreen.start();
    } else {
      tradeScreen.stop();
    }
  });

  /* turn controls */
  turnBox.addEventListener('input', syncTurn);
  prevTurn.addEventListener('click', syncTurn);
  nextTurn.addEventListener('click', syncTurn);
  
}


async function initDemographicsOverlay() {
  const demoCanvas = document.getElementById('demographicsCanvas');
  
  const demoApp = new PIXI.Application();
  await demoApp.init({
    view: demoCanvas,
    width: demoCanvas.width,
    height: demoCanvas.height,
    backgroundAlpha: 0,
    antialias: true,
    autoDensity: true
  });
  
  // Scale down the entire stage to 80%
  demoApp.stage.scale.set(0.8);

  // Optional: Center the scaled content if needed
  demoApp.stage.position.set(
    demoCanvas.width * 0.1,  // Center horizontally (10% margin on each side)
    demoCanvas.height * 0.1  // Center vertically
  );
  
  const demoScreen = new DemographicsScreenRendererPixi(demoApp);
  
  // Hook up controls
  const toggleBtn = document.getElementById('demos-button');
  const turnBox = document.getElementById('turnInput');
  const prevTurn = document.getElementById('prevTurn');
  const nextTurn = document.getElementById('nextTurn');
  
  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';
  
  toggleBtn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    demoCanvas.style.display = on ? 'block' : 'none';
    
    if (on) {
      // Aggregate all demographics data sources
      const demographicsData = {
        population: playerPops,  // Assuming these exist
        //gdp: playerGDP,
        land: playerOwnership,
        yields: playerYields,
        literacy: playerTechs,
        tourism: tourismTotal,
        treasury: treasury,
        netHappiness: happiness,
        goldenAgeTurns: goldenAgeTurns,
        /*
        military: playerMilitary,
        approval: playerApproval,
        literacy: playerLiteracy,
        production: playerProduction,
        food: playerFood,
        science: playerScience,
        gold: playerGold,
        culture: playerCulture,
        faith: playerFaith
        */
      };
      demoScreen.setDemographicsData(demographicsData);
      demoScreen.setMaxTurn(terrainMap.length);  // Set the maximum turn if known
      demoScreen.setTurn(+turnBox.value);
      demoScreen.start();
    } else {
      demoScreen.stop();
    }
  });
  
  // Sync turn changes
  turnBox.addEventListener('input', () => {
    const on = isPressed(toggleBtn);
    if (on) demoScreen.setTurn(+turnBox.value);
  });
  prevTurn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    if (on) demoScreen.setTurn(+turnBox.value);
  });
  nextTurn.addEventListener('click', () => {
    const on = isPressed(toggleBtn);
    if (on) demoScreen.setTurn(+turnBox.value);
  });
}

initTechTreeOverlay();    
initPolicyOverlay();
initReligionOverlay();
initVictoryOverlay();
initTradeOverlay();
initDemographicsOverlay();

/**
 * Build prefix element-wise maxima for A (T x P x C).
 * Returns an object with:
 *   - get(turn): view (Float32Array) of the prefix max at `turn`
 *   - toMatrix(view): convert a flat view to [P][C] nested arrays
 * Memory: ~ T*P*C*4 bytes (Float32)
 */
function buildPrefixMax3D(A) {
  const T = A.length;
  if (T === 0) throw new Error("A must have at least 1 turn");
  const P = A[0].length;
  const C = A[0][0].length;
  const PC = P * C;

  // Flattened buffer storing all prefix maxima for each turn
  const buf = new Float32Array(T * PC);

  // Helpers
  const idx = (t, p, c) => t * PC + p * C + c;

  // t = 0 (copy A[0])
  for (let p = 0; p < P; p++) {
    const row = A[0][p];
    for (let c = 0; c < C; c++) {
      buf[idx(0, p, c)] = row[c];
    }
  }

  // t >= 1: prefix max against previous
  for (let t = 1; t < T; t++) {
    const frame = A[t];
    for (let p = 0; p < P; p++) {
      const row = frame[p];
      for (let c = 0; c < C; c++) {
        const prev = buf[idx(t - 1, p, c)];
        const v = row[c];
        buf[idx(t, p, c)] = v > prev ? v : prev;
      }
    }
  }

  // Reusable -Infinity matrix for turn = 0 (optional: allocate lazily)
  const negInf = new Float32Array(PC);
  for (let i = 0; i < PC; i++) negInf[i] = -Infinity;

  return {
    T, P, C, PC,
    /**
     * get(turn): returns a flat Float32Array *view* of length P*C.
     * NOTE: view aliases internal storage—treat as read-only.
     */
    get(turn) {
      if (turn < 0 || turn > T) throw new RangeError(`turn must be in [0, ${T}]`);
      if (turn === 0) return negInf;                 // max over empty prefix
      const t = turn - 1;
      const off = t * PC;
      return buf.subarray(off, off + PC);
    },
    /**
     * toMatrix(view): converts a flat P*C view to nested [P][C] arrays.
     * (Handy if you prefer 2D.)
     */
    toMatrix(view) {
      if (view.length !== PC) throw new Error("bad view length");
      const out = Array.from({ length: P }, () => new Float32Array(C));
      for (let p = 0; p < P; p++) {
        out[p].set(view.subarray(p * C, (p + 1) * C));
      }
      return out;
    }
  };
}

// Hide loading overlay
const loadingOverlay = document.getElementById('loadingOverlay');
loadingOverlay.classList.add('fade-out');
setTimeout(() => {
  loadingOverlay.style.display = 'none';
}, 300); // Wait for fade animation

// City management screen. We include this here instead of an independent file
// for severa reasons. The main one is convenience, though. Sorry.
const viewState = {
  selectedCity        : null,
  selectedCityPlayer  : null,
  selectedCityNum     : null,
  selectedCityXY      : null,
  zoom                : 1.0,
  turn                : 0,
  buildPanelScroll: 0,
};


function centreCameraOnPixel(px, py) {
  /* app.screen is the logical render target (already DPI-scaled) */
  const vpW = app.screen.width;
  const vpH = app.screen.height;
  const z   = world.scale.x;          // uniform zoom

  /* 1️⃣ move the pivot so that (px,py) is the local origin */
  world.pivot.set(px, py);

  /* 2️⃣ put that origin in the exact centre of the viewport */
  world.position.set(vpW / 2, vpH / 2);
}

/* convert axial (col,row) to map-pixel centre */
function hexToPixel(col, row, r = hexRadius) {
  const w = Math.sqrt(3) * r;          // flat-to-flat
  const h = 2 * r;
  const vert = h * 0.75;
  const offX = (row & 1) ? w / 2 : 0;
  
  const x = col * w + offX + w / 2;
  const y = row * vert + r
  viewState.selectedCityXY = { x, y };
  return { x: x,  y: y };
}

function moveCameraToFirstCity(playerIdx, stepIdx = +turnInput.value) {
  const slice = playerCities[stepIdx];              // [P][C][2]
  //for (let p = 0; p < slice.length; p++) {
    for (let c = 0; c < slice[playerIdx].length; c++) {
      const [row, col] = slice[playerIdx][c];
      if (row >= 0 && col >= 0) {
        viewState.selectedCity = {row, col};
        viewState.selectedCityPlayer = playerIdx;
        viewState.selectedCityNum = c;
        const { x, y } = hexToPixel(col, row);
        centreCameraOnPixel(x, y);                  // ← new pivot logic
        return;
      }
    }
  //}
}


document.getElementById('cityview-button').addEventListener('click', () => {
  const nowPressed = document.getElementById('cityview-button').getAttribute('aria-pressed') === 'true';
  document.getElementById('cityview-button').setAttribute('aria-pressed', !nowPressed);   // toggle
  //renderMapDynamic(getLayerToggleStates());
  const toggleBtn = document.getElementById('cityview-button');
  const isPressed = btn => btn.getAttribute('aria-pressed') === 'true';
  const on = isPressed(toggleBtn);
  
  const cityScroller = document.getElementById("cityScroller");
  const playerDropdown = document.getElementById("playerDropdown");

  if (on) {
    moveCameraToFirstCity(parseInt(playerDropdown.value));   // <——  centre on a city
    viewState.cityViewEnabled = true;    // flag you’ll need for step 2
    
    cityScroller.style.display = "flex";
    cityScroller.querySelectorAll("button").forEach(btn => {
      btn.disabled = false;
    });
    playerDropdown.style.display = "flex";

  } else {
    viewState.cityViewEnabled = false;

    cityScroller.style.display = "none";
    cityScroller.querySelectorAll("button").forEach(btn => {
      btn.disabled = true;
    });
    playerDropdown.style.display = "none";
  }
  renderMapDynamic(getLayerToggleStates());                    // re-draw the frame right away
});


document.getElementById("playerDropdown").addEventListener("change", () => {
  moveCameraToFirstCity(parseInt(playerDropdown.value));
  renderMapDynamic(getLayerToggleStates());
});


function scrollToPreviousCity() {
  const turn = document.getElementById("turnInput");
  const prevTurn = document.getElementById("prevTurn");
  const nextTurn = document.getElementById("nextTurn");
  const nCitiesCurrPlayer = playerCities[turn.value][viewState.selectedCityPlayer].length;
  const currCity = viewState.selectedCityNum;

  if (nCitiesCurrPlayer === 0) return;
  const POPS = playerPops[turn.value][viewState.selectedCityPlayer]; // e.g., [3, 0, 0, 2, 0, 5]
  const n = POPS.length;
  let prevCityIdx = viewState.selectedCityNum;

  // Search up to n times, stepping backward and wrapping
  for (let i = 1; i <= n; i++) {
    const idx = (prevCityIdx - i + n) % n;
    if (POPS[idx] > 0) {
      prevCityIdx = idx;
      break;
    }
  }

  // Now prevCityIdx is the previous city with POP > 0
  viewState.selectedCityNum = prevCityIdx;

  // Decrement and wrap
  //const prevCityIdx = (currCity === 0) ? nCitiesCurrPlayer - 1 : currCity - 1;
  //viewState.selectedCityNum = prevCityIdx;

  // Now to extract the previous city's row-col
  const rc = playerCities[turn.value][viewState.selectedCityPlayer][prevCityIdx];
  const { x: px, y: py } = hexToPixel(rc[1], rc[0], hexRadius);
  centreCameraOnPixel(px, py);
  renderMapDynamic(getLayerToggleStates());                    // re-draw the frame right away
  //renderMap(getLayerToggleStates());
}

function scrollToNextCity() {
  // We can access information in the viewstate like current player, current city num, etc
  const turn = document.getElementById("turnInput");
  const prevTurn = document.getElementById("prevTurn");
  const nextTurn = document.getElementById("nextTurn");
  
  const nCitiesCurrPlayer = playerCities[turn.value][viewState.selectedCityPlayer].length;
  const currCity = viewState.selectedCityNum;
  
  if (nCitiesCurrPlayer === 0) return;
 // Increment and wrap
  const POPS = playerPops[turn.value][viewState.selectedCityPlayer];
  const n = POPS.length;
  let nextCityIdx = viewState.selectedCityNum;

  // Search up to n times (wraparound)
  for (let i = 1; i <= n; i++) {
    const idx = (nextCityIdx + i) % n;
    if (POPS[idx] > 0) {
      nextCityIdx = idx;
      break;
    }
  }
  viewState.selectedCityNum = nextCityIdx;

  //const nextCityIdx = (currCity + 1 >= nCitiesCurrPlayer || POP <= 0) ? 0 : currCity + 1;
  //viewState.selectedCityNum = nextCityIdx;

  // Now to extract the next city's row-col
  const rc = playerCities[turn.value][viewState.selectedCityPlayer][nextCityIdx];
  const { x: px, y: py } = hexToPixel(rc[1], rc[0], hexRadius);
  centreCameraOnPixel(px, py);
  renderMapDynamic(getLayerToggleStates());                    // re-draw the frame right away
  //renderMap(getLayerToggleStates());
}

document.getElementById("prevCity").addEventListener("click", () => {
  if (!viewState.cityViewEnabled) return;
  scrollToPreviousCity();
});

document.getElementById("nextCity").addEventListener("click", () => {
  if (!viewState.cityViewEnabled) return;
  scrollToNextCity();
});

function drawCityYieldPanelPixi(step, pID, cID) {
  /* 0️⃣  --- fetch data ----------------------------------- */
  const yields = playerYields?.[step]?.[pID]?.[cID];   // [8] numbers
  //if (!yields) return;

  const POP    = playerPops?.[step]?.[pID]?.[cID] ?? 0;
  //console.log("XY ", world.toLocal(new PIXI.Point(0, 0)));
  const globalXY = world.toLocal(new PIXI.Point(0, 0));

  /* 1️⃣  --- layout constants ------------------------------ */
  const pad       = 6;
  const lineH     = 32;
  const panelW    = 210;
  const panelH    = pad * 2 + lineH * 9 + 10;          // header + 8 rows
  const panelX    = constants.cityview_x_offset_column1;   // screen-space
  const panelY    = 8 + constants.cityview_y_offset;
  const inset = 20;

  const yieldNames = [
    'Food', 'Production', 'Gold', 'Faith',
    'Culture', 'Science', 'Happiness', 'Tourism'
  ];

  /* 2️⃣  --- build panel container ------------------------- */
  const panel = new PIXI.Container();
  panel.position.set(panelX, panelY);

  /* background + border */
  const g = new PIXI.Graphics()
    .beginFill(0x242119, 0.75)
    .drawRect(0, 0, panelW, panelH)
    .endFill()
    .lineStyle(2, "#B68F40")
    .drawRect(0, 0, panelW, panelH)
    .endFill();
  panel.addChild(g);

  /* 3️⃣  --- population header ----------------------------- */
  const header = new PIXI.Graphics()
    .beginFill(0x10b121)                            // Civ-V green
    .drawCircle(pad + inset, pad + inset, 10)             // 20-px circle
    .endFill();
  panel.addChild(header);

  /* population icon */
  const popSprite = new PIXI.Sprite(cityYieldsTextures[0]);
  popSprite.anchor.set(0.5);
  popSprite.scale.set(constants.cityYieldsDef[0].s);
  popSprite.position.set(pad + inset, pad + inset);
  panel.addChild(popSprite);

  /* “X Citizens” label */
  const popText = new PIXI.Text(`${POP} Citizens`, headerStyle);
  popText.anchor.set(0, 0.5);
  popText.position.set(58, pad + 19);
  panel.addChild(popText);
  
  const HEADER_H = pad + 35
   /* 4️⃣  --- yield rows ------------------------------------ */
  for (let k = 0; k < 8; k++) {
    const amount = yields[k];
    const rowY   = pad + HEADER_H + k * lineH;      // HEADER_H = 32

    /* icon */
    const icon = new PIXI.Sprite(cityYieldsTextures[k + 1]);
    icon.anchor.set(0.5);
    icon.scale.set(constants.cityYieldsDef[k + 1].s);
    icon.position.set(pad + 10, rowY + lineH / 2);
    panel.addChild(icon);
    
    /* yield name (left) */
    const nameTxt = new PIXI.Text(yieldNames[k], detailStyle2);
    //nameTxt.anchor.set(0, 0.5);
    nameTxt.position.set(pad + 30, rowY + lineH / 2 - 15);
    panel.addChild(nameTxt);

    /* value (right) */
    const valTxt = new PIXI.Text(amount.toFixed(0), detailStyleRight);
    //valTxt.anchor.set(1, 0.5);
    valTxt.position.set(panelW - pad - 5 - 25, rowY + lineH / 2 - 15);
    panel.addChild(valTxt);

  }

  /* 5️⃣  --- add to overlay layer -------------------------- */
  cityUILayer.addChild(panel);
}

function drawWorkedRing(cx, cy) {
  const r    = hexRadius * 0.95;                  // world-units radius
  const z    = world.scale.x;                     // current zoom
  const wPx  = 3;                                 // 3 px on screen
  const wWorld = wPx / z;                         // convert to world-units

  workedTileLayer.lineStyle(wWorld, 0x00ff00, 0.8)
                 .beginFill(0x00ff00, 0.2);

  for (let i = 0; i < 6; i++) {
    const ang = (60 * i - 30) * Math.PI / 180;
    const vx  = cx + r * Math.cos(ang) + 5.5;
    const vy  = cy + r * Math.sin(ang);
    i ? workedTileLayer.lineTo(vx, vy)
      : workedTileLayer.moveTo(vx, vy);
  }
  workedTileLayer.closePath().endFill();
}
 
function drawWorkedTilesForCity(step, playerIdx, cityIdx,
                                          baseRow, baseCol) {

  const slotArr = workedSlots?.[step]?.[playerIdx]?.[cityIdx];
  if (!slotArr || slotArr.every(v => v === 0)) return;

  const OFF = (baseRow & 1)
            ? constants.ODD_ROW_THREE_RING_OFFSETS
            : constants.EVEN_ROW_THREE_RING_OFFSETS;

  for (let k = 0; k < 36; k++) {
    if (slotArr[k] !== 1) continue;
    const [dr, dc] = OFF[k];
    const row = baseRow + dr;
    const col = baseCol + dc;
    const { x, y } = hexToPixel(col, row);
    drawWorkedRing(x, y);
  }
}


function drawCityBuildingPanelPixi(turn, playerIdx, cityIdx, scrollY = 0) {
  const builds = playerBuildings?.[turn]?.[playerIdx]?.[cityIdx];
  if (!builds) return;

  /* ---------- build list ---------------------------------- */
  const entries = [];
  builds.forEach((has, idx) => { if (has) entries.push(constants.buildingNames[idx]); });

  clampBuildingScroll(entries.length);
  scrollY = viewState.buildPanelScroll;

  /* ---------- layout constants ---------------------------- */
  const padX   = 8;
  const padY   = 8;
  const lineH  = 20;
  const panelW = 210 + 2;
  const panelH = 180;
  const panelX = constants.cityview_x_offset_column1;            // to the right of yield panel
  const panelY = 325 + constants.cityview_y_offset;

  /* ========================================================
       build / update the panel container
     ======================================================== */
  const bldgPanel = new PIXI.Container();
  bldgPanel.position.set(panelX, panelY);

  /* background + border */
  const bg = new PIXI.Graphics()
    .beginFill(0x242119, 0.75)
    .drawRect(0, 0, panelW, panelH)
    .endFill()
    .lineStyle(4, "#B68F40")
    .drawRect(0, 0, panelW, panelH)
    .endFill();
  bldgPanel.addChild(bg);

  /* mask so list can scroll */
  const maskG = new PIXI.Graphics()
    .beginFill(0xffffff)
    .drawRect(0, 0, panelW, panelH)
    .endFill();
  bldgPanel.addChild(maskG);
  bldgPanel.mask = maskG;

  /* header icon */
  const icon = new PIXI.Sprite(palaceTexture);
  icon.anchor.set(0.5);
  icon.scale.set(constants.palaceIconDef.s);
  icon.position.set(32.5, 32.5 - 7);
  bldgPanel.addChild(icon);

  /* header text */
  const header = new PIXI.Text('Buildings', headerStyle);
  header.anchor.set(0.5);
  header.position.set(32.5 * 3 + 30 , 32.5 - 7 );
  bldgPanel.addChild(header);


  
  // Mask to block out building names from obscurring the panel label
  const listAreaY = 50;
  const visibleH  = panelH - listAreaY - padY; // 180-50-8 = 122

  const listMask = new PIXI.Graphics()
    .beginFill(0xffffff)
    .drawRect(padX, listAreaY, panelW - padX * 2, visibleH)
    .endFill();                              // mask ONLY the list area
  bldgPanel.addChild(listMask);

  /* inner container for scrolling rows */
  const list = new PIXI.Container();
  list.position.set(padX + 5, listAreaY);    // inside mask
  list.mask = listMask;                      // ← clipping!
  bldgPanel.addChild(list);
  bldgPanel.list       = list;
  bldgPanel.visibleH   = visibleH;
  bldgPanel.listAreaY  = listAreaY;
  
  cityUILayer.addChild(bldgPanel);

  /* ---------- rebuild text rows --------------------------- */
  bldgPanel.list.removeChildren();
  
  entries.forEach((txt, idx) => {
    const row = new PIXI.Text(txt, detailStyle2);
    row.y = idx * lineH - scrollY;
    bldgPanel.list.addChild(row);
  });
}

function isOverBuildPanel(x, y) {
  const panelX = constants.cityview_x_offset_column1;
  const panelY = 325 + constants.cityview_y_offset;
  const panelW = 210;
  const panelH = 180;
  return x >= panelX && x <= panelX + panelW && y >= panelY && y <= panelY + panelH;
}
/* ==========================================================
   (optional) helper to change scroll inside your wheel
   or button-scroll handlers
   ========================================================== */
export function clampBuildingScroll(visibleRows) {
  const maxScroll = Math.max(0, (visibleRows - 1) * 18);
  viewState.buildPanelScroll = Math.max(
    0,
    Math.min(viewState.buildPanelScroll, maxScroll)
  );
}

function drawCityAccelPanelPixi(turn, pID, cID) {

  /* 0️⃣  — fetch data */
  const accels = playerWonderAccel?.[turn]?.[pID]?.[cID];
  if (!accels) return;

  const b  = playerBldgAccel?.        [turn]?.[pID]?.[cID] ?? 0;
  const m  = playerMilitaryBldgAccel?.[turn]?.[pID]?.[cID] ?? 0;
  const c  = playerCultureBldgAccel?. [turn]?.[pID]?.[cID] ?? 0;
  const s  = playerSeaBldgAccel?.     [turn]?.[pID]?.[cID] ?? 0;
  const sc = playerScienceBldgAccel?. [turn]?.[pID]?.[cID] ?? 0;
  const r  = playerReligionBldgAccel?.[turn]?.[pID]?.[cID] ?? 0;
  const e  = playerEconBldgAccel?.    [turn]?.[pID]?.[cID] ?? 0;

  /* 1️⃣  — layout constants */
  const padX   = 8;
  const padY   = 8;
  const lineH  = 23;
  const panelW = 210;
  const panelH = 440;
  const panelX = constants.cityview_x_offset_column2;
  const panelY = constants.cityview_y_offset;

  const wonderEras = [
    'Ancient','Classical','Medieval','Renaissance',
    'Industrial','Modern','Post-modern','Future'
  ];

  /* 2️⃣  — build container (create once, update text later) */
  const panel = new PIXI.Container();
  panel.position.set(panelX, panelY  + 8);

  const g = new PIXI.Graphics()
    .beginFill(0x242119, 0.75)
    .drawRect(0, 0, panelW, panelH)
    .endFill()
    .lineStyle(2, "#B68F40")
    .drawRect(0, 0, panelW, panelH)
    .endFill();
  panel.addChild(g);

  /* header ------------------------------------------------ */
  const header = new PIXI.Text('Building \n \t\tAccel', headerStyle);
  header.anchor.set(0.5);
  header.position.set(panelW / 2, padY + 24);
  panel.addChild(header);

  /* “Wonders” sub-header --------------------------------- */
  //const subHead = new PIXI.Text('Wonders', detailStyle);
  //subHead.anchor.set(0.5);
  //subHead.position.set(panelW / 2 - 10, padY + 16 + 35);
  //panel.addChild(subHead);

  /* list container for rows (easy to clear/update) */
  const list = new PIXI.Container();
  list.position.set(0, padY + 16 + 45);
  panel.addChild(list);

  /* 3️⃣  — build row helper */
  const makeRow = (label, value, idx) => {
    const y = idx * lineH;

    const left = new PIXI.Text(label, detailStyle2);
    left.position.set(padX + 10, y);
    list.addChild(left);

    const right = new PIXI.Text(`${value.toFixed(2)}x`, detailStyleRight);
    right.anchor.set(1, 0);
    right.position.set(panelW - padX - 10, y);
    list.addChild(right);
  };

  /* 4️⃣  — wonder-era rows */
  wonderEras.forEach((era, i) => makeRow(era, accels[i] ?? 0, i));

  /* 5️⃣  — other accel rows */
  let idx = wonderEras.length;
  makeRow('Buildings' , b , idx++);
  makeRow('Military'  , m , idx++);
  makeRow('Culture'   , c , idx++);
  makeRow('Sea'       , s , idx++);
  makeRow('Science'   , sc, idx++);
  makeRow('Religion'  , r , idx++);
  makeRow('Economic'  , e , idx++);

  /* 6️⃣  — add to overlay layer */
  cityUILayer.addChild(panel);
}


function drawCityIsBuildingPanelPixi(turn, pID, cID) {
  const constructing = isConstructing?.[turn]?.[pID]?.[cID];
  const prod = prodReserves?.[turn]?.[pID]?.[cID];
  console.log("Making ",  constructing);
  //if (!constructing) return;

  // Layout
  const padX = 8;
  const padY = 8;
  const lineH  = 18;
  const panelW = 210;
  const panelH = 160;
  const panelX = constants.cityview_x_offset_column1;            // to the right of yield panel
  const panelY = 325 + constants.cityview_y_offset + 180 + 6;

  /* ========================================================
       build / update the panel container
     ======================================================== */
  const constructionPanel = new PIXI.Container();
  constructionPanel.position.set(panelX, panelY);

  /* background + border */
  const bg = new PIXI.Graphics()
    .beginFill(0x242119, 0.75)
    .drawRect(0, 0, panelW, panelH)
    .endFill()
    .lineStyle(2, "#B68F40")
    .drawRect(0, 0, panelW, panelH)
    .endFill();
  constructionPanel.addChild(bg);

  /* header text */
  const header = new PIXI.Text('Making', headerStyle);
  header.position.set(padX + 44, padY);
  constructionPanel.addChild(header);

  if (constructing >= 0) {
    const buildingName = constants.buildingNames[constructing];
    const buildingCost = constants.buildingCosts[constructing];
    const buildingText = new PIXI.Text(buildingName, detailStyle2);
    //buildingText.style.wordWrap      = true;
    //buildingText.style.wordWrapWidth = panelW - 10;
    //buildingText.updateText();
    buildingText.position.set(padX, padY + 80);
    constructionPanel.addChild(buildingText);

    const RIGHT_EDGE_X = panelW - padX - 30;
    const prodText = new PIXI.Text(`${prod} / ${buildingCost}`,  detailStyle2);
    prodText.anchor.x = 1;                // 0 = left, 0.5 = centre, 1 = right
    prodText.x = RIGHT_EDGE_X;
    //prodText.anchor.y = 0;
    prodText.y = 50;
    constructionPanel.addChild(prodText);

    const pText = new PIXI.Text("Prog:", detailStyle2);
    pText.position.set(padX, 48);
    constructionPanel.addChild(pText);
    
    const icon = new PIXI.Sprite(cityYieldsTextures[2]);
    icon.anchor.set(0.5);
    icon.scale.set(constants.cityYieldsDef[2].s);
    icon.position.set(padX + 68 + panelW - 95, padY + 55);
    constructionPanel.addChild(icon);
  }
  
  cityUILayer.addChild(constructionPanel);
  
}

const globalMapState = {
  previousFeatures: null,
  previousImprovements: null,
  previousYields: null,
}

// Effort to improve FPS: change detection
function hasArrayChanged(oldFeatures, newFeatures) {
  for (let r = 0; r < 42; r++) {
    for (let c = 0; c < 66; c++) {
      if (oldFeatures[r][c] !== newFeatures[r][c]) return true;
    }
  }
  return false;
}

function hasYieldArrayChanged(oldFeatures, newFeatures) {
  for (let r = 0; r < 42; r++) {
    for (let c = 0; c < 66; c++) {
      for (let i = 0; i < 7; i++) {
        if (oldFeatures[r][c][i] !== newFeatures[r][c][i]) return true;
      }
    }
  }
  return false;
}



function drawOwnershipLayer(cx, cy, _playerOwnership) {
  // _playerOwnership is an integer specifying who owns thie tile on cxcy
  if (_playerOwnership === 0) { return }
  const color = constants.playerColors[_playerOwnership];
  const hex = drawHexOwnershipv2(cx, cy, color);
  ownershipLayer.addChild(hex);
}

function writeCSName(cx, cy, csIdx, relIdx, typeIdx) {
  const name = constants.csNames[csIdx];

    // Create a container for this city's name and badge
  const nameContainer = new PIXI.Container();
  
  // Create text first to measure dimensions
  const textStyle = new PIXI.TextStyle(detailStyleCenter);
  
  const text = new PIXI.Text(name, textStyle);
  text.resolution = 4;

  // Calculate badge dimensions with padding
  const padding = 8;
  const badgeWidth = text.width + padding * 2 + 25;
  const badgeHeight = text.height + padding * 2 - 15;
  const cornerRadius = 6;
  
  // Create the badge graphics
  const badge = new PIXI.Graphics();
  
  // Draw the rounded rectangle with fill and stroke
  badge.beginFill(0x1a1a2e); // Background color
  badge.lineStyle(1, 0xB68F40); // Border color and width
  badge.drawRoundedRect(0, 0, badgeWidth, badgeHeight, cornerRadius);
  badge.endFill();
  
  // Position the badge centered on the hex coordinates
  const yOffset = 20;
  badge.x = cx - badgeWidth / 2;
  badge.y = (cy - badgeHeight / 2) - yOffset;
  
  // Position the text centered within the badge
  text.x = cx - text.width / 2;
  text.y = (cy - text.height / 2) - yOffset;
  
  // Add both to the container
  nameContainer.addChild(badge);
  nameContainer.addChild(text);
  
  // Add to the city name layer
  cityNameLayer.addChild(nameContainer);

  // Icon for CityState type
  const typeIcon = new PIXI.Sprite(csTypesTextures[typeIdx]);
  typeIcon.anchor.set(0.5);
  typeIcon.scale.set(constants.csConfigs[typeIdx].s);
  typeIcon.x = cx - (badgeWidth / 2) + padding + 2;
  typeIcon.y = cy - yOffset;
  cityNameLayer.addChild(typeIcon);
  
  if (relIdx >= 0) {
    const texture = religionIcons[relIdx];
    const scale = constants.religionConfigs[relIdx].s / 6.5; 
    const spr = new PIXI.Sprite(texture);
    spr.anchor.set(0.5);
    spr.scale.set(scale);                   // cityDef.s = 0.05
    spr.x = cx + (badgeWidth / 2) - padding - 2;
    spr.y = cy - yOffset;
    cityNameLayer.addChild(spr);
  }
}

function drawCSReligion(cx, cy, idx) {
  if (idx >= 0) {
    const texture = religionIcons[idx];
    const scale = constants.religionConfigs[idx].s / 6.5; 
    const spr = new PIXI.Sprite(texture);
    const yOffset = 20;
    spr.anchor.set(0.5);
    spr.scale.set(scale);                   // cityDef.s = 0.05
    spr.x = cx + (badgeWidth / 2) + padding + 2;
    spr.y = cy - yOffset;
    cityLayer.addChild(spr);
  }
}

function drawHexCoords(cx, cy, row, col) {
  const container = new PIXI.Container();
  container.x = cx;
  container.y = cy;
  const coord = new PIXI.Text(`${row}, ${col}`, detailStyle);
  coord.anchor.set(0.5, 0.5);
  coord.style.fontSize = 12;
  container.addChild(coord);
  debugLayer.addChild(container);
}

// Main rendering fns
// First is the static stuff (i.e., the things that will not change from turn-to-turn)
// I think we'll need to just draw everything on the static call, except for things
// that are toggleable?
async function renderMapStatic({ showYieldsBool = false, showResourcesBool = false, showImprovementsBool = false, turn = 0, showTechTreeBool = false, yieldTypeIdx = 0} = {}) {

  // This function is only ever called once, so let's go ahead and build the per-city lookup table of max health per turn
  globalMapState.cityMaxHPLookup = buildPrefixMax3D(cityHP);

  // Clear previous tiles
  //app.stage.removeChildren();
  //mapLayer.removeChildren();

  const terrainMapLocal = ISREPLAY ? terrainMap[turn] : terrainMap;
  const lakeMapLocal = ISREPLAY ? lakeMap[turn] : lakeMap;
  const elevationMapLocal = ISREPLAY ? elevationMap[turn] : elevationMap;
  const featureMapLocal = ISREPLAY ? featureMap[turn] : featureMap;
  const nwMapLocal = ISREPLAY ? nwMap[turn] : nwMap;
  const riverMapLocal = ISREPLAY ? riverMap[turn] : riverMap;
  const csOwnershipBordersLocal = ISREPLAY ? csOwnershipBorders[turn] : csOwnershipBorders;
  const csOwnershipLocal = ISREPLAY ? csOwnership[turn] : csOwnership;
  const cs_citiesLocal = ISREPLAY ? cs_cities[turn] : cs_cities;
  const unitsTypeLocal = ISREPLAY ? unitsType[turn] : unitsType;
  const unitsMilitaryLocal = ISREPLAY ? unitsMilitary[turn] : unitsMilitary;
  const unitsRowColLocal = ISREPLAY ? unitsRowCol[turn] : unitsRowCol;
  const playerCitiesLocal = ISREPLAY ? playerCities[turn] : playerCities;
  const playerOwnershipLocal = ISREPLAY ? playerOwnership[turn] : playerOwnership;
  const playerOwnershipBordersLocal = ISREPLAY ? playerOwnershipBorders[turn] : playerOwnershipBorders;
  const allResourceMapLocal = ISREPLAY ? allResourceMap[turn] : allResourceMap;
  const unitHealthLocal = unitHealth[turn];

  
  const use_textures = true;
  const clip_textures = false;
  resourceLayer.visible = false;

  globalMapState.previousFeatures = featureMapLocal;
  globalMapState.previousImprovements = featureMapLocal;  // spoofing, since we don't have improvements in static! Should trigger immediately 
  globalMapState.previousYields = playerYieldMap[0][0];

  for (let row = 0; row < terrainMapLocal.length; row++) {
    for (let col = 0; col < terrainMapLocal[0].length; col++) {
      const offsetX = (row % 2) * (hexWidth / 2);  // odd-row stagger
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing + hexRadius;
      
      const isLake = lakeMapLocal[row][col];
      let terrainType = terrainMapLocal[row][col];
      terrainType = isLake * 6 + (1 - isLake) * terrainType;
      const texture = terrainTextures[terrainType];
      const config = constants.terrainRenderConfigs[terrainType];

      const tile = drawHex(cx, cy, config?.fillColor, texture);
      //app.stage.addChild(tile);
      mapLayer.addChild(tile);
      drawHexCoords(cx, cy, row, col);

      /* --- elevation overlay --- */
      let elevID = elevationMapLocal[row][col];    // 0-3
      elevID = (1 - isLake) * elevID;
      addElevationSprite(cx, cy, elevID, elevTextures);

      /* --- feature overlay --- */
      /* determine if that tile is a hill */
      const onHill   = elevID === 2;               // true only for hills, not mountains

      const featID = featureMapLocal[row][col];       // 0–5 as per table
      addFeature(featID, onHill, cx, cy, featureTextures);
      
      /* --- Natural Wonders --- */
      const nwID = nwMapLocal[row][col];   // 0–17
      addNaturalWonder(nwID, cx, cy);
      
      //riverLayer.clear();
      const riverEdges = riverMapLocal[row]?.[col];             // array[6] booleans

      for (let e = 0; e < 6; e++) {
        if (riverEdges[e]) {
          traceRiverEdge(riverLayer, cx, cy, e);
        }
      }

      // City ownership borders
      /* ------- city-state borders ------- */
      const csID = csOwnershipLocal[row][col];          // 0 = none
      if (csID > 0) {
        const edges = csOwnershipBordersLocal[csID-1][row][col];
        const colour = hexStringToNumber(constants.csColors[csID]);
        for (let e = 0; e < 6; e++)
          if (edges[e]) addBorderEdge(borderLayer, cx, cy, e, colour);
      }

      /* ------- player borders ------- */
      const plID = playerOwnershipLocal[row][col];
      if (plID > 0) {
        const edges = playerOwnershipBordersLocal[plID-1][row][col];
        const colour = hexStringToNumber(constants.playerColors[plID]);
        for (let e = 0; e < 6; e++)
          if (edges[e]) addBorderEdge(borderLayer, cx, cy, e, colour);
      }

      // Resources
      // These are static (i.e., they do no change throughout the course of the game)
      // **from the perspective of an oracle viewer**
      addResource(allResourceMapLocal[row][col], cx, cy)
    }
  }

  for (let i = 0; i < cs_citiesLocal.length; i++) {
    const unit_rowcol = cs_citiesLocal[i];
    const row = unit_rowcol[0]; 
    const col = unit_rowcol[1];

    const offsetX = (row % 2) * (hexWidth / 2);  // odd-row stagger
    const cx = col * horizSpacing + offsetX + hexRadius;
    const cy = row * vertSpacing + hexRadius;
    drawCity(cx, cy, cityTexture, constants.cityDef.s);
  }


  for (let i = 0; i < playerCitiesLocal.length; i++) {
    for (let j = 0; j < playerCitiesLocal[i].length; j++) {
      const [row, col] = playerCitiesLocal[i][j];
      if (row === -1) continue;             // skip empty slot

      const offsetX = (row % 2) * (hexWidth / 2);
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing   + hexRadius;

      drawCity(cx, cy, cityTexture, constants.cityDef.s);
      
    }
  }

  for (let i = 0; i < unitsTypeLocal.length; i++) {
    for (let j = 0; j < unitsTypeLocal[0].length; j++) {
      const [row, col] = unitsRowColLocal[i][j];
      const offsetX = (row % 2) * (hexWidth / 2);
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing + hexRadius;

      drawUnits(i,                      // civ index
                unitsTypeLocal[i][j],   // 0‒3
                unitsMilitaryLocal[i][j],// 0 civilian, 1 military
                unitHealthLocal[i][j],
                cx, cy);
    }
  }
}



function renderMapDynamic({ showYieldsBool = false, showResourcesBool = false, showImprovementsBool = false, turn = 0, showTechTreeBool = false, yieldTypeIdx = 0, showOwnershipBool = false, showFOWBool = false, fowPlayerIdx = 0} = {}) {
  
  const terrainMapLocal = ISREPLAY ? terrainMap[turn] : terrainMap;
  const lakeMapLocal = ISREPLAY ? lakeMap[turn] : lakeMap;
  const elevationMapLocal = ISREPLAY ? elevationMap[turn] : elevationMap;
  const featureMapLocal = ISREPLAY ? featureMap[turn] : featureMap;
  const nwMapLocal = ISREPLAY ? nwMap[turn] : nwMap;
  const riverMapLocal = ISREPLAY ? riverMap[turn] : riverMap;
  const csOwnershipBordersLocal = ISREPLAY ? csOwnershipBorders[turn] : csOwnershipBorders;
  const csOwnershipLocal = ISREPLAY ? csOwnership[turn] : csOwnership;
  const cs_citiesLocal = ISREPLAY ? cs_cities[turn] : cs_cities;
  const unitsTypeLocal = ISREPLAY ? unitsType[turn] : unitsType;
  const unitsMilitaryLocal = ISREPLAY ? unitsMilitary[turn] : unitsMilitary;
  const unitsRowColLocal = ISREPLAY ? unitsRowCol[turn] : unitsRowCol;
  const unitsTradePlayerToLocal = ISREPLAY ? unitsTradePlayerTo[turn] : unitsTradePlayerTo;
  const unitsTradeCityToLocal = ISREPLAY ? unitsTradeCityTo[turn] : unitsTradeCityTo;
  const unitsTradeCityFromLocal = ISREPLAY ? unitsTradeCityFrom[turn] : unitsTradeCityFrom;
  const unitsTradeYieldLocal = ISREPLAY ? unitsTradeYield[turn] : unitsTradeYield;
  const playerCitiesLocal = ISREPLAY ? playerCities[turn] : playerCities;
  const playerOwnershipLocal = ISREPLAY ? playerOwnership[turn] : playerOwnership;
  const playerOwnershipBordersLocal = ISREPLAY ? playerOwnershipBorders[turn] : playerOwnershipBorders;
  const improvementMapLocal = ISREPLAY ? improvementMap[turn] : improvementMap;
  const unitHealthLocal = unitHealth[turn];
  const cityHPLocal = cityHP[turn];
  const roadMapLocal = roadMap[turn];
  
  const use_textures = true;
  const clip_textures = false;

  // Switching between different players' views of the yields (or GT) -- playerYieldMap
  if (yieldTypeIdx === "GT") {
    //console.log("IN GT BLOCK");
    //let yieldMapToUse = gtYieldMap;
    var YieldMapLocal = ISREPLAY ? gtYieldMap[turn] : gtYieldMap;
  } else {
    //console.log(playerYieldMap);
    //let yieldMapToUse = playerYieldMap;
    var YieldMapLocal = ISREPLAY ? playerYieldMap[turn][Number(yieldTypeIdx)] : playerYieldMap;
  }


  resourceLayer.visible = showResourcesBool;
  improvementsLayer.visible = showImprovementsBool;
  yieldLayer.visible = showYieldsBool;
  cityUILayer.visible = viewState.cityViewEnabled;

  // Checking to see if features have changed at all
  const featuresChangedBool = hasArrayChanged(globalMapState.previousFeatures, featureMapLocal);
  
  let improvementsChangedBool = false;
  if (showImprovementsBool) {
    improvementsChangedBool = hasArrayChanged(globalMapState.previousImprovements, improvementMapLocal);
  }

  let yieldsChangedBool = false;
  if (showYieldsBool) {
    yieldsChangedBool = hasYieldArrayChanged(globalMapState.previousYields, YieldMapLocal);
  }
  
  if (featuresChangedBool) {
    featureLayer.removeChildren();
    globalMapState.previousFeatures = featureMapLocal;
  }
  
  if (improvementsChangedBool && showImprovementsBool) {
    improvementsLayer.removeChildren();
    globalMapState.previousImprovements = improvementMapLocal;
  }

  if (yieldsChangedBool && showYieldsBool) {
    //  yields are [turn, player, 42, 66, 7]
    //  local version: [42, 66, 7]
    yieldLayer.removeChildren();
    globalMapState.previousYields = YieldMapLocal;
  }

  if (showOwnershipBool) {
    //ownershipLayer.removeChildren().forEach(c => c.destroy(true));
    //ownershipLayer.clear();

//    ownershipLayer.removeChildren();
  }
  ownershipLayer.removeChildren();
  
  unitsLayer.removeChildren();
  traderouteLayer.removeChildren(); // or should be .clear()?
  cityLayer.removeChildren();

  cityUILayer.removeChildren();
  workedTileLayer.clear(); 
  borderLayer.clear();
  cityNameLayer.removeChildren();
  fowLayer.removeChildren();

  for (let row = 0; row < terrainMapLocal.length; row++) {
    for (let col = 0; col < terrainMapLocal[0].length; col++) {
      const offsetX = (row % 2) * (hexWidth / 2);  // odd-row stagger
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing + hexRadius;
      
      const isLake = lakeMapLocal[row][col];

      let elevID = elevationMapLocal[row][col];    // 0-3

      if (featuresChangedBool){
        elevID = (1 - isLake) * elevID;
        const onHill = elevID === 2;               // true only for hills, not mountains
        const featID = featureMapLocal[row][col];       // 0–5 as per table
        addFeature(featID, onHill, cx, cy, featureTextures);
      }

      // City ownership borders
      /* ------- city-state borders ------- */
      const csID = csOwnershipLocal[row][col];          // 0 = none
      if (csID > 0) {
        const edges = csOwnershipBordersLocal[csID-1][row][col];
        const colour = hexStringToNumber(constants.csColors[csID]);
        for (let e = 0; e < 6; e++)
          if (edges[e]) addBorderEdge(borderLayer, cx, cy, e, colour);
      }

      /* ------- player borders ------- */
      const plID = playerOwnershipLocal[row][col];
      if (plID > 0) {
        const edges = playerOwnershipBordersLocal[plID-1][row][col];
        const colour = hexStringToNumber(constants.playerColors[plID]);
        for (let e = 0; e < 6; e++)
          if (edges[e]) addBorderEdge(borderLayer, cx, cy, e, colour);
      }
      
      // Improvements
      if (improvementsChangedBool && showImprovementsBool) {
        const improvementType = improvementMapLocal[row][col];
        drawImprovements(improvementType, cx, cy);
      }

      if (roadMapLocal[row][col] > 0) {
        drawRoads(cx, cy);
      }
      
      if (yieldsChangedBool && showYieldsBool) {
        drawYield(cx, cy, YieldMapLocal[row][col]);
      }

      if (showOwnershipBool) {
        drawOwnershipLayer(cx, cy, playerOwnershipLocal[row][col]);
      }

      if (showFOWBool) {
        drawFOW(cx, cy, fow[turn][fowPlayerIdx][row][col]);
      }
       
    }
  }

  for (let i = 0; i < cs_citiesLocal.length; i++) {
    //console.log("NUM CS: ", cs_citiesLocal.length);
    const unit_rowcol = cs_citiesLocal[i];
    const row = unit_rowcol[0]; 
    const col = unit_rowcol[1];

    const offsetX = (row % 2) * (hexWidth / 2);  // odd-row stagger
    const cx = col * horizSpacing + offsetX + hexRadius;
    const cy = row * vertSpacing + hexRadius;
    drawCity(cx, cy, cityTexture, constants.cityDef.s);
    
    // turn?
    const typeIdx = csType[turn][i];
    const religionPop = csReligiousPopulation[turn][i];

    if (religionPop.reduce((acc, num) => acc + num, 0) === 0) {
      writeCSName(cx, cy, i, -1, typeIdx);
    } else {
      writeCSName(cx, cy, i, religionPop.indexOf(Math.max(...religionPop)), typeIdx);
    }
  }

  for (let i = 0; i < playerCitiesLocal.length; i++) {
    for (let j = 0; j < playerCitiesLocal[i].length; j++) {
      const [row, col] = playerCitiesLocal[i][j];
      if (row === -1) continue;             // skip empty slot

      const offsetX = (row % 2) * (hexWidth / 2);
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing   + hexRadius;

      drawCity(cx, cy, cityTexture, constants.cityDef.s);
      drawCityHealth(cx, cy, turn, i, j, cityHPLocal[i][j]);
      
      // Cap will always be in the first slot, as it is not capturable
      if (j === 0) {
        drawCap(cx, cy, capTexture, constants.capDef.s);
      }

      // Drawing religion symbol on the top-right of the hex
      const religionPop = playerReligiousPop[turn][i][j];
    
      if (religionPop.reduce((acc, num) => acc + num, 0) === 0) {
        drawCityReligion(cx, cy, -1);
      } else {
        drawCityReligion(cx, cy, religionPop.indexOf(Math.max(...religionPop)));
      }

      if (
        viewState.cityViewEnabled &&
        i === viewState.selectedCityPlayer &&
        j === viewState.selectedCityNum
      ) {
        drawWorkedTilesForCity(turn, i, j, row, col);
        drawCityYieldPanelPixi(turn, i, j);
        drawCityBuildingPanelPixi(turn, i, j);
        drawCityIsBuildingPanelPixi(turn, i, j);
        drawCityAccelPanelPixi(turn, i,  j);

      }
    }
  }


  for (let i = 0; i < unitsTypeLocal.length; i++) {
    for (let j = 0; j < unitsTypeLocal[0].length; j++) {
      const [row, col] = unitsRowColLocal[i][j];
      const offsetX = (row % 2) * (hexWidth / 2);
      const cx = col * horizSpacing + offsetX + hexRadius;
      const cy = row * vertSpacing + hexRadius;
    
      if (unitsTypeLocal[i][j] === 5) {
        if (unitsTradeYieldLocal[i][j].flat().reduce((a, b) => a + b, 0) > 0) {
          drawTraderoute(cx, cy, i, unitsTradeCityFromLocal[i][j], unitsTradePlayerToLocal[i][j], unitsTradeCityToLocal[i][j], playerCitiesLocal, cs_citiesLocal);
        }
      }
      drawUnits(i,                      // civ index
                unitsTypeLocal[i][j],   // 0‒3
                unitsMilitaryLocal[i][j],// 0 civilian, 1 military
                unitHealthLocal[i][j],
                cx, cy);
    }
  }
}
