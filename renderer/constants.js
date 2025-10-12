import * as PIXI from 'https://cdn.jsdelivr.net/npm/pixi.js@8.x/dist/pixi.mjs';

export const terrainColors = [
  //99: "#30b2c5", # rivers
  //98: "#30b2c5", # lakes
  "#08306b",  // ocean
  "#74c476",  // grassland
  "#e5c07b",  // plains
  "#f0e442",  // desert
  "#999999",  // tundra
  "#ffffff", // snow
]

export const csColors = [
  0, "#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#264653", "#4CC9F0", "#5C4D7D", "#F28482", "#1D3557", "#9D4EDD", "#FBB13C", "#B0BEC5"];

export const playerColors = [
  0, "#FF2A2F", "#FFD700", "#0A63BA", "#32CD32", "#FFFFFF", "#D55E0D"
]
export const playerColorsScreens = [
  "#FF2A2F", "#FFD700", "#0A63BA", "#32CD32", "#FFFFFF", "#D55E0D"
]

export const csNames = [
  "Antwerp", "Genoa", "Kabul", "Lhasa",
  "Malacca", "Hong Kong", "Monaco", "Wittenberg",
  "Ormus", "Tyre", "Ragusa", "Zanzibar",
]

export const csConfigs = [
  { path: './icons/cs/cs_cultural.png', s: 0.06 },
  { path: './icons/cs/cs_agri.png', s: 0.06 },
  { path: './icons/cs/cs_merc.png', s: 0.06 },
  { path: './icons/cs/cs_rel.png', s: 0.06 },
  { path: './icons/cs/cs_sci.png', s: 0.06 },
  { path: './icons/cs/cs_mil.png', s: 0.06 },
]

export async function loadCsIcons() {
  const icons = [];
  for (let i = 0; i < csConfigs.length; i++) {
    const cfg = csConfigs[i];
    icons[i] = cfg?.path ? await PIXI.Assets.load(cfg.path) : null;
  }
  return icons;
}

// Textures for the map tiles
const textureMap = {
  grassland: await PIXI.Assets.load('./textures/grassland_entire_map.png'),
  plains: await PIXI.Assets.load('./textures/plains_entire_map.png'),
  desert: await PIXI.Assets.load('./textures/desert_entire_map.png'),
  tundra: await PIXI.Assets.load('./textures/asia_tundra_d.png'),
  snow: await PIXI.Assets.load('./textures/generic_snow_d.png'),
  lakes: await PIXI.Assets.load('./textures/water_entire_map.png'),
  ocean: await PIXI.Assets.load('./textures/ocean_test.png'),
};

export const terrainRenderConfigs = [
  { fillColor: "#08306b" },
  { texturePath: './textures/grassland_entire_map.png' },  
  { texturePath: './textures/plains_entire_map.png' },     
  { texturePath: './textures/desert_entire_map.png' },     
  { texturePath: './textures/asia_tundra_d.png' },         
  { texturePath: './textures/generic_snow_d.png' },        
  { texturePath: './textures/water_entire_map.png'}
];

export async function loadTerrainTextures() {
  const terrainTextures = [];

  for (let i = 0; i < terrainRenderConfigs.length; i++) {
    const config = terrainRenderConfigs[i];
    if (config.texturePath) {
      terrainTextures[i] = await PIXI.Assets.load(config.texturePath);
    } else {
      terrainTextures[i] = null; // still fill the array
    }
  }
  return terrainTextures;
}


/* icons + per-type parameters */
export const elevationConfigs = [
  null,  // 0 ocean
  null,  // 1 flatland
  { path: './icons/hill.png',      scale: 0.255, yOff:  0 },  // 2 hill
  { path: './icons/mountain2.png', scale: 0.25, yOff: -5 }  // 3 mountain
];

/* load only the needed textures */
export async function loadElevationTextures() {
  const textures = [];
  for (let i = 0; i < elevationConfigs.length; i++) {
    const cfg = elevationConfigs[i];
    textures[i] = cfg?.path ? await PIXI.Assets.load(cfg.path) : null;
  }
  return textures;
}

// religions
export const religionConfigs = [
  { path: './icons/religions/bud.png', s: 0.4 },
  { path: './icons/religions/confus.png', s: 0.25 },
  { path: './icons/religions/shinto.png', s: 0.25 },
  { path: './icons/religions/taosim.png', s: 0.23 },
  { path: './icons/religions/tengriism.png', s: 0.25 },
  { path: './icons/religions/zoroastrianism.png', s: 0.22 },
]

export async function loadReligionIcons() {
  const icons = [];
  for (let i = 0; i < religionConfigs.length; i++) {
    const cfg = religionConfigs[i];
    icons[i] = cfg?.path ? await PIXI.Assets.load(cfg.path) : null;
  }
  return icons;
}

/* id: 0 ocean / none, 1 forest, 2 jungle, 3 marsh, 4 oasis, 5 floodplains */
export const featureDefs = [
  /* 0 */ null,
  /* 1 */ {
          normal: { path: './icons/forest.png',      scale: 0.29, yOff:   0 },
          hill  : { path: './icons/foresthill.png',  scale: 0.275, yOff:  +5 }
        },
  /* 2 */ {
          normal: { path: './icons/jungle.png',      scale: 0.25, yOff:   0 },
          hill  : { path: './icons/junglehill.png',  scale: 0.253, yOff:   -4 }
        },
  /* 3 */ { normal: { path: './icons/marsh.png',     scale: 0.22, yOff:  0 } },
  /* 4 */ { normal: { path: './icons/oasis.png',     scale: 0.22, yOff: -6 } },
  /* 5 */ { normal: { path: './icons/floodplains.png', scale: 0.34, yOff: -17 } },
];

export async function loadFeatureTextures() {
  const textures = [];
  for (let id = 0; id < featureDefs.length; id++) {
    const def = featureDefs[id];
    if (!def) { textures[id] = null; continue; }

    const obj = {};
    if (def.normal?.path)
      obj.normal = await PIXI.Assets.load(def.normal.path);
    if (def.hill?.path)
      obj.hill   = await PIXI.Assets.load(def.hill.path);

    textures[id] = obj;
  }
  return textures; 
}

export const nwDefs = [
  null,
  { path: './icons/krakatoa.png', scale: 0.24, yOff: -12 },
  { path: './icons/gibraltar.png', scale: 0.24, yOff: -10 },
  { path: './icons/kailash.png', scale: 0.24, yOff: -10 },
  { path: './icons/kilimanjaro.png', scale: 0.24, yOff:  -2 },
  { path: './icons/sinai.png', scale: 0.24, yOff:  -4 },
  { path: './icons/sripada.png', scale: 0.24, yOff: -10 },
  { path: './icons/cerro.png', scale: 0.24, yOff:  -5 },
  { path: './icons/fuji.png', scale: 0.24, yOff:  -5 },
  { path: './icons/uluru.png', scale: 0.24, yOff:  -1 },
  { path: './icons/barringer.png', scale: 0.24, yOff:  -1 },
  { path: './icons/grand_mesa.png', scale: 0.24, yOff:  -1 },
  { path: './icons/oldfaithful.png', scale: 0.24, yOff:  -8 },
  { path: './icons/fountain.png', scale: 0.24, yOff:  -1 },
  { path: './icons/gbr.png', scale: 0.24, yOff:  -1 },
  { path: './icons/eldorado.png', scale: 0.24, yOff:  -1 },
  { path: './icons/solomon.png', scale: 0.24, yOff:  -1 },
  { path: './icons/lakevictoria.png', scale: 0.24, yOff:  -1 },
];

export async function loadNWTextures() {
  const tex = [];
  for (let id = 0; id < nwDefs.length; id++) {
    const def = nwDefs[id];
    tex[id] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return tex;
}

//const nwTextures = await loadNWTextures();
export const resourceDefs = [
  /* 0  */ null,

  /* 1  */ { path: './icons/resources/dyes.png',        s: 0.12 },
  /* 2  */ { path: './icons/resources/copper.png',      s: 0.12 },
  /* 3  */ { path: './icons/resources/deer.png',        s: 0.12 },
  /* 4  */ { path: './icons/resources/ivory.png',       s: 0.12 },
  /* 5  */ { path: './icons/resources/silver.png',      s: 0.12 },
  /* 6  */ { path: './icons/resources/jewelry.png',     s: 0.12 },
  /* 7  */ { path: './icons/resources/uranium.png',     s: 0.12 },
  /* 8  */ { path: './icons/resources/lapis.png',       s: 0.12 },
  /* 9  */ { path: './icons/resources/gems.png',        s: 0.12 },
  /* 10 */ { path: './icons/resources/iron.png',        s: 0.12 },
  /* 11 */ { path: './icons/resources/wine.png',        s: 0.12 },
  /* 12 */ { path: './icons/resources/cow.png',         s: 0.12 },
  /* 13 */ { path: './icons/resources/coconut.png',     s: 0.17 },
  /* 14 */ { path: './icons/resources/wheat.png',       s: 0.12 },
  /* 15 */ { path: './icons/resources/oil.png',         s: 0.12 },
  /* 16 */ { path: './icons/resources/marble.png',      s: 0.13 },
  /* 17 */ { path: './icons/resources/tobacco.png',     s: 0.13 },
  /* 18 */ { path: './icons/resources/maize.png',       s: 0.165 },
  /* 19 */ { path: './icons/resources/whales.png',      s: 0.12 },
  /* 20 */ { path: './icons/resources/olives.png',      s: 0.12 },
  /* 21 */ { path: './icons/resources/truffles.png',    s: 0.12 },
  /* 22 */ { path: './icons/resources/bison.png',       s: 0.12 },
  /* 23 */ { path: './icons/resources/sugar.png',       s: 0.12 },
  /* 24 */ { path: './icons/resources/horses.png',      s: 0.12 },
  /* 25 */ { path: './icons/resources/citrus.png',      s: 0.12 },
  /* 26 */ { path: './icons/resources/cotton.png',      s: 0.12 },
  /* 27 */ { path: './icons/resources/salt.png',        s: 0.12 },
  /* 28 */ { path: './icons/resources/gold.png',        s: 0.12 },
  /* 29 */ { path: './icons/resources/aluminium.png',   s: 0.12 },
  /* 30 */ { path: './icons/resources/incense.png',     s: 0.12 },
  /* 31 */ { path: './icons/resources/coffee.png',      s: 0.12 },
  /* 32 */ { path: './icons/resources/crabs.png',       s: 0.12 },
  /* 33 */ { path: './icons/resources/silk.png',        s: 0.12 },
  /* 34 */ { path: './icons/resources/perfume.png',     s: 0.12 },
  /* 35 */ { path: './icons/resources/glass.png',       s: 0.12 },
  /* 36 */ { path: './icons/resources/spices.png',      s: 0.12 },
  /* 37 */ { path: './icons/resources/amber.png',       s: 0.12 },
  /* 38 */ { path: './icons/resources/chocolate.png',   s: 0.12 },
  /* 39 */ { path: './icons/resources/rubber.png',      s: 0.12 },
  /* 40 */ { path: './icons/resources/coal.png',        s: 0.12 },
  /* 41 */ { path: './icons/resources/sheep.png',       s: 0.12 },
  /* 42 */ { path: './icons/resources/coral.png',       s: 0.12 },
  /* 43 */ { path: './icons/resources/furs.png',        s: 0.12 },
  /* 44 */ { path: './icons/resources/porcelain.png',   s: 0.12 },
  /* 45 */ { path: './icons/resources/fish.png',        s: 0.12 },
  /* 46 */ { path: './icons/resources/tea.png',         s: 0.12 },
  /* 47 */ { path: './icons/resources/hardwood.png',    s: 0.16 },
  /* 48 */ { path: './icons/resources/obsidian.png',    s: 0.169 },
  /* 49 */ { path: './icons/resources/banana.png',      s: 0.12 },
  /* 50 */ { path: './icons/resources/jade.png',        s: 0.12 },
  /* 51 */ { path: './icons/resources/pearls.png',      s: 0.12 },
  /* 52 */ { path: './icons/resources/stone.png',       s: 0.12 },
];

export async function loadResourceTextures() {
  const tex = [];
  for (let id = 0; id < resourceDefs.length; id++) {
    const def = resourceDefs[id];
    tex[id] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return tex;
}

export const ALL_RESOURCES = [
  null,
  "dyes",
  "copper",
  "deer",
  "ivory",
  "silver",
  "jewelry",
  "uranium",
  "lapis",
  "gems",
  "iron",  
  "wine",
  "cow",
  "coconut",
  "wheat",
  "oil",
  "marble",
  "tobacco",
  "maize",
  "whales",
  "olives",  
  "truffles",
  "bison",
  "sugar",
  "horses",
  "citrus",
  "cotton",
  "salt",
  "gold",
  "aluminium",
  "incense",  
  "coffee",
  "crabs",
  "silk",
  "perfume",
  "glass",
  "spices",
  "amber",
  "chocolate",
  "rubber",
  "coal",  
  "sheep",
  "coral",
  "furs",
  "porcelain",
  "fish",
  "tea",
  "hardwood",
  "obsidian",
  "banana",
  "jade",  
  "pearls",
  "stone",
]


export const improvementDefs = [
  null,  // 0 no improvement

  /* 1 */ { path: './icons/improvements/farmicon2.png',      s: 0.12 },
  /* 2 */ { path: './icons/improvements/pastureicon.png',    s: 0.12 },
  /* 3 */ { path: './icons/improvements/miningicon.png',     s: 0.12 },
  /* 4 */ { path: './icons/improvements/fishingboaticon.png',s: 0.12 },
  /* 5 */ { path: './icons/improvements/plantationicon.png', s: 0.12 },
  /* 6 */ { path: './icons/improvements/campicon.png',       s: 0.12 },
  /* 7 */ { path: './icons/improvements/quarryicon.png',     s: 0.12 },
  /* 8 */ { path: './icons/improvements/lumbermillicon.png', s: 0.12 },
  /* 9 */ { path: './icons/improvements/forticon.png',       s: 0.12 },
  /*10 */ { path: './icons/improvements/tradingposticon.png',s: 0.12 },
  /*11 */ { path: './icons/improvements/road.png',           s: 0.07 }
];

export async function loadImprovementTextures() {
  const tex = [];
  for (let id = 0; id < improvementDefs.length; id++) {
    const def = improvementDefs[id];
    tex[id] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return tex;
}


// 6 yield kinds × 5 magnitudes (1-5)  ➜ 30 entries
export const yieldDefs = [
  // k = kind index 0-5, m = magnitude 1-5
  // use id = k*5 + (m-1) + 1    (0 reserved = none)
  /*  0 */ null,

  /* FOOD (k=0) */
  { path:'./icons/yield_food.png',      s:0.029 + 0.001 },
  { path:'./icons/yield_food2.png',     s:0.055 + 0.001 },
  { path:'./icons/yield_food3.png',     s:0.055 + 0.001 },
  { path:'./icons/yield_food4.png',     s:0.055 + 0.001 },
  { path:'./icons/yield_food5.png',     s:0.055 + 0.001 },

  /* PROD (k=1) */
  { path:'./icons/yield_prod.png',      s:0.022 + 0.001 },
  { path:'./icons/yield_prod2.png',     s:0.085 + 0.001 },
  { path:'./icons/yield_prod3.png',     s:0.085 + 0.001 },
  { path:'./icons/yield_prod4.png',     s:0.085 + 0.001 },
  { path:'./icons/yield_prod5.png',     s:0.043 + 0.001 },

  /* GOLD (k=2) */
  { path:'./icons/yield_gold.png',      s:0.024 + 0.001 },
  { path:'./icons/yield_gold2.png',     s:0.05 + 0.001 },
  { path:'./icons/yield_gold3.png',     s:0.05 + 0.001 },
  { path:'./icons/yield_gold4.png',     s:0.05 + 0.001 },
  { path:'./icons/yield_gold5.png',     s:0.04 + 0.001 },

  /* FAITH (k=3) */
  { path:'./icons/yield_faith.png',     s:0.023 + 0.001 },
  { path:'./icons/yield_faith2.png',    s:0.04 + 0.001 },
  { path:'./icons/yield_faith3.png',    s:0.04 + 0.001 },
  { path:'./icons/yield_faith4.png',    s:0.04 + 0.001 },
  { path:'./icons/yield_faith5.png',    s:0.03 + 0.001 },

  /* CULTURE (k=4) */
  { path:'./icons/yield_culture.png',   s:0.024 + 0.001 },
  { path:'./icons/yield_culture2.png',  s:0.038 + 0.001 },
  { path:'./icons/yield_culture3.png',  s:0.038 + 0.001 },
  { path:'./icons/yield_culture4.png',  s:0.038 + 0.001 },
  { path:'./icons/yield_culture5.png',  s:0.038 + 0.001 },

  /* SCIENCE (k=5) */
  { path:'./icons/yield_science.png',   s:0.02 + 0.001 },
  { path:'./icons/yield_science2.png',  s:0.033 + 0.001 },
  { path:'./icons/yield_science3.png',  s:0.033 + 0.001 },
  { path:'./icons/yield_science4.png',  s:0.033 + 0.001 },
  { path:'./icons/yield_science5.png',  s:0.031 + 0.001 },
];

/* special delimiter icon (5+ circle) */
export const yieldOver5 = { path:'./icons/yield_over_5.png', s:0.0 };

export async function loadYieldTextures() {
  const tex = [];
  for (let i = 0; i < yieldDefs.length; i++) {
    const def = yieldDefs[i];
    tex[i] = def ? await PIXI.Assets.load(def.path) : null;
  }
  tex['over5'] = await PIXI.Assets.load(yieldOver5.path);
  return tex;
}

export const unitBGDefs = [
  null,   // 0  = no badge

  /* triangles (shape 0) */
  { path:'./icons/units/bg_triangle_1.png', s:0.1 },
  { path:'./icons/units/bg_triangle_2.png', s:0.1 },
  { path:'./icons/units/bg_triangle_3_new.png', s:0.1 },
  { path:'./icons/units/bg_triangle_4.png', s:0.1 },
  { path:'./icons/units/bg_triangle_5.png', s:0.1 },
  { path:'./icons/units/bg_triangle_6.png', s:0.1 },

  /* circles (shape 1) */
  { path:'./icons/units/bg_circle_1.png',   s:0.1 },
  { path:'./icons/units/bg_circle_2.png',   s:0.1 },
  { path:'./icons/units/bg_circle_3_new.png',   s:0.1 },
  { path:'./icons/units/bg_circle_4.png',   s:0.1 },
  { path:'./icons/units/bg_circle_5.png',   s:0.1 },
  { path:'./icons/units/bg_circle_6.png',   s:0.1 },
];

/* Loader – call ONCE at startup */
export async function loadUnitBGTextures() {
  const textures = [];
  for (let i = 0; i < unitBGDefs.length; i++) {
    const def = unitBGDefs[i];
    textures[i] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return textures;           // textures[id] ready for draw call
}

export const unitNames = ['Settler', 'Warrior', 'Worker', 'Archer', 'Caravan', 'Chariot Archer', 'Pikeman', 'Scout', 'Spearman', 'Catapult', 'Composite Bowman', 'Horseman', 'Swordsman', 'Crossbowman', 'Knight', 'Longswordsman', 'Trebuchet', 'Cannon', 'Lancer', 'Musketman', 'Airship', 'Artillery', 'Cavalry', 'Expeditionary Force', 'Gatling Gun', 'Rifleman', 'Anti Tank Rifle', 'Infantry', 'Landship', 'Machine Gun', 'Anti Tank Gun', 'Bazooka', 'Helicopter Gunship', 'Marine', 'Rocket Artillery', 'Tank', 'XCOM Squad'];


export const unitDefs = [
  null,
  { path:'./icons/units/settler.png', s:0.05 },
  { path:'./icons/units/warrior.png', s:0.05 },
  { path:'./icons/units/worker.png',  s:0.04 },
  { path:'./icons/units/archer.png',  s:0.05 },
  { path:'./icons/units/caravan.png',  s:0.04 },
  { path:'./icons/units/chariot_archer.png',  s:0.05 },
  { path:'./icons/units/pikeman.png',  s:0.05 },
  { path:'./icons/units/scout.png',  s:0.05 },
  { path:'./icons/units/spearman.png',  s:0.05 },
  { path:'./icons/units/catapult.png',  s:0.05 },
  { path:'./icons/units/composite_bowman.png',  s:0.052 },
  { path:'./icons/units/horseman.png',  s:0.063 },
  { path:'./icons/units/swordsman.png',  s:0.063 },
  { path:'./icons/units/crossbowman.png',  s:0.049 },
  { path:'./icons/units/knight.png',  s:0.05 },
  { path:'./icons/units/longswordsman.png',  s:0.065 },
  { path:'./icons/units/trebuchet.png',  s:0.065 },
  { path:'./icons/units/cannon.png',  s:0.06 },
  { path:'./icons/units/lancer.png',  s:0.05 },
  { path:'./icons/units/musketman.png',  s:0.05 },
  { path:'./icons/units/airship.png',  s:0.05 },
  { path:'./icons/units/artillery.png',  s:0.057 },
  { path:'./icons/units/cavalry.png',  s:0.06 },
  { path:'./icons/units/expeditionary_forces.png',  s:0.06 },
  { path:'./icons/units/gatlinggun.png',  s:0.06 },
  { path:'./icons/units/rifleman.png',  s:0.055 },
  { path:'./icons/units/antitank_rifle.png',  s:0.05 },
  { path:'./icons/units/infantry.png',  s:0.055 },
  { path:'./icons/units/landship.png',  s:0.055 },
  { path:'./icons/units/machine_gun.png',  s:0.055 },
  { path:'./icons/units/antitank_gun.png',  s:0.055 },
  { path:'./icons/units/bazooka.png',  s:0.055 },
  { path:'./icons/units/helicopter_gunship.png',  s:0.055 },
  { path:'./icons/units/marine.png',  s:0.05 },
  { path:'./icons/units/rocket_artillery.png',  s:0.055 },
  { path:'./icons/units/tank.png',  s:0.055 },
  { path:'./icons/units/xcom.png',  s:0.055 },
];

/* Loader – call ONCE at startup */
export async function loadUnitTextures() {
  const textures = [];
  for (let i = 0; i < unitDefs.length; i++) {
    const def = unitDefs[i];
    textures[i] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return textures;           // textures[id] ready for draw call
}

export const cityDef = { path: './icons/city_icon.png', s: 0.27 };

export async function loadCityTexture() {
  return await PIXI.Assets.load(cityDef.path);   // returns a PIXI.Texture
}

export const capDef = { path: './icons/cap_city.png', s: 0.05 };

export async function loadCapTexture() {
  return await PIXI.Assets.load(capDef.path);   // returns a PIXI.Texture
}

export const religiousTenetNames = ['Altars Of Worship', 'Ancestor Worship', 'Dance Of The Aurora', 'Desert Folklore', 'Divine Judgement', 'Earth Mother', 'God Of Craftsmen', 'God Of The Open Sky', 'God Of The Sea', 'God Of War', 'God King', 'Goddess Of Festivals', 'Goddess Of Love', 'Goddess Of Protection', 'Goddess Of The Fields', 'Goddess Of The Hunt', 'Harvest Festival', 'Messenger Of The Gods', 'Mystic Rituals', 'Oceans Bounty', 'One With Nature', 'Oral Tradition', 'Rain Dancing', 'Religious Idols', 'Religious Settlements', 'Rite Of Spring', 'Ritual Sacrifice', 'Sacred Path', 'Seafood Rituals', 'Spirit Animals', 'Spirit Trees', 'Starlight Guidance', 'Stone Circles', 'Sun God', 'Tears Of The Gods', 'Vision Quests', 'Works Spirituals', 'Ceremonial Burial', 'Church Property', 'Dawah', 'Initiation Rites', 'Messiah', 'Missionary Zeal', 'Mithraea', 'Religious Unity', 'Salat', 'Tithe', 'World Church', 'Zakat', 'Cathedrals', 'Choral Music', 'Devoted Elite', 'Devout Performers', 'Divine Inspiration', 'Feed The World', 'Followers Of The Refined Crafts', 'Gurdwaras', 'Guruship', 'Holy Warriors', 'Liturgical Drama', 'Mandirs', 'Mosques', 'Pagodas', 'Peace Gardens', 'Religious Art', 'Religious Community', 'Sacred Waters', 'Synagogues', 'Viharas', 'Defender Of The Faith', 'Dharma', 'Disciples', 'Hajj', 'Jizya', 'Just War', 'Karma', 'Kotel', 'Pilgrimage', 'Promised Land', 'Religious Troubadours', 'Sanctified Innovations', 'Unity Of The Prophets', 'Apostolic Palace', 'City Of God', 'Houses Of Worship', 'Indulgences', 'Jesuit Education', 'Sacred Sites', 'Swords Into Plowshares', 'Underground Sect', 'Work Ethic']

export const religiousTenetCategories = ['pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'pantheon', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'founder', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'follower', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'enhancer', 'reformation', 'reformation', 'reformation', 'reformation', 'reformation', 'reformation', 'reformation', 'reformation', 'reformation']


export const cityYieldsDef = [
  { path: './icons/pop.png', s: 0.17 },
  { path:'./icons/yield_food.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_prod.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_gold.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_faith.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_culture.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_science.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_happiness.png',      s:0.09 + 0.001 },
  { path:'./icons/yield_tourism.png',      s:0.09 + 0.001 },

];

export async function loadCityYieldsTexture() {
  const textures = [];
  for (let i = 0; i < cityYieldsDef.length; i++) {
    const def = cityYieldsDef[i];
    textures[i] = def ? await PIXI.Assets.load(def.path) : null;
  }
  return textures;           // textures[id] ready for draw call
}


export const EVEN_ROW_THREE_RING_OFFSETS = [
  [-3,  -2], [-3, -1], [-3, 0], [-3,  +1], 
  [-2, -2], [-2, -1], [-2, 0], [-2, +1], [-2, +2],
  [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, +1], [-1, +2],
  [0, -3], [0, -2], [0, -1], [0, +1], [0, +2], [0, +3],  // center
  [+1, -3], [+1, -2], [+1, -1], [+1, 0], [+1, +1], [+1, +2],
  [+2, -2], [+2, -1], [+2, 0], [+2, +1], [+2, +2],
  [+3, -2], [+3, -1], [+3, 0], [+3, +1]
];

export const ODD_ROW_THREE_RING_OFFSETS = [
  [-3,  -1], [-3, 0], [-3, +1], [-3,  +2], 
  [-2, -2], [-2, -1], [-2, 0], [-2, +1], [-2, +2],
  [-1, -2], [-1, -1], [-1, 0], [-1, +1], [-1, +2], [-1, +3],
  [0, -3], [0, -2], [0, -1], [0, +1], [0, +2], [0, +3],  // center
  [+1, -2], [+1, -1], [+1, 0], [+1, 1], [+1, +2], [+1, +3],
  [+2, -2], [+2, -1], [+2, 0], [+2, +1], [+2, +2],
  [+3, -1], [+3, 0], [+3, +1], [+3, +2]
];


export const buildingNames = ['Courthouse', 'Seaport', 'Stable', 'Watermill', 'Circus', 'Forge', 'Windmill', 'Hydro Plant', 'Solar Plant', 'Mint', 'Observatory', 'Monastery', 'Garden', 'Lighthouse', 'Harbor', 'Colosseum', 'Theatre', 'Stadium', 'Monument', 'Temple', 'Opera House', 'Museum', 'Broadcast Tower', 'Barracks', 'Armory', 'Military Academy', 'Arsenal', 'Walls', 'Castle', 'Military Base', 'Granary', 'Hospital', 'Medical Lab', 'Workshop', 'Factory', 'Nuclear Plant', 'Spaceship Factory', 'Market', 'Bank', 'Stock Exchange', 'Library', 'University', 'Public School', 'Laboratory', 'Palace', 'Heroic Epic', 'National College', 'National Epic', 'Circus Maximus', 'National Treasury', 'Ironworks', 'Oxford University', 'Hermitage', 'Great Lighthouse', 'Stonehenge', 'Great Library', 'Pyramids', 'Colossus', 'Oracle', 'Hanging Garden', 'Great Wall', 'Angkor Wat', 'Hagia Sophia', 'Chichen Itza', 'Machu Pichu', 'Notre Dame', 'Porcelain Tower', 'Himeji Castle', 'Sistine Chapel', 'Kremlin', 'Forbidden Palace', 'Taj Mahal', 'Big Ben', 'Louvre', 'Brandenburg Gate', 'Statue Of Liberty', 'Cristo Redentor', 'Eiffel Tower', 'Pentagon', 'Sydney Opera House', 'Aqueduct', 'Stone Works', 'Statue Zeus', 'Temple Artemis', 'Mausoleum of Halicarnassus', 'Amphitheater', 'Shrine', 'Recycling Center', 'Bomb Shelter', 'Constable', 'Police Station', 'Intelligence Agency', 'Alhambra', 'CN Tower', 'Hubble', 'Leaning Tower', 'Mosque Of Djenne', 'Neuschwanstein', 'Petra', 'Terracotta Army', 'Great Firewall', 'Cathedral', 'Mosque', 'Pagoda', 'Grand Temple', 'Tourist Center', 'Writers Guild', 'Artists Guild', 'Musicians Guild', 'Hotel', 'Caravansary', 'Airport', 'Uffizi', 'Globe Theater', 'Broadway', 'Red Fort', 'Prora Resort', 'Borobudur', 'Parthenon', 'International Space Station', 'Gurdwara', 'Synagogue', 'Conservatory', 'Vihara', 'Mandir', 'St. Peters', 'Althing', 'Gemcutter', 'Textile Maker', 'Censer Maker', 'Brewery', 'Grocer', 'Refinery', 'Grand Stele', 'Panama', 'Artist House', 'Writer House', 'Music House', 'Huey Teocali', 'Apollo Program', 'SS Booster 1', 'SS Booster 2', 'SS Booster 3', 'SS Engine', 'SS Cockpit',  'SS Stasis Chamber', 'Gallery', 'Scriptorium', 'Settler', 'Warrior', 'Worker', 'Archer', 'Caravan', 'Chariot Archer', 'Pikeman', 'Scout', 'Spearman', 'Catapult', 'Composite Bowman', 'Horseman', 'Swordsman', 'Crossbowman', 'Knight', 'Longswordsman', 'Trebuchet', 'Cannon', 'Lancer', 'Musketman', 'Airship', 'Artillery', 'Cavalry', 'Expeditionary Force', 'Gatling Gun', 'Rifleman', 'Anti Tank Rifle', 'Infantry', 'Landship', 'Machine Gun', 'Anti Tank Gun', 'Bazooka', 'Helicopter Gunship', 'Marine', 'Rocket Artillery', 'Tank', 'Xcom Squad']

export const buildingCosts = [50, 167, 67, 50, 50, 80, 167, 280, 333, 67, 133, -1, 80, 50, 80, 67, 120, 333, 27, 67, 133, 200, 333, 50, 107, 200, 200, 50, 107, 333, 40, 240, 333, 80, 240, 333, 240, 67, 133, 240, 50, 107, 200, 333, -1, 83, 83, 83, 83, 83, 83, 83, 83, 123, 123, 123, 123, 123, 167, 167, 167, 267, 200, 200, 200, 267, 417, 333, 333, 707, 333, 417, 500, 500, 500, 707, 802, 707, 802, 833, 67, 50, 123, 123, 123, 67, 27, 333, 200, 107, 200, 83, 267, 802, 833, 333, 167, 500, 167, 167, 833, -1, -1, -1, 83, 267, 67, 100, 133, 200, 60, 267, 417, 333, 707, 417, 707, 200, 167, -1, -1, -1, -1, -1, -1, 83, 200, 50, 67, 50, 67, 67, 133, 83, 707, -1, -1, -1, 200, 1005, 1005, 1005, 1005, 503, 503, 503, -1, -1, 49, 27, 32, 27, 50, 38, 60, 17, 38, 50, 50, 50, 50, 80, 80, 80, 80, 124, 124, 100, 100, 168, 151, 181, 151, 151, 201, 281, 261, 261, 241, 302, 342, 302, 342, 302, 342]

export const palaceIconDef = { path: './icons/icon_palace.png', s: 0.15 };

export async function loadPalaceTexture() {
  return await PIXI.Assets.load(palaceIconDef.path);   // returns a PIXI.Texture
}

// This will be the down-screen offset for the cityview screen
export const cityview_y_offset = 140;
export const cityview_x_offset_column1 = 22;
export const cityview_x_offset_column2 = 240;

export const tradeRouteLength = 2 - 1;
export const trdeDealLength = 15;
