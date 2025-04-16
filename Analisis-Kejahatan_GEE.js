// Zajy NR
// zaky@siberin.id
// Teknologi Informasi UNJAYA

// 1. AREA STUDI
var jakarta = ee.Geometry.Rectangle([106.6, -6.4, 107.0, -6.1]);
Map.centerObject(jakarta, 11);

// 2. DATA KEJAHATAN
var crimes = ee.List(['Pencurian', 'Perampokan', 'Vandalisme', 'Penipuan', 'Kekerasan']);
var randomImage = ee.Image.random().multiply(5).int();

var sampledPoints = randomImage.sample({
  region: jakarta,
  geometries: true,
  scale: 30,
  numPixels: 100
});

var crimeData = sampledPoints.map(function(feature) {
  var id = ee.Number(feature.get('random')).int();
  return feature.set({
    'crime_type_id': id,
    'crime_type': crimes.get(id),
    'severity': id.add(1),
    'year': 2023,
    'month': ee.Number(ee.Image.random().multiply(12).add(1).floor())
  });
});
print("Data Kejahatan (Sample 5):", crimeData.limit(5));

// 3. CITRA LINGKUNGAN
// LANDCOVER
var landcover = ee.ImageCollection('ESA/WorldCover/v100')
  .filterDate('2020-01-01', '2021-12-31')
  .first()
  .select('Map')
  .rename('landcover')
  .clip(jakarta);
print('Landcover OK:', landcover);

// POPULATION
var populationRaw = ee.ImageCollection('JRC/GHSL/P2023A/GHS_POP')
  .filterDate('2020-01-01', '2020-12-31')
  .first();
var population = populationRaw.select('population_count').rename('population').clip(jakarta);
print('Population OK:', population);

// ELEVATION
var elevation = ee.Image('USGS/SRTMGL1_003').clip(jakarta);
print('Elevation OK:', elevation);

// DISTANCE (simulasi)
var distToRoads = ee.Image.random().multiply(5000).clip(jakarta).rename('dist_to_roads');
var distToPolice = ee.Image.random().multiply(10000).clip(jakarta).rename('dist_to_police');
var distToSchools = ee.Image.random().multiply(8000).clip(jakarta).rename('dist_to_schools');

// 4. GABUNGKAN CIRI-CIRI
var features = ee.Image.cat([
  landcover,
  population,
  elevation.rename('elevation'),
  distToRoads,
  distToPolice,
  distToSchools
]);
print("Band Names:", features.bandNames());

// 5. DATA TRAINING
var trainingData = features.sampleRegions({
  collection: crimeData,
  properties: ['crime_type_id', 'severity'],
  scale: 30
});
print("Data Training (Sample 5):", trainingData.limit(5));

// 6. SPLIT TRAIN/VALIDASI
var withRandom = trainingData.randomColumn();
var train = withRandom.filter(ee.Filter.lt('random', 0.7));
var valid = withRandom.filter(ee.Filter.gte('random', 0.7));
print("Jumlah Data Training:", train.size());
print("Jumlah Data Validasi:", valid.size());

// 7. TRAINING MODEL
var decisionTree = ee.Classifier.smileCart().train({
  features: train,
  classProperty: 'crime_type_id',
  inputProperties: features.bandNames()
});
print("Model Decision Tree:", decisionTree.explain());

var randomForest = ee.Classifier.smileRandomForest(50).train({
  features: train,
  classProperty: 'crime_type_id',
  inputProperties: features.bandNames()
});
print("Model Random Forest:", randomForest.explain());

// 8. PREDIKSI KLASIFIKASI
var predCrime = features.classify(randomForest);
var severityModel = ee.Classifier.smileRandomForest(50).train({
  features: train,
  classProperty: 'severity',
  inputProperties: features.bandNames()
});
var predSeverity = features.classify(severityModel);

// 9. TAMPILKAN DI MAP
Map.addLayer(crimeData, {color: 'red'}, 'Data Kejahatan');

var crimePalette = ['#D50000', '#FF6D00', '#FFD600', '#00C853', '#2979FF'];
Map.addLayer(predCrime, {min: 0, max: 4, palette: crimePalette}, 'Prediksi Jenis Kejahatan');

var severityPalette = ['#C8E6C9', '#81C784', '#4CAF50', '#2E7D32', '#1B5E20'];
Map.addLayer(predSeverity, {min: 1, max: 5, palette: severityPalette}, 'Prediksi Tingkat Keparahan');

// === LEGEND JENIS KEJAHATAN ===
var legendCrime = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});
legendCrime.add(ui.Label({
  value: 'Legenda Jenis Kejahatan',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}
}));

var crimeLabels = ['Pencurian', 'Perampokan', 'Vandalisme', 'Penipuan', 'Kekerasan'];
var crimeColors = ['#D50000', '#FF6D00', '#FFD600', '#00C853', '#2979FF'];

for (var i = 0; i < crimeLabels.length; i++) {
  legendCrime.add(ui.Panel([
    ui.Label({
      style: {
        backgroundColor: crimeColors[i],
        padding: '8px',
        margin: '0 8px 4px 0'
      }
    }),
    ui.Label(crimeLabels[i])
  ], ui.Panel.Layout.Flow('horizontal')));
}
Map.add(legendCrime);


// === LEGEND TINGKAT KEPARAHAN ===
var legendSeverity = ui.Panel({
  style: {
    position: 'bottom-right',
    padding: '8px 15px'
  }
});
legendSeverity.add(ui.Label({
  value: 'Legenda Tingkat Keparahan',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}
}));

var severityLabels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'];
var severityColors = ['#C8E6C9', '#81C784', '#4CAF50', '#2E7D32', '#1B5E20'];

for (var j = 0; j < severityLabels.length; j++) {
  legendSeverity.add(ui.Panel([
    ui.Label({
      style: {
        backgroundColor: severityColors[j],
        padding: '8px',
        margin: '0 8px 4px 0'
      }
    }),
    ui.Label(severityLabels[j])
  ], ui.Panel.Layout.Flow('horizontal')));
}
Map.add(legendSeverity);

