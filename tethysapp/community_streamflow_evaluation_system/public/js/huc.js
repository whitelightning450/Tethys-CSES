// ----- CSRF helper (official Django recipe) -----
function getCookie(name) {
  const cookies = document.cookie.split(';');
  for (const c of cookies) {
    const [key, value] = c.trim().split('=');
    if (key === name) return decodeURIComponent(value);
  }
  return null;
}
const csrftoken = getCookie('csrftoken'); // same-origin only

const inject_map_data = (layer, metadata) => {
  layer.tethys_legend_title = metadata.legend_title;
  layer.tethys_legend_classes = metadata.legend_classes;
  layer.tethys_legend_extent = metadata.legend_extent;
  layer.tethys_legend_extent_projection = metadata.legend_extent_projection;
  layer.tethys_editable = metadata.editable;
  layer.tethys_data = metadata.data;
};

document.getElementById('huc-eval-form').addEventListener('submit', updateData);

function updateData(event) {
  event.preventDefault();

  // Show loading message
  const loadingDiv = document.querySelector('.loading-text');
  loadingDiv.innerHTML = "Updating data and layers...";
  loadingDiv.style.display = 'block';
  let start_date = document.getElementById("start-date").value;
  let end_date = document.getElementById("end-date").value;
  let huc_ids = document.getElementById("huc_ids").value;
  let model_id = document.getElementById("model_id").value;

  var data = new URLSearchParams();
  data.append('method', 'update_huc_eval_data');
  data.append('start_date', start_date);
  data.append('end_date', end_date);
  data.append('huc_ids', huc_ids);
  data.append('model_id', model_id);

  fetch('.', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'X-CSRFToken': csrftoken
    },
    body: data
  })
  .then(resp => resp.ok ? resp.json() : Promise.reject(resp))
  .then(data => {
    console.log('Server replied:', data);
    var olMap = TETHYS_MAP_VIEW.getMap();
    const mapProj = olMap.getView().getProjection();

    olMap.getLayers().forEach(layer => {
      if (layer instanceof ol.layer.Vector) {
        const features = new ol.format.GeoJSON().readFeatures(
          data.geojson,
          { dataProjection: 'EPSG:4326', featureProjection: mapProj }
        );
        const newSource = new ol.source.Vector({ features: features });
        layer.setSource(newSource);
        const extent = newSource.getExtent();
        inject_map_data(layer, data.metadata);
        if (!ol.extent.isEmpty(extent)) {
          olMap.getView().fit(extent, { padding: [40, 40, 40, 40], duration: 500 });
        }   
      }
    });
    loadingDiv.innerHTML = data.message;
  })
  
  .catch((err) => {
    loadingDiv.style.display = 'none';
    console.error('REST call failed:', err)}
  )
}