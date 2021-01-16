// creating map
var mymap = L.map('mapid').setView([52.06883124080639, 19.479736645844262], 5);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 18,
  tileSize: 256,
  zoomOffset: 0,
  }).addTo(mymap);

// adding marker to map
var marker = L.marker();

function onMapClick(e) {
    marker
        .setLatLng(e.latlng)
        // .setContent("You clicked the map at " + e.latlng.toString())
        .addTo(mymap);
    $('#lat')
        .val(e.latlng.lat.toString())
    $('#lng')
        .val(e.latlng.lng.toString())
}

mymap.on('click', onMapClick);
