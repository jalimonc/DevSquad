<template>
  <div class="map-wrap">
    <div class="map" ref="mapContainer"></div>
    <div v-if="coordinates" class="coords-display">
      Lat: {{ coordinates.lat.toFixed(4) }} | Lng: {{ coordinates.lng.toFixed(4) }}
    </div>
  </div>
</template>

<script setup>
import { Map, MapStyle, Marker, config, NavigationControl, GeolocateControl } from '@maptiler/sdk';
import { shallowRef, ref, onMounted, onUnmounted, markRaw } from 'vue';
import '@maptiler/sdk/dist/maptiler-sdk.css';

// Clase custom para el control de estilos (sin cambios)
class StyleControl {
  constructor({ position = 'top-right' } = {}) {
    this.position = position;
    this.container = document.createElement('div');
    this.container.className = 'maplibregl-ctrl maplibregl-ctrl-group maptiler-style-ctrl';
    this.container.style.display = 'flex';
    this.container.style.flexDirection = 'column';
    this.container.style.gap = '4px';

    // Botón principal
    const toggleBtn = document.createElement('button');
    toggleBtn.className = 'maplibregl-ctrl-group__button';
    toggleBtn.type = 'button';
    toggleBtn.setAttribute('aria-label', 'Cambiar estilo del mapa');
    toggleBtn.innerHTML = '🗺️';

    // Dropdown (sin cambios)
    const dropdown = document.createElement('div');
    dropdown.className = 'maptiler-style-dropdown';
    dropdown.style.display = 'none';
    dropdown.style.background = 'white';
    dropdown.style.border = '1px solid #ccc';
    dropdown.style.borderRadius = '4px';
    dropdown.style.padding = '4px';
    dropdown.style.boxShadow = '0 2px 8px rgba(0,0,0,0.2)';
    dropdown.style.zIndex = '10';
    dropdown.style.position = 'absolute';
    dropdown.style.top = '100%';
    dropdown.style.right = '0';
    dropdown.style.minWidth = '120px';

    const options = [
      { label: 'Callejero', style: 'streets' },
      { label: 'Satélite', style: 'satellite' },
      { label: 'Híbrido', style: 'hybrid' },
      { label: '3D Exterior', style: 'outdoor' }
    ];

    options.forEach(opt => {
      const btn = document.createElement('button');
      btn.innerHTML = opt.label;
      btn.style.width = '100%';
      btn.style.padding = '4px 8px';
      btn.style.border = 'none';
      btn.style.background = 'none';
      btn.style.textAlign = 'left';
      btn.style.cursor = 'pointer';
      btn.style.borderRadius = '2px';
      btn.onmouseover = () => btn.style.background = '#f0f0f0';
      btn.onmouseout = () => btn.style.background = 'none';
      btn.onclick = () => {
        this.onStyleChange(opt.style);
        dropdown.style.display = 'none';
      };
      dropdown.appendChild(btn);
    });

    let isOpen = false;
    toggleBtn.onclick = (e) => {
      e.stopPropagation();
      isOpen = !isOpen;
      dropdown.style.display = isOpen ? 'block' : 'none';
    };

    document.addEventListener('click', () => {
      dropdown.style.display = 'none';
      isOpen = false;
    });

    this.container.appendChild(toggleBtn);
    this.container.appendChild(dropdown);
  }

  onAdd(map) {
    this._map = map;
    this._container = this.container;
    return this.container;
  }

  onRemove() {
    if (this.container.parentNode) {
      this.container.parentNode.removeChild(this.container);
    }
  }

  onStyleChange(styleName) {
    window.changeStyle(styleName);
  }
}

const mapContainer = shallowRef(null);
const map = shallowRef(null);
const marker = shallowRef(null);
const coordinates = ref(null);
const currentStyle = ref(MapStyle.STREETS);
let navControl = null;
let geoControl = null;
let styleControl = null;
const isControlsAdded = ref(false); // Flag para agregar solo una vez

onMounted(() => {
  config.apiKey = 'rMmvTAYvHTo4a2k2vxXZ';
  
  const initialState = { lng: -65.25959686357317, lat:  -19.04775511516892, zoom: 14 };

  map.value = markRaw(new Map({
    container: mapContainer.value,
    style: currentStyle.value,
    center: [initialState.lng, initialState.lat],
    zoom: initialState.zoom,
    // Desactiva los controles por default para evitar duplicados
    navigationControl: false,
    geolocateControl: false,
    attributionControl: true // Mantiene el attribution por créditos
  }));

  // Agrega controles solo una vez
  if (!isControlsAdded.value) {
    // Navigation (zoom + compass)
    navControl = new NavigationControl({ position: 'top-right' });
    map.value.addControl(navControl);
    
    // Geolocate (tu ubicación)
    geoControl = new GeolocateControl({ position: 'top-right', trackUserLocation: true });
    map.value.addControl(geoControl);

    // Custom style control (al final, para que quede abajo)
    styleControl = new StyleControl({ position: 'top-right' });
    map.value.addControl(styleControl);

    isControlsAdded.value = true;
    console.log('Controles agregados una sola vez');
  }

  // Marcador inicial
  marker.value = new Marker({ color: "#FF0000" })
    .setLngLat([139.7525, 35.6846])
    .addTo(map.value);

  // Listener de click
  map.value.on('click', (e) => {
    const { lng, lat } = e.lngLat;
    coordinates.value = { lat, lng };

    if (marker.value) {
      marker.value.setLngLat([lng, lat]);
    } else {
      marker.value = new Marker({ color: "#FF0000" })
        .setLngLat([lng, lat])
        .addTo(map.value);
    }

    console.log('Coordenadas seleccionadas:', { lat, lng });
  });
});

// Función para cambiar estilo (expuesta global)
window.changeStyle = (styleName) => {
  let newStyle;
  switch (styleName) {
    case 'streets':
      newStyle = MapStyle.STREETS;
      break;
    case 'satellite':
      newStyle = MapStyle.SATELLITE;
      break;
    case 'hybrid':
      newStyle = MapStyle.HYBRID;
      break;
    case 'outdoor':
      newStyle = MapStyle.OUTDOOR;
      break;
    default:
      newStyle = MapStyle.STREETS;
  }
  
  if (map.value && newStyle !== currentStyle.value) {
    map.value.setStyle(newStyle);
    currentStyle.value = newStyle;
    console.log(`Estilo cambiado a: ${styleName}`);
  }
};

onUnmounted(() => {
  if (map.value) {
    map.value.remove();
  }
  delete window.changeStyle;
});
</script>

<style scoped>
.map-wrap {
  position: relative;
  width: 100%;
  height: calc(100vh - 77px);
}

.map {
  position: absolute;
  width: 100%;
  height: 100%;
}

.coords-display {
  position: absolute;
  top: 10px;
  left: 10px;
  background: white;
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  z-index: 1000;
}

.maptiler-style-dropdown button:hover {
  background: rgba(0, 0, 0, 0.05) !important;
}
</style>