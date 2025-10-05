
<template>
  <div class="map-wrap">
    <!-- Barra de búsqueda abajo centro -->
    <div class="search-bar">
      
      <input
        v-model="searchQuery"
        @input="debounceSearch"
        @keydown.enter="performSearch"
        @focus="showSuggestions"
        @blur="hideSuggestions"
        placeholder="Busca un lugar"
        type="text"
      />
      
      <button @click="performSearch" class="search-btn">🔍</button>
      <SearchControls :lat="coordinates?.lat" :lng="coordinates?.lng" @submit="onSubmit" />
      
      <!-- Dropdown de sugerencias -->
      <ul v-if="showDropdown && suggestions.length > 0" class="suggestions-dropdown">
        <li
          v-for="(suggestion, index) in suggestions"
          :key="index"
          @click="selectSuggestion(suggestion)"
          class="suggestion-item"
        >
          {{ suggestion.display_name }}
        </li>
      </ul>
    </div>
   
    <div class="map" ref="mapContainer"></div>

    <div v-if="coordinates" class="coords-display">
      Lat: {{ coordinates.lat.toFixed(4) }} | Lng: {{ coordinates.lng.toFixed(4) }}
    </div>
  </div>
</template>

<script setup>
import { Map, MapStyle, Marker, config, NavigationControl, GeolocateControl } from '@maptiler/sdk';
import { shallowRef, ref, onMounted, onUnmounted, markRaw, nextTick } from 'vue';
import '@maptiler/sdk/dist/maptiler-sdk.css';
import SearchControls from './SearchControls.vue';

const onSubmit = (payload) => {
  console.log('Payload para backend:', payload);
  // fetch('/api/clima', { method: 'POST', body: JSON.stringify(payload) });
};

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

// Para la búsqueda con Nominatim
const searchQuery = ref('');
const suggestions = ref([]); // Array de sugerencias
const showDropdown = ref(false);
const searchTimeout = ref(null); // Para debounce

// Debounce para input (evita búsquedas en cada tecla)
const debounceSearch = () => {
  if (searchTimeout.value) clearTimeout(searchTimeout.value);
  searchTimeout.value = setTimeout(() => {
    if (searchQuery.value.length > 2) {
      fetchSuggestions(); // Llama a sugerencias en lugar de búsqueda directa
    } else {
      suggestions.value = [];
      showDropdown.value = false;
    }
  }, 300); // 300ms para más responsivo
};

// Fetch sugerencias (usa limit=5 para dropdown)
const fetchSuggestions = async () => {
  const query = encodeURIComponent(searchQuery.value);
  const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=5&addressdetails=1`;

  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'ClimaDevSquat-Hackathon/1.0 (your.email@example.com)' // Requerido por Nominatim
      }
    });
    
    if (!response.ok) {
      throw new Error(`Error HTTP: ${response.status}`);
    }
    
    const data = await response.json();
    suggestions.value = data; // Array directo de Nominatim
    showDropdown.value = data.length > 0;
  } catch (error) {
    console.error('Error en sugerencias:', error);
    suggestions.value = [];
  }
};

// Mostrar/ocultar dropdown
const showSuggestions = () => {
  if (suggestions.value.length > 0) {
    showDropdown.value = true;
  }
};

const hideSuggestions = () => {
  // Delay para permitir clics
  setTimeout(() => {
    showDropdown.value = false;
  }, 200);
};

// Seleccionar sugerencia
const selectSuggestion = (suggestion) => {
  searchQuery.value = suggestion.display_name;
  showDropdown.value = false;

  // Realiza la acción inmediatamente
  const lat = parseFloat(suggestion.lat);
  const lng = parseFloat(suggestion.lon);
  map.value.setCenter([lng, lat]);
  map.value.setZoom(14);

  if (marker.value) {
    marker.value.setLngLat([lng, lat]);
  } else {
    marker.value = new Marker({ color: "#FF0000" })
      .setLngLat([lng, lat])
      .addTo(map.value);
  }

  coordinates.value = { lat, lng };
  console.log('Sugerencia seleccionada:', suggestion.display_name, { lat, lng });
};

const performSearch = async () => {
  if (!searchQuery.value.trim()) return;

  const query = encodeURIComponent(searchQuery.value);
  // URL de Nominatim (sin key, gratis)
  const url = `https://nominatim.openstreetmap.org/search?q=${query}&format=json&limit=1&addressdetails=1`;

  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'ClimaDevSquat-Hackathon/1.0 (your.email@example.com)' // Requerido por Nominatim
      }
    });
    
    // Chequea si la respuesta es OK
    if (!response.ok) {
      throw new Error(`Error HTTP: ${response.status} - ${response.statusText}`);
    }
    
    const data = await response.json();

    if (data && data.length > 0) {
      const result = data[0];
      const lat = parseFloat(result.lat);
      const lng = parseFloat(result.lon); // Nominatim: lon primero, pero lo usamos como lng

      // Actualiza centro del mapa
      map.value.setCenter([lng, lat]);
      map.value.setZoom(14); // Zoom fijo para búsqueda

      // Mueve el marcador
      if (marker.value) {
        marker.value.setLngLat([lng, lat]);
      } else {
        marker.value = new Marker({ color: "#FF0000" })
          .setLngLat([lng, lat])
          .addTo(map.value);
      }

      // Actualiza coords
      coordinates.value = { lat, lng };

      console.log('Lugar encontrado:', result.display_name, { lat, lng });
    } else {
      console.log('No se encontró el lugar:', searchQuery.value);
      // Opcional: Mostrar toast o alerta
    }
  } catch (error) {
    console.error('Error en búsqueda:', error);
    // Opcional: Muestra un mensaje al user, ej: alert('No se pudo buscar: ' + error.message);
  }
};

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

  // Marcador inicial (actualiza coords iniciales)
  marker.value = new Marker({ color: "#FF0000" })
    .setLngLat([initialState.lng, initialState.lat]) // Usa las coords iniciales
    .addTo(map.value);
  coordinates.value = { lat: initialState.lat, lng: initialState.lng }; // Inicializa coords

  // Listener de click (sin cambios)
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
  if (searchTimeout.value) clearTimeout(searchTimeout.value);
  delete window.changeStyle;
});
</script>

<style scoped>

.map-wrap {
  position: fixed; top: 0; left: 0; right: 0;
  width: 100%;
  height: calc(100vh - 5px);
  align-items: flex-end;
  
}

.map {
  position: absolute;
  width: 100%;
  height: 100%;
}

.search-bar {
  position: fixed;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  display: flex;
  flex-direction: row; /* Para que el dropdown vaya abajo del input */
  background: white;
  border-radius: 25px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  padding: 4px;
  width: 350px; /* Ajusta según necesites */
}

.search-bar input {
  flex: 1;
  border: none;
  padding: 12px 16px;
  font-size: 16px;
  outline: none;
  border-radius: 20px;
}

.search-btn {
  background: none;
  border: none;
  padding: 12px 16px;
  cursor: pointer;
  font-size: 18px;
  border-radius: 50%;
  transition: background 0.2s;
}

.search-btn:hover {
  background: #f0f0f0;
}

/* Estilos para dropdown de sugerencias */
.suggestions-dropdown {
  position: absolute;
  bottom: 100%; /* Lo pone arriba del input */
  left: 0;
  right: 0;
  margin: 0;
  padding: 0;
  list-style: none;
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  max-height: 200px;
  overflow-y: auto;
  z-index: 1001;
  margin-bottom: 4px; /* Espacio con el input */
}

.suggestion-item {
  padding: 12px 16px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background 0.2s;
}

.suggestion-item:hover {
  background: #f8f9fa;
}

.suggestion-item:last-child {
  border-bottom: none;
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
.controls-wrapper { /* Clase del nuevo comp */
  position: absolute;
  bottom: 120px; /* Debajo de search-bar (20px + altura) */
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
}
</style>