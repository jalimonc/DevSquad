<!-- components/Result.vue (actualizado para usar data.prediction y 3 columnas) -->
<template>
  <div class="result-modal">
    <header class="modal-header">
      <h2>Resultados del Clima</h2>
      <button @click="$emit('close')" class="close-btn">&times;</button>
    </header>
    
    <div v-if="data && data.prediction && Object.keys(data.prediction).length > 0" class="result-content">
      <!-- Formulario no editable (display-only) en 3 columnas -->
      <div class="form-grid">
        <div class="form-field">
          <label>Fecha:</label>
          <span class="form-value">{{ formatDate(data.prediction.fecha) }}</span>
        </div>
        
        <div class="form-field">
          <label>Temperatura (T2M):</label>
          <span class="form-value">{{ data.prediction.T2M }}°C</span>
        </div>
        
        <div class="form-field">
          <label>Precipitación (PRECTOTCORR):</label>
          <span class="form-value">{{ data.prediction.PRECTOTCORR }} mm</span>
        </div>
        
        <div class="form-field">
          <label>Presión (PS):</label>
          <span class="form-value">{{ data.prediction.PS }} hPa</span>
        </div>
        
        <div class="form-field">
          <label>Humedad (RH2M):</label>
          <span class="form-value">{{ data.prediction.RH2M }}%</span>
        </div>
        
        <div class="form-field">
          <label>Viento (WS2M):</label>
          <span class="form-value">{{ data.prediction.WS2M }} m/s</span>
        </div>
        
        <div class="form-field">
          <label>Radiación Solar (ALLSKY_SFC_SW_DWN):</label>
          <span class="form-value">{{ data.prediction.ALLSKY_SFC_SW_DWN }} W/m²</span>
        </div>
        
        <div class="form-field">
          <label>Nubosidad (CLOUD_AMT):</label>
          <span class="form-value">{{ data.prediction.CLOUD_AMT }}%</span>
        </div>
      </div>
      
      <!-- Opcional: Info adicional del response (ej: período de predicción) -->
      <div class="info-section" v-if="data.startYear && data.endYear">
        <p>El día se presentará cálido y mayormente soleado, con una temperatura aproximada de 25.8 °C. Se esperan vientos moderados 
          que alcanzarán hasta 20 km/h, lo que dará una sensación de frescura durante la jornada. La humedad se mantendrá baja, por lo que el ambiente será seco.
          Aunque el cielo estará con poca nubosidad, existe la probabilidad de lluvias ligeras en algunos sectores. La radiación solar será alta, por lo que se 
          recomienda protegerse del sol durante las horas de mayor intensidad.</p>
      </div>
    </div>
    
    <div v-else class="loading">Cargando resultados... o datos vacíos</div>
    
    <footer class="modal-footer">
      <button @click="$emit('close')" class="close-btn-footer">Cerrar</button>
    </footer>
  </div>
</template>

<script setup>
const props = defineProps({
  data: Object
});

const emit = defineEmits(['close']);

// Debug: Log en consola cuando llega data al componente
console.log('Data recibida en Result.vue:', props.data);

// Formatear fecha YYYYMMDDHH a legible (ej: 2025101011 -> 10/10/2025 11:00)
const formatDate = (fechaStr) => {
  if (!fechaStr || fechaStr.length !== 10) return 'N/A';
  const year = fechaStr.substring(0, 4);
  const month = fechaStr.substring(4, 6);
  const day = fechaStr.substring(6, 8);
  const hour = fechaStr.substring(8, 10);
  return `${day}/${month}/${year} ${hour}:00`;
};
</script>

<style scoped>
.result-modal {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px 16px;
  border-bottom: 1px solid #e5e7eb;
  background: linear-gradient(135deg, #039dbf 0%, #0284a5 100%);
  color: white;
  border-radius: 12px 12px 0 0;
}

.modal-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background 0.2s;
}

.close-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.result-content {
  padding: 24px;
  flex: 1;
  overflow-y: auto;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr); /* 3 columnas */
  gap: 16px;
  margin-bottom: 16px;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  background: #f9fafb;
  border-radius: 8px;
  border-left: 4px solid #039dbf;
}

.form-field label {
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.form-value {
  font-size: 16px;
  font-weight: 600;
  color: #111827;
  background: white;
  padding: 8px 12px;
  border-radius: 6px;
  border: 1px solid #d1d5db;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.info-section {
  background: #e3f2fd;
  padding: 12px;
  border-radius: 8px;
  border-left: 4px solid #0284a5;
  font-size: 14px;
  color: #0c4a6e;
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  font-size: 16px;
  color: #6b7280;
}

.modal-footer {
  padding: 16px 24px 24px;
  border-top: 1px solid #e5e7eb;
  text-align: right;
  background: #f9fafb;
  border-radius: 0 0 12px 12px;
}

.close-btn-footer {
  background: #039dbf;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.close-btn-footer:hover {
  background: #0284a5;
}

/* Responsive: En móviles, baja a 1 columna */
@media (max-width: 768px) {
  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>