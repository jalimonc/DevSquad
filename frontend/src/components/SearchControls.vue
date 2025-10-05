<template>
  <div class="controls-wrapper" @click.stop>
    <!-- Input Fecha (cápsula directa) -->
    <div class="control-item">
      <input
        v-model="selectedDate"
        type="date"
        :min="minDate"
        :max="maxDate"
        @change="onDateChange"
        class="capsule-btn date-btn"
        placeholder="Fecha"
        ref="dateInput"
      />
    </div>

    <!-- Botón Hora (cápsula) -->
    <div class="control-item">
      <button @click="toggleTimePicker" class="capsule-btn date -btn">
        {{ selectedTime ? selectedTime.padStart(2, '0') + ':00' : 'Hora' }}
      </button>
      <!-- Picker de hora (toggle) -->
      <div v-if="showTimePicker" class="time-picker" ref="timePickerRef">
        <!-- Botones + y - arriba/abajo -->
        <button @click="incrementHour" class="time-btn time-btn-up" :disabled="isMaxHour">+</button>
        
        <!-- Rueda horizontal de horas con selected en medio -->
        <div class="wheel-container">
          <div 
            ref="hourWheel"
            class="hour-wheel"
            @wheel="handleWheel"
            @scroll="handleScroll"
          >
            <div
              v-for="hour in 24"
              :key="hour"
              :class="['hour-option', { disabled: isHourDisabled(hour) }]"
              @click="selectHour(hour)"
            >
              {{ hour.toString().padStart(2, '0') }}:00
            </div>
          </div>
          <!-- Selected time centrado sobre la rueda -->
          <div class="selected-time-overlay">{{ selectedTime ? selectedTime.padStart(2, '0') + ':00' : '--:--' }}</div>
        </div>
        
        <button @click="decrementHour" class="time-btn time-btn-down" :disabled="isMinHour">-</button>
      </div>
    </div>

    <!-- Botón Consultar (cápsula azul) -->
    <div class="control-item">
      <button @click="handleConsult" class="capsule-btn consult-btn" :disabled="!isValid">
        Consultar
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue';

const props = defineProps({
  lat: { type: Number, default: null },
  lng: { type: Number, default: null }
});

const emit = defineEmits(['submit']);

const selectedDate = ref(''); // Valor inicial vacío
const selectedTime = ref(''); // Hora seleccionada (ej: '12')
const showTimePicker = ref(false); // Toggle para hora

// Hoy: dinámico a fecha actual
const today = ref(new Date().toISOString().split('T')[0]);
const currentHour = ref(new Date().getHours()); // Hora actual para disable
const minDate = computed(() => today.value); // Min: hoy
const maxDate = computed(() => {
  const todayDate = new Date(today.value);
  todayDate.setMonth(todayDate.getMonth() + 12); // +12 meses
  return todayDate.toISOString().split('T')[0];
});

const hourWheel = ref(null); // Ref para el div de rueda
const dateInput = ref(null); // Ref para input date (ahora directo)
const timePickerRef = ref(null); // Ref para picker de hora (para click fuera)
let scrollTimeout = null; // Para debounce en handleScroll

// Listener global para cerrar pickers al click fuera (solo hora ahora)
const handleClickOutside = (event) => {
  if (showTimePicker.value && timePickerRef.value && !timePickerRef.value.contains(event.target)) {
    showTimePicker.value = false;
  }
};

// Al cambiar fecha, actualiza si es hoy para disable horas
const onDateChange = (event) => {
  console.log('Fecha seleccionada:', event.target.value);
  if (event.target.value === today.value) {
    // Si es hoy, disable horas pasadas
    currentHour.value = new Date().getHours();
    // Opcional: si selectedTime es pasada, ajusta a actual
    if (parseInt(selectedTime.value || '0') < currentHour.value) {
      selectedTime.value = currentHour.value.toString();
    }
  } else {
    // Si es futuro, todas las horas disponibles
    currentHour.value = 0;
  }
};

// Toggle picker hora (cierra fecha si abierto - pero fecha ya no tiene toggle)
const toggleTimePicker = () => {
  showTimePicker.value = !showTimePicker.value;
  if (showTimePicker.value) {
    nextTick(() => {
      // Siempre posiciona la rueda en la hora actual seleccionada (o inicial si no hay)
      let hour = selectedTime.value ? parseInt(selectedTime.value) : parseInt(currentHour.value);
      selectedTime.value = hour.toString();
      if (hourWheel.value) {
        scrollToHour(hour, true); // true para 'auto' en inicial
      }
    });
  }
};

// Disable hora si es hoy y hora < actual
const isHourDisabled = (hour) => {
  return selectedDate.value === today.value && hour < currentHour.value;
};

// Min/Max para botones
const isMinHour = computed(() => parseInt(selectedTime.value || '0') <= (selectedDate.value === today.value ? currentHour.value : 0));
const isMaxHour = computed(() => parseInt(selectedTime.value || '23') >= 23);

// Validar para disable botón consultar
const isValid = computed(() => !!selectedDate.value && !!selectedTime.value && props.lat !== null && props.lng !== null);

// Seleccionar hora al clic (NO cierra picker)
const selectHour = (hour) => {
  if (isHourDisabled(hour)) return; // No selecciona si disabled
  selectedTime.value = hour.toString();
  console.log('Hora seleccionada:', selectedTime.value);
  // Scroll suave a la seleccionada
  scrollToHour(hour);
  // NO cerrar aquí, solo en click fuera
};

// Scroll a una hora específica
const scrollToHour = (hour, instant = false) => {
  if (hourWheel.value) {
    const optionWidth = 80; // Ancho aproximado de cada .hour-option
    const behavior = instant ? 'auto' : 'smooth';
    hourWheel.value.scrollTo({
      left: hour * optionWidth,
      behavior
    });
  }
};

// Incrementar hora (NO cierra picker)
const incrementHour = () => {
  const nextHour = parseInt(selectedTime.value || '0') + 1;
  if (nextHour < 24 && !isHourDisabled(nextHour)) {
    selectHour(nextHour);
  }
};

// Decrementar hora (NO cierra picker)
const decrementHour = () => {
  const prevHour = parseInt(selectedTime.value || '23') - 1;
  if (prevHour >= (selectedDate.value === today.value ? currentHour.value : 0) && !isHourDisabled(prevHour)) {
    selectHour(prevHour);
  }
};

// Manejar rueda del mouse (scroll horizontal)
const handleWheel = (event) => {
  event.preventDefault();
  const delta = event.deltaY > 0 ? 1 : -1;
  const currentScroll = hourWheel.value.scrollLeft;
  const optionWidth = 80;
  hourWheel.value.scrollLeft = currentScroll + (delta * optionWidth);
};

// Manejar scroll (con debounce para evitar resets durante animación)
const handleScroll = () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(() => {
    // Snap to nearest hour y actualiza selectedTime
    const scrollLeft = hourWheel.value.scrollLeft;
    const optionWidth = 80;
    const hour = Math.round(scrollLeft / optionWidth);
    if (hour >= 0 && hour < 24 && !isHourDisabled(hour)) {
      selectedTime.value = hour.toString();
      // Scroll suave al snapped para alinear
      const snapped = hour * optionWidth;
      hourWheel.value.scrollTo({ left: snapped, behavior: 'smooth' });
    }
  }, 100); // Espera 100ms después del último evento scroll
};

// Formatear fecha para botón (ya no se usa, pero lo dejo por si acaso)
const formatDate = (dateStr) => {
  if (!dateStr) return 'Fecha';
  const date = new Date(dateStr);
  return date.toLocaleDateString('es-ES'); // dd/mm/yyyy
};

// Handle consultar
const handleConsult = () => {
  if (!isValid.value) return;
  
  // Convertir selectedDate (string) a objeto Date
  const date = new Date(selectedDate.value);
  
  if (isNaN(date.getTime())) {
    console.error('selectedDate no es una fecha válida:', selectedDate.value);
    return; // Maneja el error como prefieras (ej: alert o toast)
  }
  
  const payload = {
    lat: props.lat,
    long: props.lng,
    day: date.getDate(),  // Día del mes (1-31)
    month: date.getMonth() + 1,  // Mes (1-12, ya que getMonth() es 0-11)
    year: date.getFullYear(),  // Año completo
    hour: selectedTime.value
  };
  console.log('Enviando al backend:', payload);
  emit('submit', payload);
  // Aquí: fetch('/api/clima', { method: 'POST', body: JSON.stringify(payload) });
};

onMounted(() => {
  // Inicial: fecha hoy, hora actual
  selectedDate.value = today.value;
  selectedTime.value = currentHour.value.toString();
  // Agrega listener global para click fuera
  document.addEventListener('click', handleClickOutside);
});

onUnmounted(() => {
  // Limpia listener y timeout
  document.removeEventListener('click', handleClickOutside);
  if (scrollTimeout) clearTimeout(scrollTimeout);
});
</script>

<style scoped>
.controls-wrapper {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
  background: transparent; /* Sin fondo para minimalista */
  padding: 10px;
  width: fit-content;
  margin-top: 10px; /* Espacio con search-bar */
}

.control-item {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.capsule-btn {
  border: none;
  padding: 10px 20px;
  font-size: 14px;
  font-weight: bold;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.2s;
  min-width: 80px;
  text-align: center;
  box-sizing: border-box;
}

.date-btn,
.time-btn {
  background: #f8f9fa;
  color: #333;
}

.date-btn:hover,
.time-btn:hover {
  background: #e9ecef;
  transform: scale(1.05);
}

/* Estilos específicos para input type="date" como botón */
.date-btn[type="date"] {
  border: 1px solid #ddd; /* Agrega borde sutil para simular botón */
  background: #f8f9fa;
  color: #333;
}

.date-btn[type="date"]:focus {
  outline: none;
  box-shadow: 0 0 0 2px rgba(3, 157, 191, 0.2); /* Azul sutil en focus */
}

.date-btn[type="date"]::-webkit-calendar-picker-indicator {
  cursor: pointer;
  opacity: 0.7; /* Hace visible el icono de calendario si el navegador lo muestra */
}

.consult-btn {
  background: #039dbf;
  color: white;
}

.consult-btn:hover:not(:disabled) {
  background: #0284a5;
  transform: scale(1.05);
}

.consult-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.time-picker {
  position: absolute;
  top: 100%;
  margin-top: 5px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 5px;
  background: white;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 25px; /* Cápsula redondeada */
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 10;
  min-width: 120px; /* Ajustado para cápsula */
}

/* Estilos para picker de hora (sin cambios) */
.time-btn {
  width: 30px;
  height: 30px;
  border: 1px solid #ddd;
  background: white;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  font-weight: bold;
  transition: background 0.2s;
}

.time-btn:hover:not(:disabled) {
  background: #f0f0f0;
}

.time-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.time-btn-up {
  margin-bottom: 5px;
}

.time-btn-down {
  margin-top: 5px;
}

.wheel-container {
  position: relative;
  width: 80px;
  height: 50px;
}

.hour-wheel {
  display: flex;
  overflow-x: auto;
  overflow-y: hidden;
  width: 100%;
  height: 100%;
  border: 1px solid #ddd;
  border-radius: 8px;
  scroll-snap-type: x mandatory;
  scrollbar-width: none; /* Oculta scrollbar en Firefox */
}

.hour-wheel::-webkit-scrollbar {
  display: none; /* Oculta en Chrome */
}

.hour-option {
  flex: 0 0 80px; /* Ancho fijo por hora */
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: bold;
  scroll-snap-align: center;
  cursor: pointer;
  transition: background 0.2s;
  min-height: 50px;
}

.hour-option:hover:not(.disabled) {
  background: #f8f9fa;
}

.hour-option.disabled {
  opacity: 0.5;
  cursor: not-allowed;
  color: #999;
}

/* Selected time centrado en la rueda */
.selected-time-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(255, 255, 255, 0.9);
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  color: #333;
  pointer-events: none; /* No interfiere con scroll/clics */
  z-index: 1;
}

.controls-wrapper {
  position: absolute;
  bottom: 180px; /* Aumentado para debajo del buscador */
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
}
</style>