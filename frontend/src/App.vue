<template>
    <div>
        <Navbar />
        <Map />
        <button @click="testPredict">Probar Endpoint</button>
    </div>
</template>

<script setup>
import { ref } from 'vue';
import Map from './components/Map.vue';
import Navbar from './components/Navbar.vue';
import { postPredict } from './services/service.js'; // Ajusta la ruta si es necesario

const testPredict = async () => {
  console.log('¡Click detectado! Iniciando prueba con datos estáticos...'); // Agregado para debug: confirma si el click llega aquí
  try {
    const data = await postPredict('-19.047900', '-65.261380', 10, 7, 2026, 15);
    console.log('¡Éxito! Respuesta del backend:', data);
  } catch (error) {
    console.error('Error en la prueba:', error);
  }
};
</script>
    
<style lang="scss" scoped>
button {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 999;
  background: white;
  padding: 10px 15px;
  border: 1px solid #ccc;
  border-radius: 5px;
  cursor: pointer;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
}
</style>