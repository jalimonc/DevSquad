<template>
  <div class="heading">
    <button class="btn">Climate Prediction Platform</button>
  </div>
</template>

<script setup>

</script>

<style scoped>
.heading {
  margin: 0;
  padding: 10px 20px; /* Padding lateral para que el botón no toque los bordes */
  background-color: transparent;
  color: black;
  text-align: center;
  position: relative; /* Para z-index y superposición */
  z-index: 1000; /* Asegura que esté encima del mapa */
  border-bottom: none; /* Quita borde inferior si hay */ /* Sombra sutil para separar del mapa */
  
}

.btn {
  background: black; /* Gradiente para cápsula fancy */
  color: white;
  border: none;
  padding: 12px 30px; /* Padding interno para forma alargada */
  font-size: 18px;
  font-weight: bold;
  border-radius: 50px; /* Cápsula redondeada (ajusta a 30px si quieres menos curva) */
  cursor: pointer;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); /* Sombra para profundidad */
  transition: all 0.3s ease; /* Suave hover */
  width: 1000px; /* Ocupa todo el ancho del heading para que se vea como barra */
  max-width: 250px; /* Limita ancho para no estirarse demasiado */
  margin: 0 auto; /* Centrado */
}



/* Para que el mapa se "vea por los costados": Ajusta el height del mapa en el componente padre */
/* En Map.vue, cambia height: calc(100vh - 77px); a calc(100vh - 60px); o lo que mida este heading */
</style>