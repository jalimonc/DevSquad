// services/service.js

export const postPredict = async (lat, long, day, month, year, hour = undefined) => {
  const requestBody = {
    lat: Number(lat),
    long: Number(long),
    day: Number(day),
    month: Number(month),
    year: Number(year),
    ...(hour !== undefined && { hour: Number(hour) })
  };

  try {
    const response = await fetch('http://localhost:3000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`Error en la respuesta: ${response.status}`);
    }

    const data = await response.json();
    console.log('Respuesta del backend:', data);
    return data;
  } catch (error) {
    console.error('Error al hacer la petición:', error);
    throw error;
  }
};