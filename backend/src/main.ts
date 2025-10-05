import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // Habilita CORS para permitir requests desde el frontend (temporal para dev)
  app.enableCors({
    origin: '*',  // O 'http://localhost:5173' para solo tu Vue app
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
    credentials: true,
  });
  
  await app.listen(3000);
}
bootstrap();