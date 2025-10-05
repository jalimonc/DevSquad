import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { PredictModule } from './predict/predict.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
    PredictModule
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
