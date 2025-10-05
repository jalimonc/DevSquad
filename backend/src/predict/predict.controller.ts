import { Controller, Post, Body, BadRequestException } from '@nestjs/common';
import { PredictService } from './predict.service';

class PredictDto {
	lat: number;
	long: number;
	day: number;
	month: number;
	year: number;
	hour?: number;
}

@Controller('predict')
export class PredictController {
	constructor(private readonly predictService: PredictService) {}

	@Post()
	async postPredict(@Body() body: PredictDto) {
		const { lat, long, day, month, year, hour } = body || {} as PredictDto;

		if (lat === undefined || long === undefined || day === undefined || month === undefined || year === undefined) {
			throw new BadRequestException('Faltan parámetros. Requiere lat, long, day, month, year');
		}

		const latNum = Number(lat);
		const longNum = Number(long);
		const dayNum = Number(day);
		const monthNum = Number(month);
		const yearNum = Number(year);
		const hourNum = hour !== undefined ? Number(hour) : undefined;

		if (isNaN(latNum) || isNaN(longNum) || isNaN(dayNum) || isNaN(monthNum) || isNaN(yearNum)) {
			throw new BadRequestException('Parámetros numéricos inválidos');
		}

		if (monthNum < 1 || monthNum > 12) throw new BadRequestException('month debe estar entre 1 y 12');
		if (dayNum < 1 || dayNum > 31) throw new BadRequestException('day inválido');
		if (hourNum !== undefined && (isNaN(hourNum) || hourNum < 0 || hourNum > 23)) throw new BadRequestException('hour inválido (0-23)');

		return await this.predictService.findDates(latNum, longNum, dayNum, monthNum, yearNum, hourNum);
	}

}
