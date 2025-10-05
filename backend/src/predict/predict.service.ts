import { Injectable } from '@nestjs/common';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class PredictService {

    async findDates(lat: number, long: number, day: number, month: number, year: number, hour?: number) {
        const startYear = year - 23;
        const currentYear = new Date().getFullYear();
        const mm = String(month).padStart(2, '0');

        const temporal = 'hourly';

        const parameters = [
            'T2M',
            'PRECTOTCORR',
            'PS',
            'RH2M',
            'WS2M',
            'ALLSKY_SFC_SW_DWN',
            'CLOUD_AMT',
        ].join(',');

        const base = 'https://power.larc.nasa.gov/api/temporal';

        const years: number[] = [];
        for (let y = startYear; y <= currentYear; y++) years.push(y);

        try {
            const requests = years.map(y => {
                const daysInMonth = new Date(y, month, 0).getDate();
                const start = `${y}${mm}01`;
                const end = `${y}${mm}${String(daysInMonth).padStart(2, '0')}`;
                const url = `${base}/${temporal}/point?parameters=${parameters}&community=RE&longitude=${long}&latitude=${lat}&start=${start}&end=${end}&format=JSON`;
                return axios.get(url).then(r => ({ year: y, start, end, data: r.data }));
            });

            const responses = await Promise.all(requests);

            const final: Record<string, any> = {};

            let stopProcessing = false;
            for (const resp of responses) {
                if (stopProcessing) break;
                const y = resp.year;
                const paramObj = resp.data?.properties?.parameter || {};

                const yearResult: Record<string, { values: Array<{ ts: string; value: number | null }>; avg?: number | null; min?: number | null; max?: number | null }> = {};
                for (const [paramKey, pObj] of Object.entries(paramObj)) {
                    const entries = Object.entries(pObj || {}).filter(([ts, val]) => typeof ts === 'string' && /^\d{6,12}$/.test(ts) && val !== null);

                    // Filtrar por mes mm (asegurar que corresponda al mes solicitado)
                    const filtered = entries.filter(([ts, val]) => {
                        const m = ts.slice(4, 6);
                        return m === mm;
                    });

                    // Si algún valor filtrado es exactamente -999, no tomamos en cuenta este mes y paramos
                    const hasBad = filtered.some(([ts, val]) => Number(val) === -999);
                    if (hasBad) {
                        stopProcessing = true;
                        break; // salir del loop de parámetros, no añadimos este año
                    }

                    const numericValues = filtered.map(([ts, val]) => Number(val)).filter(v => !isNaN(v));

                    yearResult[paramKey] = {
                        values: filtered.map(([ts, val]) => ({ ts, value: val === null ? null : Number(val) })),
                        avg: numericValues.length ? numericValues.reduce((a, b) => a + b, 0) / numericValues.length : null,
                        min: numericValues.length ? Math.min(...numericValues) : null,
                        max: numericValues.length ? Math.max(...numericValues) : null,
                    };
                }

                if (stopProcessing) break; // no añadimos este año ni procesamos posteriores

                final[String(y)] = {
                    start: resp.start,
                    end: resp.end,
                    data: yearResult,
                };
            }

            // --- Generar un solo CSV con el nombre recibido desde el frontend ---
            // El nombre del archivo será el valor de day, month, year, hour concatenados (ejemplo: 2023010112.csv)
            // El contenido tendrá todas las filas de todos los timestamps y parámetros

            const outputDir = path.join(process.cwd(), 'python', 'datasets');
            if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
            }

            const paramList = [
                'T2M',
                'PRECTOTCORR',
                'PS',
                'RH2M',
                'WS2M',
                'ALLSKY_SFC_SW_DWN',
                'CLOUD_AMT',
            ];

            // Map: timestamp -> { param: value, ... }
            const tsMap: Record<string, Record<string, number|null>> = {};

            for (const yearData of Object.values(final)) {
                const data = (yearData as any).data || {};
                for (const param of paramList) {
                    const paramVals = data[param]?.values || [];
                    for (const entry of paramVals) {
                        const ts = entry.ts;
                        if (!tsMap[ts]) tsMap[ts] = {};
                        tsMap[ts][param] = entry.value;
                    }
                }
            }

            // Ordenar los timestamps
            const sortedTimestamps = Object.keys(tsMap).sort();

            // Nombre del archivo: YYYYMMDDHH.csv (usando los datos recibidos)
            const fileName = `${year}${String(month).padStart(2, '0')}${String(day).padStart(2, '0')}${hour !== undefined ? String(hour).padStart(2, '0') : ''}.csv`;
            const filePath = path.join(outputDir, fileName);

            // Escribir el CSV
            const csvHeader = ['fecha', ...paramList].join(',') + '\n';
            let csvContent = csvHeader;
            for (const ts of sortedTimestamps) {
                const row = tsMap[ts];
                csvContent += [ts, ...paramList.map(p => row[p] !== undefined ? row[p] : '')].join(',') + '\n';
            }
            fs.writeFileSync(filePath, csvContent, 'utf8');

            return {
                requestedMonth: Number(mm),
                startYear,
                endYear: currentYear,
                temporal,
                years: final,
                csvFile: filePath,
                csvRows: sortedTimestamps.length,
            };
        } catch (error) {
            return { error: 'Error fetching data from NASA POWER', details: (error as any).message || error };
        }
    }

}
