// Нагрузочный тест GeoSeg (k6).
//  sync   — загрузка снимка 512×512 с синхронной сегментацией (force=true: каждый раз полный пересчёт);
//  async  — загрузка в асинхронном режиме: raster.ingested → tiler → inference-worker ×2 → geostore (Kafka);
//  query  — запросы полигонов из PostGIS (GeoJSON по классу).
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8000';
const TIF = open((__ENV.SCRIPTS_DIR || '/scripts') + '/sample.tif', 'b');

export const options = {
  scenarios: {
    sync: { executor: 'ramping-vus', exec: 'sync', startVUs: 1,
            stages: [{ target: 2, duration: '1m' }, { target: 4, duration: '1m' }, { target: 0, duration: '20s' }] },
    async: { executor: 'constant-arrival-rate', exec: 'asyncUpload', rate: 12, timeUnit: '1m', duration: '2m20s',
             preAllocatedVUs: 4, maxVUs: 10 },
    query: { executor: 'constant-arrival-rate', exec: 'query', rate: 10, timeUnit: '1s', duration: '2m20s',
             preAllocatedVUs: 5, maxVUs: 20 },
  },
  thresholds: { 'http_req_failed': ['rate<0.01'], 'http_req_duration{scenario:query}': ['p(95)<1000'] },
};

const form = () => ({ file: http.file(TIF, 'sample.tif', 'image/tiff') });

export function sync() {
  const r = http.post(`${API}/v1/rasters?mode=sync&force=true`, form(), { tags: { name: 'POST /v1/rasters sync' }, timeout: '120s' });
  check(r, { 'sync 201': (x) => x.status === 201 });
}

export function asyncUpload() {
  const r = http.post(`${API}/v1/rasters?mode=async&force=true`, form(), { tags: { name: 'POST /v1/rasters async' } });
  check(r, { 'async 201': (x) => x.status === 201 });
}

export function query() {
  const cls = ['forest', 'water', 'building', 'road', 'farmland'][Math.floor(Math.random() * 5)];
  const r = http.get(`${API}/v1/landcover?cls=${cls}&limit=50`, { tags: { name: 'GET /v1/landcover' } });
  check(r, { 'geojson 200': (x) => x.status === 200 });
}
