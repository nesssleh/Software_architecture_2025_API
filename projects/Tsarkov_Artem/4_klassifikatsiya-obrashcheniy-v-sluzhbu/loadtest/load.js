// Нагрузочный тест TicketFlow (k6).
//  async  — поток обращений: API → Kafka tickets.created → classifier-service ×2 → tickets.classified → routing-service
//           (рост 10 → 80 тикетов/с);
//  sync   — синхронное создание тикета (классификация + маршрутизация в API);
//  search — поиск похожих обращений в Elasticsearch.
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8000';
const DATA = JSON.parse(open((__ENV.SCRIPTS_DIR || '/scripts') + '/data.json'));
const H = { headers: { 'Content-Type': 'application/json' } };
const rnd = (a) => a[Math.floor(Math.random() * a.length)];

export const options = {
  scenarios: {
    async: { executor: 'ramping-arrival-rate', exec: 'asyncTicket', startRate: 10, timeUnit: '1s', preAllocatedVUs: 20, maxVUs: 150,
             stages: [{ target: 40, duration: '1m' }, { target: 80, duration: '1m' }, { target: 80, duration: '40s' },
                      { target: 0, duration: '20s' }] },
    sync: { executor: 'constant-arrival-rate', exec: 'syncTicket', rate: 5, timeUnit: '1s', duration: '3m',
            preAllocatedVUs: 10, maxVUs: 40 },
    search: { executor: 'constant-arrival-rate', exec: 'search', rate: 3, timeUnit: '1s', duration: '3m',
              preAllocatedVUs: 5, maxVUs: 20 },
  },
  thresholds: { 'http_req_failed': ['rate<0.01'], 'http_req_duration{scenario:async}': ['p(95)<500'] },
};

export function asyncTicket() {
  const t = rnd(DATA);
  const r = http.post(`${API}/v1/tickets?mode=async`, JSON.stringify(t), { ...H, tags: { name: 'POST /v1/tickets async' } });
  check(r, { 'queued 201': (x) => x.status === 201 });
}

export function syncTicket() {
  const r = http.post(`${API}/v1/tickets`, JSON.stringify(rnd(DATA)), { ...H, tags: { name: 'POST /v1/tickets sync' } });
  check(r, { 'routed 201': (x) => x.status === 201 });
}

export function search() {
  const q = rnd(DATA).text.split(' ').slice(0, 4).join(' ');
  const r = http.get(`${API}/v1/search?q=${encodeURIComponent(q)}&k=5`, { tags: { name: 'GET /v1/search' } });
  check(r, { 'search 200': (x) => x.status === 200 });
}
