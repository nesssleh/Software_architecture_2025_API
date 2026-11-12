// Нагрузочный тест ReturnGuard (k6). Три параллельных сценария:
//  score   — синхронный скоринг возвратов через investigation-api (рост 10 → 150 RPS);
//  events  — поток заказов/доставок/возвратов через order-ingest → Kafka → feature-service → scoring-service;
//  analyst — работа аналитика: очередь кейсов и профили риска.
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8000';
const ING = __ENV.INGEST_URL || 'http://localhost:8001';
const JSON_H = { 'Content-Type': 'application/json' };
const AUTH = { ...JSON_H, 'X-API-Token': 'dev-token' };
const REASONS = ['size', 'changed_mind', 'not_as_described', 'damaged', 'defect', 'not_received'];
const CATS = ['apparel', 'shoes', 'electronics', 'home', 'beauty', 'luxury'];

export const options = {
  scenarios: {
    score: { executor: 'ramping-arrival-rate', exec: 'score', startRate: 10, timeUnit: '1s',
             preAllocatedVUs: 30, maxVUs: 200,
             stages: [{ target: 50, duration: '1m' }, { target: 100, duration: '1m' },
                      { target: 150, duration: '1m' }, { target: 0, duration: '20s' }] },
    events: { executor: 'constant-arrival-rate', exec: 'events', rate: 60, timeUnit: '1s', duration: '3m20s',
              preAllocatedVUs: 30, maxVUs: 100 },
    analyst: { executor: 'constant-vus', exec: 'analyst', vus: 2, duration: '3m20s' },
  },
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'http_req_duration{scenario:score}': ['p(95)<300'],
    'http_req_duration{scenario:events}': ['p(95)<200'],
  },
};

const rnd = (a) => a[Math.floor(Math.random() * a.length)];
const iso = (d) => new Date(Date.now() - d * 86400000).toISOString();

export function score() {
  const cid = 'C' + String(Math.floor(Math.random() * 1500)).padStart(6, '0');
  const price = Math.round(20 + Math.random() * 1500);
  const body = { return_id: `K6-${__VU}-${__ITER}`, customer_id: cid, order_id: 'O-K6', ts: iso(0), amount: price,
                 reason: rnd(REASONS), refund_method: 'card', items: [{ sku: 'K6', category: rnd(CATS), price }] };
  const r = http.post(`${API}/v1/score`, JSON.stringify(body), { headers: AUTH, tags: { name: 'POST /v1/score' } });
  check(r, { 'score 200': (x) => x.status === 200 });
}

export function events() {
  const id = `${__VU}-${__ITER}-${Date.now()}`;
  const cid = `K6C-${__VU}-${__ITER % 50}`;          // повторяющиеся клиенты → копится история
  const price = Math.round(20 + Math.random() * 900);
  const items = [{ sku: 'SKU1', category: rnd(CATS), price }];
  const post = (b) => http.post(`${ING}/v1/events`, JSON.stringify(b), { headers: JSON_H, tags: { name: 'POST /v1/events' } });
  const r1 = post({ type: 'order.created', ts: iso(10), customer_id: cid, order_id: `O-${id}`, items, amount: price,
                    payment_method: 'card', address_id: `A-${cid}`, signup_ts: iso(400) });
  const r2 = post({ type: 'order.delivered', ts: iso(8), customer_id: cid, order_id: `O-${id}` });
  check(r1, { 'event 202': (x) => x.status === 202 });
  check(r2, { 'event 202': (x) => x.status === 202 });
  if (Math.random() < 0.5) {
    const r3 = post({ type: 'return.requested', ts: iso(6), customer_id: cid, order_id: `O-${id}`, return_id: `R-${id}`,
                      items, amount: price, reason: rnd(REASONS), refund_method: 'wallet' });
    check(r3, { 'event 202': (x) => x.status === 202 });
  }
}

export function analyst() {
  const r = http.get(`${API}/v1/cases?status=open&limit=20`, { headers: AUTH, tags: { name: 'GET /v1/cases' } });
  check(r, { 'cases 200': (x) => x.status === 200 });
  const cid = 'C' + String(Math.floor(Math.random() * 1500)).padStart(6, '0');
  http.get(`${API}/v1/customers/${cid}/risk`, { headers: AUTH, tags: { name: 'GET /v1/customers/{id}/risk' },
                                                responseCallback: http.expectedStatuses(200, 404) });
}
