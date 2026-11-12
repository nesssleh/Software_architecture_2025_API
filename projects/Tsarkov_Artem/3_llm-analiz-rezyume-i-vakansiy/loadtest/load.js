// Нагрузочный тест SkillMatch (k6).
//  setup   — загружает 20 вакансий и 100 резюме (синхронно);
//  ingest  — поток резюме в асинхронном режиме: API → Kafka documents.ingested → extractor-worker ×2 → pgvector;
//  match   — сопоставление резюме и вакансии с объяснением (рост 10 → 80 RPS);
//  search  — поиск кандидатов на вакансию (pgvector + переранжирование).
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8000';
const DATA = JSON.parse(open((__ENV.SCRIPTS_DIR || '/scripts') + '/data.json'));
const H = { headers: { 'Content-Type': 'application/json' } };

export const options = {
  setupTimeout: '180s',
  scenarios: {
    ingest: { executor: 'constant-arrival-rate', exec: 'ingest', rate: 20, timeUnit: '1s', duration: '3m',
              preAllocatedVUs: 10, maxVUs: 40 },
    match: { executor: 'ramping-arrival-rate', exec: 'match', startRate: 10, timeUnit: '1s', preAllocatedVUs: 20, maxVUs: 150,
             stages: [{ target: 40, duration: '1m' }, { target: 80, duration: '1m' }, { target: 80, duration: '40s' },
                      { target: 0, duration: '20s' }] },
    search: { executor: 'constant-arrival-rate', exec: 'search', rate: 3, timeUnit: '1s', duration: '3m',
              preAllocatedVUs: 5, maxVUs: 20 },
  },
  thresholds: { 'http_req_failed': ['rate<0.01'], 'http_req_duration{scenario:match}': ['p(95)<300'] },
};

export function setup() {
  const vac = [], res = [];
  DATA.vacancies.forEach((t, i) => {
    const r = http.post(`${API}/v1/documents`, JSON.stringify({ type: 'vacancy', text: t, id: `k6-vac-${i}` }), H);
    if (r.status === 201) vac.push(r.json('id'));
  });
  DATA.resumes.slice(0, 100).forEach((t, i) => {
    const r = http.post(`${API}/v1/documents`, JSON.stringify({ type: 'resume', text: t, id: `k6-res-${i}` }), H);
    if (r.status === 201) res.push(r.json('id'));
  });
  return { vac, res };
}

const rnd = (a) => a[Math.floor(Math.random() * a.length)];

export function ingest() {
  const t = rnd(DATA.resumes) + ` Обновлено ${__VU}-${__ITER}.`;
  const r = http.post(`${API}/v1/documents?mode=async`, JSON.stringify({ type: 'resume', text: t }),
                      { ...H, tags: { name: 'POST /v1/documents async' } });
  check(r, { 'queued 201': (x) => x.status === 201 });
}

export function match(d) {
  const r = http.post(`${API}/v1/match`, JSON.stringify({ resume_id: rnd(d.res), vacancy_id: rnd(d.vac), explain: true }),
                      { ...H, tags: { name: 'POST /v1/match' } });
  check(r, { 'match 200': (x) => x.status === 200 });
}

export function search(d) {
  const r = http.get(`${API}/v1/vacancies/${rnd(d.vac)}/candidates?k=10`, { tags: { name: 'GET /v1/vacancies/{id}/candidates' } });
  check(r, { 'candidates 200': (x) => x.status === 200 });
}
