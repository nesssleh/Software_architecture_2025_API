// Нагрузочный тест Drone Combat RL (k6).
//  act      — инференс политики PPO по наблюдению (рост 50 → 400 RPS) — сценарий «онлайн-управление дроном»;
//  episodes — полный эпизод на сервере (PPO / эвристика) с записью итога в PostgreSQL.
// Параллельно симулятор непрерывно пишет телеметрию в Kafka (drone.telemetry / drone.episodes).
import http from 'k6/http';
import { check } from 'k6';

const API = __ENV.API_URL || 'http://localhost:8000';
const H = { headers: { 'Content-Type': 'application/json' } };

export const options = {
  scenarios: {
    act: { executor: 'ramping-arrival-rate', exec: 'act', startRate: 50, timeUnit: '1s', preAllocatedVUs: 30, maxVUs: 200,
           stages: [{ target: 150, duration: '1m' }, { target: 300, duration: '1m' }, { target: 400, duration: '40s' },
                    { target: 0, duration: '20s' }] },
    episodes: { executor: 'constant-arrival-rate', exec: 'episode', rate: 3, timeUnit: '1s', duration: '3m',
                preAllocatedVUs: 5, maxVUs: 20 },
  },
  thresholds: { 'http_req_failed': ['rate<0.01'], 'http_req_duration{scenario:act}': ['p(95)<250'] },
};

export function act() {
  const obs = Array.from({ length: 17 }, () => Math.random() * 2 - 1);
  const r = http.post(`${API}/v1/act`, JSON.stringify({ observation: obs, policy: 'ppo' }), { ...H, tags: { name: 'POST /v1/act' } });
  check(r, { 'act 200': (x) => x.status === 200 });
}

export function episode() {
  const body = { policy: Math.random() < 0.7 ? 'ppo' : 'heuristic', scenario: ['forest', 'corridor', 'mixed'][__ITER % 3],
                 seed: 100000 + __VU * 10000 + __ITER };
  const r = http.post(`${API}/v1/episodes/run`, JSON.stringify(body), { ...H, tags: { name: 'POST /v1/episodes/run' } });
  check(r, { 'episode 200': (x) => x.status === 200 });
}
