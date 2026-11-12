#!/usr/bin/env bash
# Демонстрация проекта (Linux/macOS): стек в Docker, e2e-тесты в docker-сети, нагрузка k6 + Grafana, логи в evidence/.
set -u
cd "$(dirname "$0")/.."
NAME="returnguard"
mkdir -p evidence
step() { printf '\n\033[36m==================== %s ====================\033[0m\n' "$1"; }
save() { local f="evidence/$1"; shift; "$@" 2>&1 | tee "$f"; }

step "1/8 Сборка образов и запуск стека ($NAME)"
docker compose up -d --build
step "2/8 Ожидание готовности API"
for i in $(seq 1 120); do curl -sf "http://localhost:8000/health" >/dev/null && break; printf .; sleep 5; done; echo
save 01_containers.txt docker compose ps -a
step "3/8 Все контейнеры в одной docker-сети ${NAME}_default"
save 02_network.txt docker network inspect "${NAME}_default" --format '{{range .Containers}}{{.Name}}  {{.IPv4Address}}{{println}}{{end}}'
save 03_kafka_topics.txt docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe
step "3b Поток синтетических заказов через Kafka"
save 03b_replay.txt docker compose --profile demo run --rm replay
sleep 20
step "4/8 E2E-тесты внутри docker-сети"
save 04_e2e_tests.txt docker compose --profile e2e run --rm e2e
step "5/8 Нагрузочный тест k6 (~3.5 мин): Grafana http://localhost:3000/d/overview"
( while true; do echo "---- $(date +%T)"; docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"; sleep 15; done ) >> evidence/06_docker_stats.txt &
STATS=$!
save 05_k6_load.txt docker compose --profile load run --rm k6
kill $STATS
step "6/8 Consumer groups и lag"
save 07_consumer_groups.txt docker compose exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --all-groups
step "7/8 Логи"
docker compose logs --no-color --tail 300 > evidence/08_logs.txt 2>&1
step "8/8 Готово: evidence/, Grafana http://localhost:3000, остановить: docker compose down -v"
