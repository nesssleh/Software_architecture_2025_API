# Демонстрация проекта: запуск всего стека в Docker, e2e-тесты внутри docker-сети,
# нагрузочный тест k6 с графиками в Grafana, сбор логов-доказательств в папку evidence\.
# Запуск (PowerShell):  powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
param([switch]$SkipBuild, [switch]$SkipLoad)
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")
$Name = "skillmatch"
New-Item -ItemType Directory -Force evidence | Out-Null

function Step($t) { Write-Host ""; Write-Host "==================== $t ====================" -ForegroundColor Cyan }
function Save($file, [scriptblock]$cmd) {
  # stderr docker'а — обычный текст (без обёртки NativeCommandError), файл — UTF-8 без BOM (читается на GitHub)
  $lines = & $cmd 2>&1 | ForEach-Object {
    $l = if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.Exception.Message } else { "$_" }
    Write-Host $l; $l
  }
  [IO.File]::WriteAllLines((Join-Path (Get-Location) "evidence\$file"), [string[]]@($lines), (New-Object Text.UTF8Encoding $false))
}

Step "1/8 Сборка образов и запуск стека ($Name)"
if ($SkipBuild) { docker compose up -d } else { docker compose up -d --build }

Step "2/8 Ожидание готовности API"
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
  try { if ((Invoke-WebRequest "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3).StatusCode -eq 200) { $ok = $true; break } } catch {}
  Write-Host -NoNewline "."; Start-Sleep 5
}
Write-Host ""
if (-not $ok) { Write-Host "API не поднялся, смотри: docker compose logs" -ForegroundColor Red; exit 1 }
Save "01_containers.txt" { docker compose ps -a }

Step "3/8 Все контейнеры в одной docker-сети ${Name}_default"
Save "02_network.txt" { docker network inspect "${Name}_default" --format '{{range .Containers}}{{.Name}}  {{.IPv4Address}}{{println}}{{end}}' }
Save "03_kafka_topics.txt" { docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --describe }

Step "4/8 E2E-тесты: контейнер e2e ходит в сервисы по именам внутри сети"
Save "04_e2e_tests.txt" { docker compose --profile e2e run --rm e2e }

if (-not $SkipLoad) {
  Step "5/8 Нагрузочный тест k6 (~3.5 мин). Открываю Grafana — смотри графики в реальном времени"
  Start-Process "http://localhost:3000/d/overview"
  $root = (Get-Location).Path
  $stats = Start-Job -ArgumentList $root -ScriptBlock {
    param($d)
    while ($true) {
      "---- $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 "$d\evidence\06_docker_stats.txt"
      docker stats --no-stream --format "table {{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}" | Out-File -Append -Encoding utf8 "$d\evidence\06_docker_stats.txt"
      Start-Sleep 15
    }
  }
  Save "05_k6_load.txt" { docker compose --profile load run --rm k6 }
  Stop-Job $stats; Remove-Job $stats
}

Step "6/8 Consumer groups и lag после нагрузки"
Save "07_consumer_groups.txt" { docker compose exec -T kafka /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --all-groups }

Step "7/8 Логи сервисов"
docker compose logs --no-color --tail 300 2>&1 | Out-File -Encoding utf8 (Join-Path "evidence" "08_logs.txt")

Step "8/8 Готово"
Write-Host "Доказательства: папка evidence\ (контейнеры, сеть, топики, e2e, k6, docker stats, consumer lag, логи)" -ForegroundColor Green
Write-Host "Grafana: http://localhost:3000   Prometheus: http://localhost:9090"
Write-Host "Swagger: http://localhost:8000/docs"
Write-Host "Остановить и удалить данные: docker compose down -v"
