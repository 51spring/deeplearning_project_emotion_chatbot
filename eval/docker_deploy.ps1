<#
.SYNOPSIS
Docker 배포 비밀값과 컨테이너, Cloudflare quick tunnel을 한 스크립트에서 관리한다.

.PARAMETER Action
init, build, start, recreate, stop, logs, status, tunnel 중 실행할 작업이다.

.PARAMETER Port
호스트에서 공개할 FastAPI 포트다. 기본값은 8000이다.

.PARAMETER Force
init 작업에서 기존 docker.env를 새 무작위 값으로 덮어쓸 때 사용한다.

.OUTPUTS
Docker 또는 cloudflared 명령의 실행 결과와 상태 안내를 출력한다.
#>
[CmdletBinding()]
param(
    [ValidateSet("init", "build", "start", "recreate", "stop", "logs", "status", "tunnel")]
    [string]$Action = "start",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvFile = Join-Path $ProjectRoot "docker.env"
$ImageName = "emotion-chatbot:class-demo"
$ContainerName = "emotion-chatbot-demo"

<#
.SYNOPSIS
암호학적 난수 바이트를 읽기 쉬운 16진수 문자열로 만든다.

.PARAMETER ByteCount
생성할 난수 바이트 수다.

.OUTPUTS
ByteCount의 두 배 길이를 가진 16진수 문자열을 반환한다.
#>
function New-SecureHexToken {
    param(
        [ValidateRange(12, 128)]
        [int]$ByteCount
    )

    $buffer = New-Object byte[] $ByteCount
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($buffer)
    }
    finally {
        $rng.Dispose()
    }

    return ([System.BitConverter]::ToString($buffer) -replace "-", "").ToLowerInvariant()
}

<#
.SYNOPSIS
docker.env 파일을 읽어 환경변수 이름과 값을 반환한다.

.PARAMETER Path
읽을 Docker 환경변수 파일 경로다.

.OUTPUTS
환경변수 이름을 키로 사용하는 Hashtable을 반환한다.
#>
function Read-DockerEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }

        $separator = $trimmed.IndexOf("=")
        if ($separator -le 0) {
            throw "잘못된 docker.env 행입니다: $line"
        }

        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        $values[$name] = $value
    }

    return $values
}

<#
.SYNOPSIS
production 실행에 필요한 Docker 비밀값을 검증한다.

.PARAMETER Values
Read-DockerEnvironment가 반환한 환경변수 Hashtable이다.

.OUTPUTS
유효하면 값을 반환하지 않고, 문제가 있으면 예외를 발생시킨다.
#>
function Assert-DockerEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Values
    )

    $required = @(
        "EMOTION_CHATBOT_ENV",
        "EMOTION_CHATBOT_AUTH_SECRET",
        "EMOTION_CHATBOT_DEVELOPER_PASSWORD",
        "EMOTION_CHATBOT_ROOT_PASSWORD"
    )
    foreach ($name in $required) {
        if (-not $Values.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($Values[$name])) {
            throw "docker.env에 $name 값이 필요합니다."
        }
    }

    if ($Values["EMOTION_CHATBOT_ENV"] -ne "production") {
        throw "Docker 배포 모드는 production이어야 합니다."
    }
    if ($Values["EMOTION_CHATBOT_AUTH_SECRET"].Length -lt 32) {
        throw "EMOTION_CHATBOT_AUTH_SECRET은 32자 이상이어야 합니다."
    }
    if ($Values["EMOTION_CHATBOT_DEVELOPER_PASSWORD"].Length -lt 12) {
        throw "EMOTION_CHATBOT_DEVELOPER_PASSWORD는 12자 이상이어야 합니다."
    }
    if ($Values["EMOTION_CHATBOT_ROOT_PASSWORD"].Length -lt 12) {
        throw "EMOTION_CHATBOT_ROOT_PASSWORD는 12자 이상이어야 합니다."
    }
    if ($Values["EMOTION_CHATBOT_DEVELOPER_PASSWORD"] -eq $Values["EMOTION_CHATBOT_ROOT_PASSWORD"]) {
        throw "developer와 root 비밀번호는 서로 달라야 합니다."
    }
}

<#
.SYNOPSIS
Docker 명령을 실행하고 실패 코드를 PowerShell 예외로 변환한다.

.PARAMETER Arguments
docker 실행 파일에 전달할 인자 배열이다.

.OUTPUTS
Docker 명령의 표준 출력을 그대로 출력한다.
#>
function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker 명령이 실패했습니다: docker $($Arguments -join ' ')"
    }
}

<#
.SYNOPSIS
지정한 이름의 Docker 컨테이너 존재 여부를 확인한다.

.OUTPUTS
컨테이너가 존재하면 true, 없으면 false를 반환한다.
#>
function Test-ContainerExists {
    $name = & docker ps -a --filter "name=^/$ContainerName$" --format "{{.Names}}"
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 컨테이너 상태를 확인하지 못했습니다."
    }
    return $name -contains $ContainerName
}

<#
.SYNOPSIS
docker.env와 프로젝트 볼륨을 사용해 GPU 컨테이너를 생성한다.

.OUTPUTS
생성된 컨테이너 ID를 출력한다.
#>
function New-DeploymentContainer {
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        throw "docker.env가 없습니다. 먼저 '.\eval\docker_deploy.ps1 init'을 실행하세요."
    }

    $values = Read-DockerEnvironment -Path $EnvFile
    Assert-DockerEnvironment -Values $values

    $hfCache = Join-Path $env:USERPROFILE ".cache\huggingface"
    if (-not (Test-Path -LiteralPath $hfCache)) {
        New-Item -ItemType Directory -Path $hfCache -Force | Out-Null
    }

    $robertaCheckpoints = Join-Path $ProjectRoot "models\roberta\checkpoints"
    $qwenCheckpoints = Join-Path $ProjectRoot "models\qwen\checkpoints"
    $processedData = Join-Path $ProjectRoot "data\processed"
    $databaseDirectory = Join-Path $ProjectRoot "backend\db"

    $arguments = @(
        "run", "-d",
        "--name", $ContainerName,
        "--restart", "unless-stopped",
        "--gpus", "all",
        "-p", "${Port}:8000",
        "--env-file", $EnvFile,
        "-v", "${robertaCheckpoints}:/app/models/roberta/checkpoints",
        "-v", "${qwenCheckpoints}:/app/models/qwen/checkpoints",
        "-v", "${processedData}:/app/data/processed",
        "-v", "${databaseDirectory}:/app/backend/db",
        "-v", "${hfCache}:/app/.cache/huggingface",
        $ImageName
    )
    Invoke-Docker -Arguments $arguments
}

switch ($Action) {
    "init" {
        if ((Test-Path -LiteralPath $EnvFile) -and -not $Force) {
            Write-Host "docker.env가 이미 있습니다. 기존 값을 유지합니다."
            Write-Host "새 값으로 교체하려면 -Force를 함께 사용하세요."
            break
        }

        $authSecret = New-SecureHexToken -ByteCount 32
        $developerPassword = New-SecureHexToken -ByteCount 12
        $rootPassword = New-SecureHexToken -ByteCount 12
        $content = @(
            "EMOTION_CHATBOT_ENV=production",
            "EMOTION_CHATBOT_AUTH_SECRET=$authSecret",
            "EMOTION_CHATBOT_DEVELOPER_PASSWORD=$developerPassword",
            "EMOTION_CHATBOT_ROOT_PASSWORD=$rootPassword",
            "EMOTION_CHATBOT_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000",
            "EMOTION_CHATBOT_TRUST_PROXY_HEADERS=1",
            "EMOTION_CHATBOT_TIMEZONE=Asia/Seoul",
            "HF_HOME=/app/.cache/huggingface"
        )
        Set-Content -LiteralPath $EnvFile -Value $content -Encoding ASCII
        Write-Host "docker.env를 생성했습니다: $EnvFile"
        Write-Host "관리자 로그인 비밀번호는 이 파일에서 확인할 수 있습니다."
    }
    "build" {
        Invoke-Docker -Arguments @("build", "-t", $ImageName, $ProjectRoot)
    }
    "start" {
        if (Test-ContainerExists) {
            Invoke-Docker -Arguments @("start", $ContainerName)
        }
        else {
            New-DeploymentContainer
        }
    }
    "recreate" {
        if (Test-ContainerExists) {
            Invoke-Docker -Arguments @("rm", "-f", $ContainerName)
        }
        New-DeploymentContainer
    }
    "stop" {
        if (Test-ContainerExists) {
            Invoke-Docker -Arguments @("stop", $ContainerName)
        }
        else {
            Write-Host "$ContainerName 컨테이너가 없습니다."
        }
    }
    "logs" {
        if (-not (Test-ContainerExists)) {
            throw "$ContainerName 컨테이너가 없습니다."
        }
        Invoke-Docker -Arguments @("logs", "--tail", "100", "-f", $ContainerName)
    }
    "status" {
        Invoke-Docker -Arguments @(
            "ps", "-a",
            "--filter", "name=^/$ContainerName$",
            "--format", "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
        )
    }
    "tunnel" {
        $cloudflared = Get-Command cloudflared -ErrorAction Stop
        & $cloudflared.Source tunnel --url "http://127.0.0.1:$Port"
        if ($LASTEXITCODE -ne 0) {
            throw "cloudflared tunnel 실행에 실패했습니다."
        }
    }
}
