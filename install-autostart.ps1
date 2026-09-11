<#
.SYNOPSIS
登入時自動啟動 LocalSTT 的辨識端點。

.DESCRIPTION
註冊一個工作排程器項目，在目前使用者登入時以隱藏視窗啟動 shim。
不需要管理員權限——工作註冊在目前使用者名下，也不要求最高權限。

notes: 用工作排程器而不是「啟動」資料夾的捷徑。捷徑會閃出一個 PowerShell 視窗，
       而且沒有「登入後延遲」可設。模型要載進 GPU，登入當下跟其他自啟程式搶資源
       只會拖慢桌面出現的速度，所以延後 30 秒。

.PARAMETER Uninstall
移除該工作排程項目。
#>
param([switch]$Uninstall)

$taskName = 'LocalSTT'
$startScript = Join-Path $PSScriptRoot 'start.ps1'

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "已移除工作排程項目：$taskName"
    } else {
        Write-Host "找不到工作排程項目：$taskName。不需要移除。"
    }
    return
}

if (-not (Test-Path $startScript)) { throw "找不到啟動腳本：$startScript" }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$startScript`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = 'PT30S'
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force -Description '登入後啟動本機語音辨識端點（LocalSTT）' | Out-Null

Write-Host "已註冊工作排程項目：$taskName（登入後延遲 30 秒啟動）"
Write-Host "立即測試：Start-ScheduledTask -TaskName $taskName"
Write-Host "移除：install-autostart.ps1 -Uninstall"
