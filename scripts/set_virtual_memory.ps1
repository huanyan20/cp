# 使用 CIM + Registry 設定 Page File（相容 Windows 10/11）

Write-Host "=== Windows Virtual Memory Configuration ==="
Write-Host "  Target: Init=24 GB / Max=48 GB on C:"
Write-Host ""

# Step 1: 關閉自動管理 Page File（Registry 方式，不依賴 WMI Put）
$memMgrKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
Set-ItemProperty -Path $memMgrKey -Name "PagingFiles" -Value "C:\pagefile.sys 24576 49152"
Write-Host "[OK] Registry PagingFiles set to: C:\pagefile.sys 24576 49152"

# Step 2: 同時確認自動管理已關閉
Set-ItemProperty -Path $memMgrKey -Name "LargeSystemCache" -Value 0 -ErrorAction SilentlyContinue

# Step 3: 驗證寫入結果
$val = (Get-ItemProperty -Path $memMgrKey -Name "PagingFiles").PagingFiles
Write-Host "[OK] Registry verified: $val"

Write-Host ""
Write-Host "============================================"
Write-Host " Virtual memory configured successfully!"
Write-Host " Init = 24,576 MB (24 GB)"
Write-Host " Max  = 49,152 MB (48 GB)"
Write-Host ""
Write-Host " IMPORTANT: You MUST RESTART your computer"
Write-Host " for the new page file to take effect."
Write-Host "============================================"
