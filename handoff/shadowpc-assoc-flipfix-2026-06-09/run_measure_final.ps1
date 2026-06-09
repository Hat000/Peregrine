# Re-run the ROBUST configs at the adopted RANGE_REL_TOL=0.15 (authoritative final numbers).
$py = ".venv\Scripts\python.exe"
$out = "handoff\shadowpc-assoc-flipfix-2026-06-09"
$weights = "models\gate_yolo11s_curriculum_v2.pt"
$map = "handoff\shadowpc-firstcontact-2026-06-02\track_map.json"

foreach ($cfg in @(
    @{name = "robust_k1.0"; extra = @("--cov-inflation", "1.0")},
    @{name = "robust_k2.0"; extra = @("--cov-inflation", "2.0")}
)) {
    foreach ($g in 0..5) {
        $bundle = "handoff\perception-char-2026-06-08\pg\course_g$g"
        $json = "$out\char_g$($g)_$($cfg.name).json"
        $log = "$out\char_g$($g)_$($cfg.name).stdout.txt"
        Write-Host ">>> g$g $($cfg.name)"
        & $py scripts\characterize_perception.py --bundle $bundle --weights $weights --map $map `
            --json $json @($cfg.extra) | Out-File -Encoding utf8 $log
        if ($LASTEXITCODE -ne 0) { Write-Host "!! FAILED g$g $($cfg.name)"; exit 1 }
    }
}
Write-Host "ALL DONE"
