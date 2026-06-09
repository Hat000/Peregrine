# Before/after measurement for the robust-association + depth-sanity work (2026-06-09).
# 3 configs x 6 per-gate course bundles from data/runs/20260607_194615_course_60s.
$py = ".venv\Scripts\python.exe"
$out = "handoff\shadowpc-assoc-flipfix-2026-06-09"
$weights = "models\gate_yolo11s_curriculum_v2.pt"
$map = "handoff\shadowpc-firstcontact-2026-06-02\track_map.json"

foreach ($cfg in @(
    @{name = "naive_k1.0";  extra = @("--naive-assoc", "--cov-inflation", "1.0")},
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
