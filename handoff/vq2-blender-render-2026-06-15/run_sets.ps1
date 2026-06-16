# VQ2 photoreal BASE dataset: 2000 clean images (8-keypoint labels + masks), all-hue + VQ1-red.
# Clean (--no-augment): albumentations will fan these out into more variants later (offline multiply).
$ErrorActionPreference = "Continue"
$bl   = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$repo = "C:\Users\Shadow\Peregrine"
$env:PYTHONPATH = "C:\Users\Shadow\AppData\Roaming\Python\Python313\site-packages"
$entry = "$repo\src\racer\vision\blender_gen\render_entry.py"
$root  = "$repo\handoff\vq2-blender-render-2026-06-15\sets"
$log   = "$repo\handoff\vq2-blender-render-2026-06-15\sets_render.log"
if (Test-Path $root) { Remove-Item $root -Recurse -Force }

# 2000 base images total: all-hue (generalization driver, incl. ~20% negatives) + VQ1-red anchor.
$sets = @(
  @{n="allhue"; preset="appearance_broad"; tr=1400; va=0; s=1000},
  @{n="vq1red"; preset="vq1_faithful";     tr=600;  va=0; s=2000}
)
"SETS START $(Get-Date -Format o)" | Out-File $log -Encoding utf8
foreach ($a in $sets) {
  $out = "$root\$($a.n)"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  "=== SET $($a.n) preset=$($a.preset) train=$($a.tr) seed=$($a.s) start=$(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
  & $bl --background --python $entry -- --preset $a.preset --out $out --n-train $a.tr --n-val $a.va --seed $a.s --masks --no-augment *>> $log
  $sw.Stop()
  $nimg = (Get-ChildItem "$out\images\train" -ErrorAction SilentlyContinue).Count
  $fph = if ($sw.Elapsed.TotalHours -gt 0) { [math]::Round($nimg / $sw.Elapsed.TotalHours, 0) } else { 0 }
  "=== SET $($a.n) DONE elapsed_min=$([math]::Round($sw.Elapsed.TotalMinutes,1)) images=$nimg frames_per_hr=$fph" | Out-File $log -Append -Encoding utf8
}
"SETS DONE $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
