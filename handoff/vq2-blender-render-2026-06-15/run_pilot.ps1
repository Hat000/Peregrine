# VQ2 pilot render driver (ShadowPC, Blender 5.1 Cycles GPU). Logs per-arm timing for the throughput report.
$ErrorActionPreference = "Continue"
$bl   = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$repo = "C:\Users\Shadow\Peregrine"
$env:PYTHONPATH = "C:\Users\Shadow\AppData\Roaming\Python\Python313\site-packages"
$entry = "$repo\src\racer\vision\blender_gen\render_entry.py"
$root  = "$repo\handoff\vq2-blender-render-2026-06-15\pilot"
$log   = "$repo\handoff\vq2-blender-render-2026-06-15\pilot_render.log"
if (Test-Path $root) { Remove-Item $root -Recurse -Force }

# arm: name, preset, n_train, n_val, seed
$arms = @(
  @{n="appearance_broad"; p="appearance_broad";  tr=4000; va=400; s=0},
  @{n="negatives";        p="negatives";         tr=2000; va=200; s=101},
  @{n="terminal_approach";p="terminal_approach"; tr=1000; va=100; s=202}
)
"PILOT START $(Get-Date -Format o)" | Out-File $log -Encoding utf8
foreach ($a in $arms) {
  $out = "$root\$($a.n)"
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  "=== ARM $($a.n) preset=$($a.p) train=$($a.tr) val=$($a.va) seed=$($a.s) start=$(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
  & $bl --background --python $entry -- --preset $a.p --out $out --n-train $a.tr --n-val $a.va --seed $a.s --masks *>> $log
  $sw.Stop()
  $nimg = (Get-ChildItem "$out\images\train" -ErrorAction SilentlyContinue).Count + (Get-ChildItem "$out\images\val" -ErrorAction SilentlyContinue).Count
  $fph = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($nimg / $sw.Elapsed.TotalHours, 1) } else { 0 }
  "=== ARM $($a.n) DONE elapsed_s=$([math]::Round($sw.Elapsed.TotalSeconds,1)) images=$nimg frames_per_hr=$fph" | Out-File $log -Append -Encoding utf8
}
"PILOT DONE $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
