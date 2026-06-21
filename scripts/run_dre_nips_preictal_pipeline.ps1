param(
    [string]$Python = "python",
    [string]$MetadataDir = "D:\all_seeg_data",
    [string]$LzuRoot = ("F:\" + (-join ([char[]](0x5170,0x5927,0x4E8C,0x9662,0x65B0,0x0053,0x0045,0x0045,0x0047,0x6570,0x636E)))),
    [string]$HupRoot = "E:\DRE-nips\dataest",
    [string]$MulticenterRoot = "E:\DRE-nips\dataest",
    [string]$PediatricRoot = ("F:\" + (-join ([char[]](0x513F,0x79D1,0x533B,0x9662,0x0053,0x0045,0x0045,0x0047,0x6574,0x7406,0x6C47,0x603B)))),
    [string]$OutputRoot = "D:\nips-temp",
    [switch]$SmokeTest,
    [switch]$Strict,
    [switch]$SkipTraining,
    [switch]$SkipPathValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PatientRecordsDir = Join-Path $OutputRoot "biodynformer_patient_records"
$FeatureBankDir = Join-Path $OutputRoot "biodynformer_preictal_feature_bank"
$RunsDir = Join-Path $OutputRoot "biodynformer_runs"

$PatientRecordsPkl = Join-Path $PatientRecordsDir "patient_records.pkl"
$SummaryJson = Join-Path $PatientRecordsDir "patient_records_summary.json"
$QualityAuditCsv = Join-Path $PatientRecordsDir "quality_match_audit.csv"
$ReadAuditDir = Join-Path $PatientRecordsDir "read_audit"

$HupParticipantsPath = Join-Path $MetadataDir "participants.tsv"
$MulticenterParticipantsPath = Join-Path $MetadataDir "participants-muticenter.tsv"
$PediatricMetadataPath = Join-Path $MetadataDir "pediatric_ez_channels_final.xlsx"

function Assert-ExistingPath {
    param(
        [string]$PathValue,
        [string]$Name
    )
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Name not found: $PathValue"
    }
}

function Invoke-CheckedPython {
    param(
        [string]$StepName,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "=== $StepName ==="
    Write-Host "$Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

Push-Location $ProjectRoot
try {
    if (-not $SkipPathValidation) {
        Assert-ExistingPath $MetadataDir "MetadataDir"
        Assert-ExistingPath $LzuRoot "LzuRoot"
        Assert-ExistingPath $HupRoot "HupRoot"
        Assert-ExistingPath $MulticenterRoot "MulticenterRoot"
        Assert-ExistingPath $PediatricRoot "PediatricRoot"
        Assert-ExistingPath $HupParticipantsPath "HUP participants.tsv"
        Assert-ExistingPath $MulticenterParticipantsPath "multicenter participants-muticenter.tsv"
        Assert-ExistingPath $PediatricMetadataPath "pediatric_ez_channels_final.xlsx"
    }

    New-Item -ItemType Directory -Force -Path $PatientRecordsDir, $FeatureBankDir, $RunsDir, $ReadAuditDir | Out-Null

    $StrictFlag = if ($Strict) { "--strict" } else { "--no-strict" }
    $ReaderArgs = @(
        "scripts\build_patient_records_from_dre_nips_preictal_only.py",
        "--quality-report-root", $MetadataDir,
        "--output-pkl", $PatientRecordsPkl,
        "--output-summary-json", $SummaryJson,
        "--quality-audit-csv", $QualityAuditCsv,
        "--read-audit-dir", $ReadAuditDir,
        "--centers", "lzu,hup,multicenter,pediatric",
        "--lzu-root", $LzuRoot,
        "--hup-root", $HupRoot,
        "--hup-participants-path", $HupParticipantsPath,
        "--multicenter-root", $MulticenterRoot,
        "--multicenter-sidecar-root", $MulticenterRoot,
        "--multicenter-participants-path", $MulticenterParticipantsPath,
        "--pediatric-root", $PediatricRoot,
        "--pediatric-metadata-xlsx", $PediatricMetadataPath,
        "--no-success-only",
        $StrictFlag
    )
    if ($SmokeTest) {
        $ReaderArgs += @("--debug-limit", "1")
    }
    Invoke-CheckedPython "1. Build patient_records.pkl from dre-nips ictal readers" $ReaderArgs

    $FeatureBankArgs = @(
        "scripts\build_feature_bank.py",
        "--source", "patient-records-pkl",
        "--patient-records-pkl", $PatientRecordsPkl,
        "--output-dir", $FeatureBankDir,
        "--quality-filter",
        "--quality-keep-ratings", "GOOD,REVIEW",
        "--quality-drop-ratings", "POOR",
        "--quality-missing-policy", "keep"
    )
    Invoke-CheckedPython "2. Build preictal feature bank" $FeatureBankArgs

    if (-not $SkipTraining) {
        $RunArgs = @(
            "scripts\run_all_versions.py",
            "--feature-bank", $FeatureBankDir,
            "--output-dir", $RunsDir,
            "--versions", "v1,v2,final",
            "--tasks", "task1,task2",
            "--run-5fold",
            "--run-loco",
            "--resume"
        )
        Invoke-CheckedPython "3. Run v1/v2/final experiments" $RunArgs
    }

    Write-Host ""
    Write-Host "Pipeline outputs:"
    Write-Host "  patient records: $PatientRecordsPkl"
    Write-Host "  summary:         $SummaryJson"
    Write-Host "  quality audit:   $QualityAuditCsv"
    Write-Host "  read audit:      $ReadAuditDir"
    Write-Host "  feature bank:    $FeatureBankDir"
    Write-Host "  runs:            $RunsDir"
}
finally {
    Pop-Location
}
