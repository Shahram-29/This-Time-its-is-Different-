# Re-create a PDF (with heading bookmarks) next to every Word document in docs\ and docs\archive\.
# Uses a separate hidden Word instance and closes it only if it holds no other documents.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$files = Get-ChildItem -Path $root -Recurse -Filter *.docx | Where-Object { $_.Name -notlike '~$*' }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
foreach ($f in $files) {
    $pdf = [System.IO.Path]::ChangeExtension($f.FullName, ".pdf")
    try {
        $doc = $word.Documents.Open($f.FullName, $false, $true, $false)   # read-only: the .docx is never changed
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        [void]$doc.Fields.Update()
        # 17 = PDF; 1 = bookmarks from headings
        $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0, 0, 0, 0, $true, $true, 1, $true, $true, $false)
        $doc.Close([ref]0)
        Write-Output "OK    $($f.Name)"
    } catch {
        Write-Output "FAIL  $($f.Name): $($_.Exception.Message)"
    }
}
if ($word.Documents.Count -eq 0) { $word.Quit() }
