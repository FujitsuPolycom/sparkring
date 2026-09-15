# Encode one literal argument for the remote POSIX shell.
function ConvertTo-PosixShellArgument {
    param([AllowEmptyString()][string]$Value)
    if ($Value.Contains([char]0)) { throw "Shell arguments cannot contain NUL" }
    if ($Value -match '^[A-Za-z0-9_./:@%+=,-]+$') { return $Value }
    return "'" + $Value.Replace("'", "'\''") + "'"
}
