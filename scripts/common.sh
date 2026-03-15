#!/bin/bash

is_windows_shell() {
    case "${OSTYPE:-}" in
        msys*|cygwin*) return 0 ;;
    esac

    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
    esac

    return 1
}

project_python() {
    local project_root="$1"

    if is_windows_shell; then
        echo "$project_root/.venv/Scripts/python.exe"
    elif [ -x "$project_root/.venv/bin/python" ]; then
        echo "$project_root/.venv/bin/python"
    else
        command -v python
    fi
}

project_python_vs() {
    local project_root="$1"

    if is_windows_shell; then
        echo "$project_root/tests/run_with_vs.bat"
    else
        project_python "$project_root"
    fi
}

require_file() {
    local path="$1"
    local message="$2"

    if [ ! -f "$path" ]; then
        echo "[ERROR] $message"
        echo "        Missing: $path"
        exit 1
    fi
}

run_project_python() {
    local project_root="$1"
    shift

    local python_bin
    python_bin="$(project_python "$project_root")"
    require_file "$python_bin" "Project Python interpreter not found."
    "$python_bin" "$@"
}

run_project_python_vs() {
    local project_root="$1"
    shift

    local python_bin
    python_bin="$(project_python_vs "$project_root")"
    require_file "$python_bin" "Windows VS Python wrapper not found."
    "$python_bin" "$@"
}
