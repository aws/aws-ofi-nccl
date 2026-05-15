#!/bin/bash
# Parse Valgrind memcheck reports from aws-ofi-nccl test output.
# Extracts errors and memory leaks, classifies plugin vs non-plugin,
# deduplicates across PIDs/ranks, and produces a concise summary.
#
# Handles: Invalid read/write, Conditional jump, Use of uninitialised value,
#          Syscall param, memory leaks (definitely/indirectly/possibly lost).
#
# Usage: ./parse_valgrind.sh <valgrind_output_file>

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <valgrind_output_file>"
    exit 1
fi

INPUT="$1"
[[ -f "$INPUT" ]] || { echo "Error: $INPUT not found"; exit 1; }

PARSE_TMPDIR=$(mktemp -d)
trap 'rm -rf "$PARSE_TMPDIR"' EXIT

if ! command -v gawk &>/dev/null; then
    echo "Error: gawk is required but not found" >&2
    exit 1
fi

plugin_blocks="$PARSE_TMPDIR/plugin_errors.txt"
other_blocks="$PARSE_TMPDIR/other_errors.txt"
dedup_plugin="$PARSE_TMPDIR/dedup_plugin.txt"
plugin_leaks="$PARSE_TMPDIR/plugin_leaks.txt"
other_leaks="$PARSE_TMPDIR/other_leaks.txt"

# Single-pass awk: extract blocks, classify, and deduplicate all at once.
gawk '
BEGIN {
    plugin_re = "(nccl_ofi_[a-z_]+\\.(cpp|h)|platform-aws\\.cpp):[0-9]+"
    fullpath_re = "aws-ofi-nccl/(src|include)/"
}

function flush_block() {
    if (buf == "") return

    total_blocks++

    is_plugin = 0
    if (buf ~ plugin_re || buf ~ fullpath_re) is_plugin = 1

    is_leak = (err_type ~ /definitely lost|indirectly lost|possibly lost/)

    if (is_plugin) {
        plugin_count++
        print buf > "'"$plugin_blocks"'"

        if (is_leak) {
            print buf > "'"$plugin_leaks"'"
            plugin_leak_count++
            if (match(header_line, /== ([0-9,]+) (\([0-9,]+ direct, [0-9,]+ indirect\) )?bytes in ([0-9,]+) blocks are (definitely|indirectly|possibly) lost/, parts)) {
                gsub(/,/, "", parts[1])
                gsub(/,/, "", parts[3])
                leak_bytes = parts[1] + 0
                leak_blocks_n = parts[3] + 0
                leak_kind = parts[4]
                plugin_leak_bytes[leak_kind] += leak_bytes
                plugin_leak_blocks[leak_kind] += leak_blocks_n
            }
        } else {
            plugin_error_count++
        }

        first_src = ""
        n_lines = split(buf, lines, "\n")
        for (i = 1; i <= n_lines; i++) {
            if (first_src == "" && match(lines[i], "\\(" plugin_re "\\)")) {
                first_src = substr(lines[i], RSTART, RLENGTH)
            }
            if (first_src == "" && lines[i] ~ fullpath_re) {
                match(lines[i], /aws-ofi-nccl\/(src|include)\/[^ )]+/)
                if (RSTART > 0) first_src = substr(lines[i], RSTART, RLENGTH)
            }
        }
        dk = err_type "|" first_src
        if (!(dk in seen)) {
            seen[dk] = 1
            dedup_order[++dedup_n] = dk
            dedup_headers[dk] = header_line
            dedup_bodies[dk] = buf
            dedup_counts[dk] = 1
            dedup_is_leak[dk] = is_leak
        } else {
            dedup_counts[dk]++
        }
    } else {
        other_count++
        print buf > "'"$other_blocks"'"
        if (is_leak) {
            print buf > "'"$other_leaks"'"
            other_leak_count++
        }
    }
    buf = ""
    header_line = ""
    err_type = ""
}

/^==[0-9]+== (Syscall|Conditional|Use of|Invalid|Uninitialised)/ {
    flush_block()
    buf = $0 "\n"
    header_line = $0
    match($0, /== (Syscall|Conditional|Use of uninitialised|Invalid read|Invalid write)/, m)
    err_type = (m[1] != "") ? m[1] : "unknown"
    in_block = 1
    next
}

/^==[0-9]+== [0-9,]+ (\([0-9,]+ direct, [0-9,]+ indirect\) )?bytes in [0-9,]+ blocks are (definitely|indirectly|possibly) lost/ {
    flush_block()
    buf = $0 "\n"
    header_line = $0
    match($0, /(definitely|indirectly|possibly) lost/, m)
    err_type = m[0]
    in_block = 1
    next
}

in_block && /^==[0-9]+== $/ { buf = buf $0 "\n"; flush_block(); in_block = 0; next }
in_block && !/^==[0-9]+=/ { flush_block(); in_block = 0; next }
in_block { buf = buf $0 "\n" }
END {
    flush_block()

    dedup_error_n = 0
    dedup_leak_n = 0
    for (i = 1; i <= dedup_n; i++) {
        k = dedup_order[i]
        if (!dedup_is_leak[k]) {
            dedup_error_n++
            printf "[%d occurrences] %s\n%s\n", dedup_counts[k], dedup_headers[k], dedup_bodies[k] > "'"$dedup_plugin"'.errors"
        } else {
            dedup_leak_n++
            printf "[%d occurrences] %s\n%s\n", dedup_counts[k], dedup_headers[k], dedup_bodies[k] > "'"$dedup_plugin"'.leaks"
        }
    }

    printf "%d %d %d %d %d %d %d %d\n", \
        total_blocks, plugin_count, other_count, dedup_n, \
        plugin_error_count, plugin_leak_count, dedup_error_n, dedup_leak_n \
        > "'"$PARSE_TMPDIR"'/stats.txt"

    for (kind in plugin_leak_bytes) {
        printf "%s %d %d\n", kind, plugin_leak_bytes[kind], plugin_leak_blocks[kind] \
            > "'"$PARSE_TMPDIR"'/leak_bytes.txt"
    }
}
' "$INPUT"

# Read stats
read raw_count plugin_count other_count unique_count \
     plugin_error_count plugin_leak_count dedup_error_n dedup_leak_n \
     < "$PARSE_TMPDIR/stats.txt" 2>/dev/null || {
    echo "No valgrind errors or leaks found in $INPUT"
    exit 0
}
parsed_count=$((plugin_count + other_count))
num_pids=$(grep -oP '==\K[0-9]+(?===)' "$INPUT" | sort -u | wc -l || true)

# Output report
echo "============================================================"
echo "  Valgrind Memcheck Analysis: NCCL OFI Plugin"
echo "  Input: $INPUT"
echo "  PIDs (ranks): $num_pids"
echo "============================================================"
echo ""
echo "--- PLUGIN ERRORS (non-leak) ---"
echo "  Total: $plugin_error_count  Unique: $dedup_error_n"
echo ""
echo "--- PLUGIN LEAKS ---"
echo "  Total: $plugin_leak_count  Unique: $dedup_leak_n"
if [[ -f "$PARSE_TMPDIR/leak_bytes.txt" ]]; then
    echo "  By kind:"
    while read kind bytes blocks; do
        printf "    %-20s %'d bytes in %'d blocks\n" "$kind:" "$bytes" "$blocks"
    done < "$PARSE_TMPDIR/leak_bytes.txt" | sort
fi
echo ""
echo "--- NON-PLUGIN (third-party) ---"
echo "  Count: $other_count"
echo ""
echo "--- DEDUPLICATED PLUGIN ERRORS ---"
if [[ -f "$dedup_plugin.errors" ]]; then
    cat "$dedup_plugin.errors"
else
    echo "  None."
fi
echo ""
echo "--- DEDUPLICATED PLUGIN LEAKS ---"
if [[ -f "$dedup_plugin.leaks" ]]; then
    cat "$dedup_plugin.leaks"
else
    echo "  None."
fi
echo ""
echo "============================================================"
