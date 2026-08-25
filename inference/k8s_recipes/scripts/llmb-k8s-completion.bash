# llmb-k8s bash completion.
#
# Install (from the k8s recipe root, inference/k8s_recipes):
#   source scripts/llmb-k8s-completion.bash
# or add that line to your ~/.bashrc for it to persist.
#
# Completes:
#   - the verb list             (kept in sync with KNOWN_VERBS in scripts/llmb-k8s — see below)
#   - --recipe <path>           (directory completion under recipes/)
#   - --cluster <name>          (profile names from cluster-profiles/*.env, minus *.example / *_template)
#
# SYNC NOTE: the verb list mirrors KNOWN_VERBS (the sorted set in scripts/llmb-k8s) plus the
# discoverability `help` verb. If you add a verb to the dispatcher, add it here too.

_llmb_k8s_verbs="analyze cancel capacity collect compare deploy dry-run fleet help init install jobs logs port-recipe preflight profile publish reclaim run stage status submit teardown-all watch"

# Resolve the k8s recipe root from this script's own location, so completion works regardless of cwd.
# When sourced, BASH_SOURCE[0] is this file; its grandparent is inference/k8s_recipes.
_llmb_k8s_root() {
    local src="${BASH_SOURCE[0]}"
    local dir
    dir="$(cd "$(dirname "$src")/.." >/dev/null 2>&1 && pwd)"
    printf '%s' "$dir"
}

# Profile names from cluster-profiles/*.env, stripping the .env suffix and excluding
# *.example / *_template / _template.* scaffolding files.
_llmb_k8s_clusters() {
    local root profiles f base
    root="$(_llmb_k8s_root)"
    profiles=""
    for f in "$root"/cluster-profiles/*.env; do
        [ -e "$f" ] || continue
        base="$(basename "$f" .env)"
        case "$f" in
            *.example|*.example.env) continue ;;
        esac
        case "$base" in
            *.example|_template|*_template) continue ;;
        esac
        profiles="$profiles $base"
    done
    printf '%s' "$profiles"
}

_llmb_k8s_complete() {
    local cur prev root
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    root="$(_llmb_k8s_root)"

    case "$prev" in
        --cluster)
            COMPREPLY=( $(compgen -W "$(_llmb_k8s_clusters)" -- "$cur") )
            return 0
            ;;
        --recipe)
            # Directory completion under recipes/ (recipe cells are directories). Default the
            # empty prefix to recipes/ so the first Tab lands in the recipe tree.
            COMPREPLY=( $(cd "$root" 2>/dev/null && compgen -d -- "${cur:-recipes/}") )
            return 0
            ;;
    esac

    # First word after the command → complete a verb.
    if [ "$COMP_CWORD" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$_llmb_k8s_verbs" -- "$cur") )
        return 0
    fi

    # Otherwise offer the named flags (and let default file completion handle positionals).
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "--recipe --cluster" -- "$cur") )
        return 0
    fi

    COMPREPLY=( $(compgen -f -- "$cur") )
    return 0
}

complete -F _llmb_k8s_complete llmb-k8s
