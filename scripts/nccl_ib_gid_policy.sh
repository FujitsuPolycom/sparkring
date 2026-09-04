#!/usr/bin/env bash
# Validate the local NCCL RoCE GID policy before a launcher constructs Docker
# arguments. The caller provides die(), NCCL_IB_HCA, VLLM_HOST_IP, and the two
# GID policy variables. Automatic mode validates the same HCA/port members that
# NCCL selects, then unsets NCCL_IB_GID_INDEX so NCCL 2.21+ chooses an IPv4
# RoCEv2 GID independently for each member.

_sparkring_ipv4_gid_suffix() {
    local address=$1 octet1 octet2 octet3 octet4 octet
    IFS=. read -r octet1 octet2 octet3 octet4 <<EOF
$address
EOF
    [ -n "$octet1" ] && [ -n "$octet2" ] \
        && [ -n "$octet3" ] && [ -n "$octet4" ] || return 1
    for octet in "$octet1" "$octet2" "$octet3" "$octet4"; do
        case "$octet" in
            ''|*[!0-9]*) return 1 ;;
        esac
        [ "$((10#$octet))" -le 255 ] || return 1
    done
    printf '%02x%02x:%02x%02x' \
        "$((10#$octet1))" "$((10#$octet2))" \
        "$((10#$octet3))" "$((10#$octet4))"
}

_sparkring_netdev_ipv4s() {
    local netdev=$1 fixture_root=${SPARKRING_NETDEV_IPV4_ROOT-}
    # The root override is an offline-test seam. Deployment templates omit it,
    # so normal launches inspect addresses reported by the local ip command.
    if [ -n "$fixture_root" ]; then
        [ -r "$fixture_root/$netdev" ] || return 0
        tr '[:space:]' '\n' < "$fixture_root/$netdev" | sed '/^$/d'
        return 0
    fi
    ip -4 -o addr show dev "$netdev" 2>/dev/null \
        | awk '{print $4}' | cut -d/ -f1
}

_sparkring_nccl_atoi_port() {
    local text=$1 sign digits
    if [[ $text =~ ^[[:space:]]*([+-]?)([0-9]+) ]]; then
        sign=${BASH_REMATCH[1]}
        digits=${BASH_REMATCH[2]}
    else
        printf '0'
        return 0
    fi
    while [ "${#digits}" -gt 1 ] && [ "${digits#0}" != "$digits" ]; do
        digits=${digits#0}
    done
    if [ "${#digits}" -gt 9 ]; then
        printf '%s' "${sign}999999999"
        return 0
    fi
    digits=$((10#$digits))
    [ "$sign" != - ] || digits=$((0 - digits))
    printf '%s' "$digits"
}

_sparkring_validate_selected_gids() {
    local sysfs_root=${SPARKRING_INFINIBAND_SYSFS_ROOT:-/sys/class/infiniband}
    local selector=$NCCL_IB_HCA selector_text=$NCCL_IB_HCA
    local exclude=0 exact=0 selector_truncated=0 token_count=0
    local token name port_text port candidate_match selected_count=0
    local state link_layer dev_dir port_dir dev
    local -a token_names=() token_ports=() selected_devs=() selected_ports=()

    case "$selector" in
        ^*) exclude=1; selector=${selector#^} ;;
    esac
    case "$selector" in
        =*) exact=1; selector=${selector#=} ;;
    esac

    local -a raw_tokens=()
    IFS=, read -r -a raw_tokens <<< "$selector"
    for token in "${raw_tokens[@]}"; do
        name=${token%%:*}
        [ -n "$name" ] || continue
        if [ "$token_count" -ge 32 ]; then
            selector_truncated=1
            continue
        fi
        LC_ALL=C printf -v name '%.63s' "$name"
        port=-1
        if [[ $token == *:* ]]; then
            port_text=${token#*:}
            port_text=${port_text%%:*}
            [ -z "$port_text" ] || port=$(_sparkring_nccl_atoi_port "$port_text")
        fi
        token_names+=("$name")
        token_ports+=("$port")
        token_count=$((token_count + 1))
    done
    [ "$selector_truncated" = 0 ] || printf \
        '  GID policy note: NCCL uses only the first 32 non-empty NCCL_IB_HCA entries and ignores later entries.\n' >&2

    for dev_dir in "$sysfs_root"/*; do
        [ -d "$dev_dir/ports" ] || continue
        dev=${dev_dir##*/}
        for port_dir in "$dev_dir"/ports/*; do
            [ -d "$port_dir" ] || continue
            port=${port_dir##*/}
            state=
            if [ -r "$port_dir/state" ]; then
                state=$(<"$port_dir/state")
                state=${state#*: }
            fi
            case "$state" in
                ''|ACTIVE) ;;
                *) continue ;;
            esac
            link_layer=
            [ ! -r "$port_dir/link_layer" ] || link_layer=$(<"$port_dir/link_layer")
            case "$link_layer" in
                ''|Ethernet|InfiniBand) ;;
                *) continue ;;
            esac

            candidate_match=0
            if [ "$token_count" -eq 0 ]; then
                candidate_match=1
            else
                local index
                for ((index = 0; index < token_count; index++)); do
                    if [ "$exact" = 1 ]; then
                        [ "$dev" = "${token_names[index]}" ] || continue
                    else
                        case "$dev" in
                            "${token_names[index]}"*) ;;
                            *) continue ;;
                        esac
                    fi
                    [ "${token_ports[index]}" = -1 ] \
                        || [ "${token_ports[index]}" = "$port" ] || continue
                    candidate_match=1
                    break
                done
            fi
            [ "$candidate_match" -ne "$exclude" ] || continue
            if [ "$selected_count" -ge 32 ]; then
                printf '  GID policy note: NCCL ignores selected member %s:%s after its 32-device cap.\n' \
                    "$dev" "$port" >&2
                continue
            fi
            selected_devs+=("$dev")
            selected_ports+=("$port")
            selected_count=$((selected_count + 1))
        done
    done

    SPARKRING_NCCL_SELECTED_COUNT=$selected_count
    if [ "$selected_count" -eq 0 ]; then
        printf '  GID policy: selector %s matched no active RDMA HCA/port under %s.\n' \
            "$selector_text" "$sysfs_root" >&2
        return 1
    fi

    local preferred_suffix= preferred_ip=${VLLM_HOST_IP-}
    preferred_suffix=$(_sparkring_ipv4_gid_suffix "$preferred_ip" 2>/dev/null || true)
    local member_index gid_file gid_index gid_type gid netdev gid_suffix
    local own_ip own_suffix usable usable_text roce_v2_text failures=0
    for ((member_index = 0; member_index < selected_count; member_index++)); do
        dev=${selected_devs[member_index]}
        port=${selected_ports[member_index]}
        port_dir="$sysfs_root/$dev/ports/$port"
        usable_text=
        roce_v2_text=
        for gid_file in "$port_dir"/gids/*; do
            [ -f "$gid_file" ] || continue
            gid_index=${gid_file##*/}
            case "$gid_index" in
                ''|*[!0-9]*) continue ;;
            esac
            gid_type=
            [ ! -r "$port_dir/gid_attrs/types/$gid_index" ] \
                || gid_type=$(<"$port_dir/gid_attrs/types/$gid_index")
            [ "$gid_type" = 'RoCE v2' ] || continue
            roce_v2_text=${roce_v2_text:+$roce_v2_text,}$gid_index
            gid=$(<"$gid_file")
            gid=${gid,,}
            if [[ $gid =~ ^0000:0000:0000:0000:0000:ffff:([[:xdigit:]]{4}):([[:xdigit:]]{4})$ ]]; then
                gid_suffix=${BASH_REMATCH[1],,}:${BASH_REMATCH[2],,}
            else
                continue
            fi
            usable=0
            [ -z "$preferred_suffix" ] || [ "$gid_suffix" != "$preferred_suffix" ] \
                || usable=1
            netdev=
            [ ! -r "$port_dir/gid_attrs/ndevs/$gid_index" ] \
                || netdev=$(<"$port_dir/gid_attrs/ndevs/$gid_index")
            if [ -n "$netdev" ]; then
                while IFS= read -r own_ip; do
                    [ -n "$own_ip" ] || continue
                    own_suffix=$(_sparkring_ipv4_gid_suffix "$own_ip" 2>/dev/null || true)
                    [ -z "$own_suffix" ] || [ "$gid_suffix" != "$own_suffix" ] \
                        || usable=1
                done < <(_sparkring_netdev_ipv4s "$netdev" || true)
            fi
            [ "$usable" = 1 ] || continue
            usable_text=${usable_text:+$usable_text,}$gid_index
        done
        if [ -n "$usable_text" ]; then
            printf '  GID policy: %s:%s usable RoCEv2/IPv4 indexes: %s\n' \
                "$dev" "$port" "$usable_text"
        else
            printf '  GID policy: %s:%s usable RoCEv2/IPv4 indexes: none (RoCE v2 indexes seen: %s).\n' \
                "$dev" "$port" "${roce_v2_text:-none}" >&2
            failures=$((failures + 1))
        fi
    done
    [ "$failures" -eq 0 ]
}

sparkring_validate_nccl_gid_policy() {
    NCCL_IB_GID_AUTO=${NCCL_IB_GID_AUTO:-1}
    case "$NCCL_IB_GID_AUTO" in
        0)
            case "${NCCL_IB_GID_INDEX-}" in
                ''|*[!0-9]*) die 'NCCL_IB_GID_AUTO=0 requires a decimal NCCL_IB_GID_INDEX' ;;
            esac
            SPARKRING_NCCL_SELECTED_COUNT=0
            printf '  GID policy: pinned NCCL_IB_GID_INDEX=%s (automatic validation disabled).\n' \
                "$NCCL_IB_GID_INDEX"
            ;;
        1)
            local configured_index=${NCCL_IB_GID_INDEX-}
            _sparkring_validate_selected_gids \
                || die 'automatic RoCEv2 GID validation failed'
            [ -z "$configured_index" ] || printf \
                '  GID policy: ignoring configured NCCL_IB_GID_INDEX=%s because automatic mode is enabled.\n' \
                "$configured_index"
            unset NCCL_IB_GID_INDEX
            printf '  GID policy: automatic validation passed; NCCL_IB_GID_INDEX will be unset in the container.\n'
            ;;
        *) die "NCCL_IB_GID_AUTO must be 0 or 1: $NCCL_IB_GID_AUTO" ;;
    esac
}
