# Routed-QSFP NCCL Socket bootstrap

`scripts/routed_qsfp_nccl_bootstrap.py` preflights and, with an exact
confirmation, makes the four-rank cross-half routes reboot-recoverable. The
default mode is read-only. It never launches NCCL or a model.

## Exact topology

Fabric addresses below use the `<SUBNET_xx>` placeholders defined in
`../docs/SETUP.md` Section 1: four **distinct** /24 prefixes, one per direct
cable (`<SUBNET_01>` = rank 0-1 link, `<SUBNET_12>` = rank 1-2,
`<SUBNET_23>` = rank 2-3, `<SUBNET_30>` = rank 0-3). Substitute your site's
real prefixes; the host suffixes and route relationships must be kept exactly
as written.

The only routes managed are:

| Rank | Destination | Via | Device | Preferred source |
|---|---|---|---|---|
| 0 | `<SUBNET_23>.0/24` | `<SUBNET_30>.13` | `enp1s0f1np1` | `<SUBNET_01>.10` |
| 1 | `<SUBNET_23>.0/24` | `<SUBNET_12>.11` | `enp1s0f1np1` | `<SUBNET_01>.11` |
| 2 | `<SUBNET_01>.0/24` | `<SUBNET_12>.10` | `enp1s0f1np1` | `<SUBNET_23>.11` |
| 3 | `<SUBNET_01>.0/24` | `<SUBNET_30>.12` | `enp1s0f1np1` | `<SUBNET_23>.10` |

Runtime and boot state require `net.ipv4.ip_forward=1` and loose reverse-path
filtering (`rp_filter=2`) for `all`, `default`, `enp1s0f0np0`, and
`enp1s0f1np1`.

Each rank retains only the forwarding rule its topology needs:

- ranks 0 and 1: `f1 -> f0`, source `<SUBNET_23>.0/24`, destination
  `<SUBNET_01>.0/24`;
- ranks 2 and 3: `f1 -> f0`, source `<SUBNET_01>.0/24`, destination
  `<SUBNET_23>.0/24`.

Every managed rule has comment `sparkring-cx7-tcp-relay`. The tool never
flushes a table, changes a chain policy, or deletes an untagged/unknown rule.

The live bootstrap initially used both tagged directions on every rank. The
diagnostic JSON reports those opposite-direction rules as
`overbroad_live_rules`. Apply removes only that exact tagged opposite rule,
after all four hosts pass identity, route-conflict, interface-address, and
independent-management gates. Rollback restores the exact prior zero/one rule
counts, including the symmetric pair when it existed before Apply.

## Diagnose

Use only Wi-Fi, USB Ethernet, or Tailscale management targets:

```powershell
python .\scripts\routed_qsfp_nccl_bootstrap.py `
  --rank0 user@RANK0_MANAGEMENT `
  --rank1 user@RANK1_MANAGEMENT `
  --rank2 user@RANK2_USB_WIFI_OR_TAILSCALE `
  --rank3 user@RANK3_MANAGEMENT
```

The helper derives the controller address from `SSH_CONNECTION` and rejects
any session whose return route uses either ConnectX-7 fabric interface. It
snapshots routes, addresses, forwarding sysctls, exact tagged rules, managed
file contents, systemd enable/active state, and two sourced pings per rank.
The eight required pairs are:

- `<SUBNET_01>.10` and `<SUBNET_01>.11` to each of `<SUBNET_23>.11` and
  `<SUBNET_23>.10`;
- `<SUBNET_23>.11` and `<SUBNET_23>.10` to each of `<SUBNET_01>.10` and
  `<SUBNET_01>.11`.

One versioned `bootstrap.json` is written under a unique UTC result directory.

## Apply

```powershell
python .\scripts\routed_qsfp_nccl_bootstrap.py `
  --rank0 user@RANK0_MANAGEMENT `
  --rank1 user@RANK1_MANAGEMENT `
  --rank2 user@RANK2_USB_WIFI_OR_TAILSCALE `
  --rank3 user@RANK3_MANAGEMENT `
  --mode apply `
  --confirm 'APPLY 4-RANK ROUTED-QSFP NCCL SOCKET BOOTSTRAP'
```

Apply runs the four rank operations together and is idempotent:

1. install the exact host-specific
   `/usr/local/sbin/sparkring-cx7-tcp-relay`;
2. install and enable `sparkring-cx7-tcp-relay.service`, ordered after network
   online;
3. use `ip route replace` for only the table entries above;
4. set only the named forwarding/rp-filter sysctls;
5. remove the exact tagged opposite rule and ensure one role-appropriate rule;
6. resnapshot all ranks and require persistence plus all eight sourced pings.

Any partial apply or failed final gate invokes rollback on all four ranks.
Rollback restores the prior managed files, service enabled/active state,
missing/exact route state, sysctl values, and exact prior tagged rule counts.
The result distinguishes complete rollback from rollback failure.

## Offline tests

```powershell
python -m unittest scripts.test_routed_qsfp_nccl_bootstrap -v
```

The tests inject the SSH boundary and do not contact a Spark. They cover the
read-only default, exact confirmation, management-path failure, live
overbroad-rule detection, convergent Apply, eight-ping verification, JSON
evidence, and four-rank rollback after partial failure.
